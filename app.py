# ============================================================================
# TERRATRAQ - ROAD CONDITION PREDICTION SYSTEM
# Flask Web Application with Admin Panel
# ============================================================================

import os
import sys
import json
import pickle
import shutil
import logging
import numpy as np
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
from flask import (
    Flask, request, render_template, jsonify, redirect, url_for,
    session, flash, send_file
)
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson.objectid import ObjectId
from bson.errors import InvalidId
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me-in-production')

# MongoDB configuration
# Connection string from environment (MongoDB Atlas): mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://localhost:27017'
MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME') or 'terratraq'

# Upload configuration
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 256MB max (allows model uploads)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Model configuration
MODEL_PATH = os.path.join(app.root_path, 'model', 'model_final.h5')
CLASS_NAMES_PATH = os.path.join(app.root_path, 'model', 'class_names.pkl')
IMG_SIZE = (224, 224)

# Admin configuration (from .env, fall back to defaults)
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')

# Dataset storage for future retraining
DATASET_FOLDER = os.path.join(app.root_path, 'datasets')

# Fixed description of the dataset the model was trained on.
# The full dataset is too large to store in the repo, so this is written
# instead of computed from disk.
TRAINING_DATASET_INFO = "41,137 images (30,105 train / 5,865 val / 5,167 test)"

# Model backups
BACKUP_FOLDER = os.path.join(app.root_path, 'model', 'backups')
BACKUP_KEEP = 8

# Initialize MongoDB
mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = mongo[MONGO_DB_NAME]
users_col = db['users']
predictions_col = db['predictions']

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(BACKUP_FOLDER, exist_ok=True)
os.makedirs(app.instance_path, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

LOG_FILE = os.path.join(app.instance_path, 'app.log')

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_event(message):
    logging.info(message)

# ============================================================================
# DATABASE MODELS
# ============================================================================

def to_oid(value):
    """Safely convert a string/bytes id to a bson ObjectId (or None)."""
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None

class User:
    """Wrapper around a MongoDB 'users' document (field access kept SQLAlchemy-style)."""

    def __init__(self, doc):
        self.doc = doc
        self.id = str(doc['_id'])
        self.username = doc.get('username')
        self.password_hash = doc.get('password_hash')
        self.role = doc.get('role', 'user')
        self.created_at = doc.get('created_at')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'role': self.role,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }

    @classmethod
    def get(cls, user_id):
        oid = to_oid(user_id)
        if oid is None:
            return None
        doc = users_col.find_one({'_id': oid})
        return cls(doc) if doc else None

    @classmethod
    def by_username(cls, username):
        doc = users_col.find_one({'username': username})
        return cls(doc) if doc else None

    @classmethod
    def first_admin(cls):
        doc = users_col.find_one({'role': 'admin'})
        return cls(doc) if doc else None

    @classmethod
    def all(cls):
        return [cls(doc) for doc in users_col.find()]

    @classmethod
    def all_ordered(cls):
        return [cls(doc) for doc in users_col.find().sort('created_at', DESCENDING)]

    @classmethod
    def count(cls):
        return users_col.count_documents({})

    @classmethod
    def admin_count(cls):
        return users_col.count_documents({'role': 'admin'})

    @classmethod
    def create(cls, username, role='user', password=None):
        doc = {
            'username': username,
            'role': role,
            'created_at': datetime.utcnow(),
            'password_hash': generate_password_hash(password) if password else None
        }
        doc['_id'] = users_col.insert_one(doc).inserted_id
        return cls(doc)

    def update_role(self, new_role):
        self.role = new_role
        self.doc['role'] = new_role
        users_col.update_one({'_id': self.doc['_id']}, {'$set': {'role': new_role}})

    def delete(self):
        # Remove the user's predictions first to avoid orphaned records
        predictions_col.delete_many({'user_id': self.id})
        users_col.delete_one({'_id': self.doc['_id']})

class Prediction:
    """Wrapper around a MongoDB 'predictions' document (field access kept SQLAlchemy-style)."""

    def __init__(self, doc):
        self.doc = doc
        self.id = str(doc['_id'])
        self.user_id = doc.get('user_id')
        self.image_path = doc.get('image_path')
        self.filename = doc.get('filename')
        self.predicted_class = doc.get('predicted_class')
        self.confidence = doc.get('confidence')
        self.timestamp = doc.get('timestamp')

    @property
    def user(self):
        return User.get(self.user_id) if self.user_id else None

    def to_dict(self):
        user = self.user
        return {
            'id': self.id,
            'filename': self.filename,
            'predicted_class': self.predicted_class,
            'confidence': f"{self.confidence:.2f}%",
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S') if self.timestamp else None,
            'username': user.username if user else None
        }

    @classmethod
    def create(cls, **kwargs):
        doc = {'timestamp': datetime.utcnow(), **kwargs}
        doc['_id'] = predictions_col.insert_one(doc).inserted_id
        return cls(doc)

    @classmethod
    def get(cls, prediction_id):
        oid = to_oid(prediction_id)
        if oid is None:
            return None
        doc = predictions_col.find_one({'_id': oid})
        return cls(doc) if doc else None

    @classmethod
    def list(cls, user_id=None):
        query = {'user_id': user_id} if user_id else {}
        return [cls(doc) for doc in predictions_col.find(query).sort('timestamp', DESCENDING)]

    @classmethod
    def latest(cls, user_id=None):
        query = {'user_id': user_id} if user_id else {}
        doc = predictions_col.find_one(query, sort=[('timestamp', DESCENDING)])
        return cls(doc) if doc else None

    @classmethod
    def count(cls, user_id=None):
        query = {'user_id': user_id} if user_id else {}
        return predictions_col.count_documents(query)

    @classmethod
    def class_counts(cls, user_id=None):
        query = {'user_id': user_id} if user_id else {}
        pipeline = [
            {'$match': query},
            {'$group': {'_id': '$predicted_class', 'count': {'$sum': 1}}}
        ]
        return {doc['_id']: doc['count'] for doc in predictions_col.aggregate(pipeline)}

    @classmethod
    def delete_all(cls, user_id=None):
        query = {'user_id': user_id} if user_id else {}
        predictions_col.delete_many(query)

    def delete(self):
        predictions_col.delete_one({'_id': self.doc['_id']})

def seed_admin():
    """Create the default admin account if no admin exists."""
    if User.first_admin() is None:
        User.create(username=ADMIN_USERNAME, role='admin', password=ADMIN_PASSWORD)
        log_event(f"Seeded default admin user: {ADMIN_USERNAME}")

# Verify connection, create indexes, and seed admin on startup
try:
    mongo.admin.command('ping')
    users_col.create_index([('username', ASCENDING)], unique=True)
    predictions_col.create_index([('user_id', ASCENDING), ('timestamp', DESCENDING)])
    seed_admin()
    print(f"MongoDB connected. Database: '{MONGO_DB_NAME}'")
except Exception as e:
    print("[FATAL] Could not connect to MongoDB.")
    print("        Set MONGO_URI in .env to your MongoDB Atlas connection string")
    print("        (or start a local MongoDB) and restart the app.")
    print(f"        Error: {e}")
    sys.exit(1)

# ============================================================================
# LOAD MODEL
# ============================================================================

# Global variables for model
model = None
class_names = None

def load_model_and_classes():
    global model, class_names

    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            print(f"Model loaded from: {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print(f"Model not found at: {MODEL_PATH}")
        model = None

    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, 'rb') as f:
                class_names = pickle.load(f)
            print(f"Class names loaded: {class_names}")
        except Exception as e:
            print(f"Error loading class names: {e}")
            class_names = None
    else:
        print(f"Class names not found at: {CLASS_NAMES_PATH}")
        class_names = None

    return model, class_names

# Load model on startup
load_model_and_classes()

# ============================================================================
# AUTH HELPERS
# ============================================================================

def current_user():
    """Return the logged-in User object, or None."""
    user_id = session.get('user_id')
    if user_id is None:
        return None
    return User.get(user_id)

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if current_user() is None:
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = current_user()
        if user is None:
            return redirect(url_for('login', next=request.path))
        if user.role != 'admin':
            return render_template('error.html', error="Admin access required."), 403
        return f(*args, **kwargs)
    return wrapper

@app.context_processor
def inject_current_user():
    return {'current_user': current_user(), 'active_page': request.endpoint or ''}

# ============================================================================
# GENERAL HELPERS
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024 or unit == 'GB':
            return f"{size:.2f} {unit}" if unit != 'B' else f"{size} B"
        size /= 1024

def get_dir_size(path):
    if not os.path.exists(path):
        return 0
    return sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(path) for f in filenames
    )

def get_model_backups(limit=5):
    """Return the most recent model backups (as .h5 + matching class_names.pkl pairs)."""
    if not os.path.isdir(BACKUP_FOLDER):
        return []
    files = [f for f in os.listdir(BACKUP_FOLDER)
             if f.startswith('model_final_') and f.endswith('.h5')]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(BACKUP_FOLDER, f)), reverse=True)
    backups = []
    for f in files[:limit]:
        ts = f[len('model_final_'):-len('.h5')]
        partner = f"class_names_{ts}.pkl"
        full = os.path.join(BACKUP_FOLDER, f)
        if os.path.exists(os.path.join(BACKUP_FOLDER, partner)):
            backups.append({
                'timestamp': ts,
                'model_size': os.path.getsize(full),
                'date': datetime.fromtimestamp(
                    os.path.getmtime(full)).strftime('%Y-%m-%d %H:%M:%S'),
            })
    return backups

def prune_model_backups(max_count=BACKUP_KEEP):
    """Delete the oldest model backups beyond max_count."""
    if not os.path.isdir(BACKUP_FOLDER):
        return
    files = sorted(
        (f for f in os.listdir(BACKUP_FOLDER)
         if f.startswith('model_final_') and f.endswith('.h5')),
        key=lambda f: os.path.getmtime(os.path.join(BACKUP_FOLDER, f))
    )
    for f in files[:-max_count]:
        ts = f[len('model_final_'):-len('.h5')]
        for name in (f, f"class_names_{ts}.pkl"):
            p = os.path.join(BACKUP_FOLDER, name)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

def backup_current_model():
    """Snapshot the current live model files into the backups folder."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        return
    os.makedirs(BACKUP_FOLDER, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    shutil.copy2(MODEL_PATH, os.path.join(BACKUP_FOLDER, f"model_final_{ts}.h5"))
    shutil.copy2(CLASS_NAMES_PATH, os.path.join(BACKUP_FOLDER, f"class_names_{ts}.pkl"))
    prune_model_backups()

def predict_image(image_path):
    """Make prediction on a single image, returning class, confidence, and all probabilities"""
    if model is None or class_names is None:
        return None, None, None

    try:
        # Load and preprocess image
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx] * 100)
        predicted_class = class_names[predicted_idx]
        probabilities = [float(p * 100) for p in predictions]

        return predicted_class, confidence, probabilities
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None

# ============================================================================
# APPLICATION ROUTES (login required)
# ============================================================================

def visible_user_id():
    """Return the user_id filter for predictions visible to the current user
    (None = all predictions, i.e. admin)."""
    user = current_user()
    if user is None:
        return None
    return None if user.role == 'admin' else user.id

def can_view(prediction, user):
    """Whether a user is allowed to see a prediction (owner or admin)."""
    if user is None:
        return False
    return user.role == 'admin' or prediction.user_id == user.id

def get_mongo_size():
    """Return MongoDB database data size in bytes (best effort)."""
    try:
        return int(db.command('dbStats').get('dataSize', 0))
    except Exception:
        return 0

@app.route('/')
def index():
    """Landing page for visitors; dashboard for logged-in users."""
    if current_user() is None:
        return render_template('landing.html')
    return redirect(url_for('dashboard'))

@app.route('/health')
def health():
    """Lightweight health check for uptime monitors / keep-alive cron jobs."""
    db_ok = True
    try:
        mongo.admin.command('ping')
    except Exception:
        db_ok = False
    return jsonify({'status': 'ok' if db_ok else 'degraded', 'database': db_ok}), 200 if db_ok else 503

@app.route('/dashboard')
@login_required
def dashboard():
    """Home dashboard with stats and recent activity."""
    user = current_user()
    is_admin = user.role == 'admin'
    uid = visible_user_id()

    total = Prediction.count(uid)
    recent = Prediction.list(uid)[:5]

    class_counts = Prediction.class_counts(uid)
    most_common = max(class_counts, key=class_counts.get) if class_counts else None

    stats = {
        'total': total,
        'most_common': most_common,
        'most_common_count': class_counts.get(most_common, 0) if most_common else 0,
        'class_distribution': class_counts,
    }

    if is_admin:
        stats.update({
            'total_users': User.count(),
            'uploads_size': get_dir_size(app.config['UPLOAD_FOLDER']),
            'db_size': get_mongo_size(),
            'datasets_size': TRAINING_DATASET_INFO,
            'model_loaded': model is not None,
            'model_size': os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0,
            'classes': class_names,
        })

    return render_template('dashboard.html', s=stats, recent=recent, is_admin=is_admin)

@app.route('/upload', methods=['GET'])
@login_required
def upload():
    """Upload page - image upload form."""
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handle image upload and prediction"""
    # Check if model is loaded
    if model is None or class_names is None:
        return render_template('error.html',
                             error="Model not loaded. Please ensure model files are present.")

    # Check if image was uploaded
    if 'image' not in request.files:
        return render_template('error.html', error="No image uploaded.")

    file = request.files['image']

    if file.filename == '':
        return render_template('error.html', error="No file selected.")

    if not allowed_file(file.filename):
        return render_template('error.html', error="File type not allowed. Please upload JPG, PNG, or GIF.")

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(file_path)

        # Make prediction
        predicted_class, confidence, probabilities = predict_image(file_path)

        if predicted_class is None:
            os.remove(file_path)  # Clean up
            return render_template('error.html', error="Prediction failed. Please try again.")

        # Save to database (tied to the logged-in user)
        prediction = Prediction.create(
            user_id=current_user().id,
            image_path=file_path,
            filename=saved_filename,
            predicted_class=predicted_class,
            confidence=confidence
        )
        log_event(f"Prediction saved: {saved_filename} -> {predicted_class} ({confidence:.2f}%)")

        # Redirect to the result page
        return redirect(url_for('result_view', prediction_id=prediction.id))

    except Exception as e:
        print(f"Error: {e}")
        return render_template('error.html', error=f"An error occurred: {str(e)}")

@app.route('/result')
@login_required
def result_latest():
    """Show the current user's most recent prediction result."""
    prediction = Prediction.latest(visible_user_id())
    if prediction is None:
        flash('No predictions yet. Upload a road image to get started.', 'info')
        return redirect(url_for('upload'))
    return redirect(url_for('result_view', prediction_id=prediction.id))

@app.route('/result/<prediction_id>')
@login_required
def result_view(prediction_id):
    """Show a single prediction result."""
    prediction = Prediction.get(prediction_id)
    if prediction is None or not can_view(prediction, current_user()):
        return render_template('error.html', error="Prediction not found."), 404

    probabilities = None
    class_probabilities = []
    if class_names and os.path.exists(prediction.image_path):
        _, _, probabilities = predict_image(prediction.image_path)
        if probabilities:
            class_probabilities = list(zip(class_names, probabilities))

    return render_template('result.html',
                         prediction=prediction,
                         filename=prediction.filename,
                         confidence=f"{prediction.confidence:.2f}%",
                         confidence_value=prediction.confidence,
                         class_names=class_names,
                         probabilities=probabilities,
                         class_probabilities=class_probabilities)

@app.route('/history')
@login_required
def history():
    """Show prediction history (own for users, all for admins)."""
    is_admin = current_user().role == 'admin'
    predictions = Prediction.list(visible_user_id())

    stats = {'total': len(predictions), 'classes': 0, 'avg_conf': 0.0, 'most_common': None}
    if predictions:
        classes = [p.predicted_class for p in predictions]
        stats['classes'] = len(set(classes))
        stats['avg_conf'] = sum(p.confidence for p in predictions) / len(predictions)
        stats['most_common'] = max(set(classes), key=classes.count)

    return render_template('history.html', predictions=predictions, is_admin=is_admin, stats=stats)

@app.route('/about')
@login_required
def about():
    """About page"""
    return render_template('about.html')

@app.route('/delete/<prediction_id>', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    """Delete a prediction (owner or admin)."""
    prediction = Prediction.get(prediction_id)
    if prediction is None or not can_view(prediction, current_user()):
        flash('Prediction not found.', 'warning')
        return redirect(url_for('history'))

    # Delete the image file
    if os.path.exists(prediction.image_path):
        try:
            os.remove(prediction.image_path)
        except:
            pass

    prediction.delete()
    log_event(f"Prediction deleted: {prediction.id}")
    return redirect(url_for('history'))

@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    """Clear prediction history visible to the current user."""
    uid = visible_user_id()
    predictions = Prediction.list(uid)
    for p in predictions:
        if os.path.exists(p.image_path):
            try:
                os.remove(p.image_path)
            except:
                pass

    Prediction.delete_all(uid)
    log_event(f"Prediction history cleared by {current_user().username}")
    return redirect(url_for('history'))

# ============================================================================
# REST API
# ============================================================================

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """REST API endpoint for prediction"""
    if model is None or class_names is None:
        return jsonify({'error': 'Model not loaded'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(file_path)

        predicted_class, confidence, probabilities = predict_image(file_path)
        os.remove(file_path)  # Clean up

        if predicted_class is None:
            return jsonify({'error': 'Prediction failed'}), 500

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'confidence_value': confidence,
            'probabilities': probabilities,
            'class_names': class_names
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
@login_required
def api_history():
    """REST API endpoint for history (own for users, all for admins)"""
    predictions = Prediction.list(visible_user_id())[:50]
    return jsonify([p.to_dict() for p in predictions])

@app.route('/api/users')
@admin_required
def api_users():
    """REST API endpoint for users (admin only)"""
    return jsonify([u.to_dict() for u in User.all()])

# ============================================================================
# AUTH ROUTES
# ============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.by_username(username)
        if user and user.check_password(password):
            session['user_id'] = user.id
            log_event(f"User logged in: {user.username}")
            flash('Logged in successfully.', 'success')
            nxt = request.args.get('next')
            if nxt and nxt.startswith('/'):
                return redirect(nxt)
            return redirect(url_for('dashboard'))
        log_event(f"Failed login attempt for username: {username}")
        return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout"""
    user = current_user()
    if user:
        log_event(f"User logged out: {user.username}")
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')

        if len(username) < 3:
            return render_template('register.html', error="Username must be at least 3 characters.")
        if len(password) < 6:
            return render_template('register.html', error="Password must be at least 6 characters.")
        if password != confirm:
            return render_template('register.html', error="Passwords do not match.")
        if User.by_username(username):
            return render_template('register.html', error="Username already exists.")

        user = User.create(username=username, role='user', password=password)
        session['user_id'] = user.id
        log_event(f"New user registered and logged in: {username}")
        flash('Account created. Welcome to Terratraq!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('register.html')

# ============================================================================
# ADMIN ROUTES
# ============================================================================

@app.route('/admin')
@admin_required
def admin_dashboard():
    """Admin dashboard with system performance monitoring"""
    class_counts = Prediction.class_counts()
    stats = {
        'total_predictions': Prediction.count(),
        'total_users': User.count(),
        'admin_count': User.admin_count(),
        'class_counts': class_counts,
        'uploads_size': get_dir_size(app.config['UPLOAD_FOLDER']),
        'db_size': get_mongo_size(),
        'datasets_size': TRAINING_DATASET_INFO,
        'model_loaded': model is not None,
        'model_size': os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0,
        'classes': class_names,
        'python_version': sys.version.split()[0],
        'tensorflow_version': __import__('tensorflow').__version__,
        'now': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    return render_template('admin/dashboard.html', s=stats)

@app.route('/admin/users')
@admin_required
def admin_users():
    """Manage users"""
    users = User.all_ordered()
    return render_template('admin/users.html', users=users)

@app.route('/admin/users/<user_id>/delete', methods=['POST'])
@admin_required
def admin_user_delete(user_id):
    """Delete a user"""
    user = User.get(user_id)
    if not user:
        flash('User not found.', 'warning')
        return redirect(url_for('admin_users'))
    if user.id == current_user().id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin_users'))
    user.delete()
    log_event(f"Admin deleted user: {user.username}")
    flash(f"User '{user.username}' deleted.", 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/users/<user_id>/role', methods=['POST'])
@admin_required
def admin_user_role(user_id):
    """Change a user's role (promote/demote)"""
    user = User.get(user_id)
    new_role = request.form.get('role')
    if not user or new_role not in ('admin', 'user'):
        flash('Invalid request.', 'warning')
        return redirect(url_for('admin_users'))
    if new_role == 'user' and user.id == current_user().id:
        if User.admin_count() <= 1:
            flash('Cannot demote the last admin account.', 'danger')
            return redirect(url_for('admin_users'))
    old_role = user.role
    user.update_role(new_role)
    log_event(f"Admin changed role: {user.username} {old_role} -> {new_role}")
    flash(f"'{user.username}' role changed to {new_role}.", 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/model')
@admin_required
def admin_model():
    """Redirect to the combined model training/update page."""
    return redirect(url_for('admin_retrain'))

@app.route('/admin/model/restore/<timestamp>', methods=['POST'])
@admin_required
def admin_model_restore(timestamp):
    """Restore the live model from a saved backup."""
    if not timestamp or not all(c.isalnum() or c in ('_', '-') for c in timestamp):
        flash('Invalid backup.', 'warning')
        return redirect(url_for('admin_retrain'))
    h5_path = os.path.join(BACKUP_FOLDER, f"model_final_{timestamp}.h5")
    pkl_path = os.path.join(BACKUP_FOLDER, f"class_names_{timestamp}.pkl")
    if not os.path.exists(h5_path) or not os.path.exists(pkl_path):
        flash('Backup not found.', 'warning')
        return redirect(url_for('admin_retrain'))
    try:
        with open(pkl_path, 'rb') as f:
            new_classes = pickle.load(f)
        new_model = load_model(h5_path)
        if not isinstance(new_classes, (list, tuple)):
            raise ValueError("class_names.pkl must contain a list of class labels.")
        if new_model.output_shape[-1] != len(new_classes):
            raise ValueError(
                f"Model outputs {new_model.output_shape[-1]} classes "
                f"but class_names.pkl has {len(new_classes)}."
            )
        global model, class_names
        shutil.copyfile(h5_path, MODEL_PATH)
        shutil.copyfile(pkl_path, CLASS_NAMES_PATH)
        model = new_model
        class_names = new_classes
        log_event(f"Model restored from backup {timestamp} by admin")
        flash(f"Model restored from backup ({timestamp}).", 'success')
    except Exception as e:
        log_event(f"Model restore FAILED: {e}")
        flash(f"Restore failed: {e}", 'danger')
    return redirect(url_for('admin_retrain'))

@app.route('/admin/datasets', methods=['GET', 'POST'])
@admin_required
def admin_datasets():
    """Upload dataset files for future retraining"""
    message = None
    error = None

    if request.method == 'POST':
        files = request.files.getlist('dataset_files')
        uploaded = 0
        for f in files:
            if f and f.filename != '':
                fname = secure_filename(f.filename)
                f.save(os.path.join(DATASET_FOLDER, fname))
                uploaded += 1
        if uploaded:
            log_event(f"Admin uploaded {uploaded} dataset file(s)")
            message = f"Uploaded {uploaded} file(s) to datasets/."
        else:
            error = "No files selected."

    items = []
    for root, _, filenames in os.walk(DATASET_FOLDER):
        for fname in filenames:
            full = os.path.join(root, fname)
            items.append({
                'name': os.path.relpath(full, DATASET_FOLDER),
                'size': format_bytes(os.path.getsize(full))
            })

    return render_template('admin/datasets.html', items=items, message=message, error=error)

@app.route('/admin/datasets/delete', methods=['POST'])
@admin_required
def admin_dataset_delete():
    """Delete a single dataset file"""
    name = request.form.get('name', '')
    if not name:
        flash('No file specified.', 'warning')
        return redirect(url_for('admin_datasets'))
    full = os.path.join(DATASET_FOLDER, os.path.basename(name))
    if os.path.exists(full):
        os.remove(full)
        log_event(f"Admin deleted dataset file: {os.path.basename(name)}")
        flash(f"Deleted {os.path.basename(name)}.", 'success')
    return redirect(url_for('admin_datasets'))

@app.route('/admin/retrain', methods=['GET', 'POST'])
@admin_required
def admin_retrain():
    """Training guide + upload/update the CNN model without restarting"""
    message = None
    error = None

    if request.method == 'POST':
        model_file = request.files.get('model_file')
        classes_file = request.files.get('classes_file')

        if not model_file or model_file.filename == '' or not classes_file or classes_file.filename == '':
            error = "Please provide both the model (.h5) and class names (.pkl) files."
        else:
            if not model_file.filename.endswith('.h5'):
                error = "Model file must be a .h5 file."
            elif not classes_file.filename.endswith('.pkl'):
                error = "Class names file must be a .pkl file."
            else:
                tmp_model = os.path.join(app.root_path, 'model', '_new_model.h5')
                tmp_classes = os.path.join(app.root_path, 'model', '_new_classes.pkl')
                model_file.save(tmp_model)
                classes_file.save(tmp_classes)
                try:
                    with open(tmp_classes, 'rb') as f:
                        new_classes = pickle.load(f)
                    new_model = load_model(tmp_model)
                    if not isinstance(new_classes, (list, tuple)):
                        raise ValueError("class_names.pkl must contain a list of class labels.")
                    if new_model.output_shape[-1] != len(new_classes):
                        raise ValueError(
                            f"Model outputs {new_model.output_shape[-1]} classes "
                            f"but class_names.pkl has {len(new_classes)}."
                        )
                    global model, class_names
                    backup_current_model()
                    os.replace(tmp_model, MODEL_PATH)
                    os.replace(tmp_classes, CLASS_NAMES_PATH)
                    model = new_model
                    class_names = new_classes
                    log_event(f"Model updated by admin. Classes: {class_names}")
                    message = "Model updated and reloaded successfully."
                except Exception as e:
                    for f in (tmp_model, tmp_classes):
                        if os.path.exists(f):
                            try:
                                os.remove(f)
                            except OSError:
                                pass
                    log_event(f"Model update FAILED: {e}")
                    error = f"Failed to load new model: {e}"

    return render_template('admin/retrain.html',
                         message=message,
                         error=error,
                         backups=get_model_backups())

@app.route('/admin/settings')
@admin_required
def admin_settings():
    """System settings - consolidated system info and model artifacts."""
    history = None
    history_path = os.path.join(app.root_path, 'model', 'training_history.json')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception as e:
            print(f"Error loading training history: {e}")
    info = {
        'uploads_size': get_dir_size(app.config['UPLOAD_FOLDER']),
        'datasets_size': TRAINING_DATASET_INFO,
        'model_loaded': model is not None,
        'model_size': os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0,
        'model_file': os.path.basename(MODEL_PATH),
        'classes_file': os.path.basename(CLASS_NAMES_PATH),
        'model_classes': class_names,
        'python_version': sys.version.split()[0],
        'tensorflow_version': __import__('tensorflow').__version__,
        'admin_username': ADMIN_USERNAME,
        'has_confusion': os.path.exists(os.path.join(app.root_path, 'model', 'confusion_matrix.png')),
        'has_history': os.path.exists(history_path),
        'has_history_img': os.path.exists(os.path.join(app.root_path, 'model', 'training_history.png')),
    }
    return render_template('admin/settings.html', info=info, history=history)

@app.route('/admin/model/download/<kind>')
@admin_required
def admin_model_download(kind):
    """Download the trained model file, class names, or training artifacts (admin only)."""
    mapping = {
        'model': MODEL_PATH,
        'classes': CLASS_NAMES_PATH,
        'confusion': os.path.join(app.root_path, 'model', 'confusion_matrix.png'),
        'history': os.path.join(app.root_path, 'model', 'training_history.png'),
    }
    if kind not in mapping:
        flash('Invalid file.', 'warning')
        return redirect(url_for('admin_settings'))
    path = mapping[kind]
    if not os.path.exists(path):
        flash('File not found.', 'warning')
        return redirect(url_for('admin_settings'))
    log_event(f"Admin downloaded model artifact: {os.path.basename(path)}")
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

@app.route('/admin/model/confusion_matrix.png')
@admin_required
def admin_model_confusion():
    """Serve the confusion matrix image (admin only)."""
    path = os.path.join(app.root_path, 'model', 'confusion_matrix.png')
    if not os.path.exists(path):
        return render_template('error.html', error="Confusion matrix not found."), 404
    return send_file(path, mimetype='image/png')

@app.route('/admin/model/training_history.png')
@admin_required
def admin_model_training_history_img():
    """Serve the training history image (admin only)."""
    path = os.path.join(app.root_path, 'model', 'training_history.png')
    if not os.path.exists(path):
        return render_template('error.html', error="Training history image not found."), 404
    return send_file(path, mimetype='image/png')

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found."), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('error.html', error="Server error. Please try again."), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Check if model is loaded
    if model is None:
        print("\nWARNING: Model not loaded. The app will not work.")
        print(f"   Please ensure model_final.h5 exists in the model folder.")

    if class_names is None:
        print("WARNING: Class names not loaded.")
        print(f"   Please ensure class_names.pkl exists in the model folder.")

    print("\n" + "="*60)
    print("TERRATRAQ - ROAD CONDITION PREDICTION SYSTEM")
    print("="*60)
    print(f"Model: {'Loaded' if model else 'Not loaded'}")
    print(f"Classes: {'Loaded' if class_names else 'Not loaded'}")
    print(f"Admin:  {ADMIN_USERNAME} (change password via .env)")
    print(f"Visit:  http://localhost:5000")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
