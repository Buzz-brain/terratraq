/**
 * TERRATRAQ - Road Condition Prediction System - Custom JavaScript
 */

// ============================================================================
// Sidebar toggle (mobile)
// ============================================================================

function toggleSidebar(open) {
    const shell = document.getElementById('appShell');
    if (!shell) return;
    if (open === undefined) {
        open = !shell.classList.contains('sidebar-open');
    }
    shell.classList.toggle('sidebar-open', open);
}

document.addEventListener('DOMContentLoaded', function() {
    // Close sidebar when the browser window is resized to desktop size
    window.addEventListener('resize', function() {
        if (window.innerWidth >= 992) {
            const shell = document.getElementById('appShell');
            if (shell) shell.classList.remove('sidebar-open');
        }
    });
});

// ============================================================================
// Branded app loader (shown only during auth transitions)
// ============================================================================

function getAppLoader() {
    return document.getElementById('app-loader');
}

function showAppLoader() {
    const loader = getAppLoader();
    if (loader) loader.classList.add('show');
}

function hideAppLoader() {
    const loader = getAppLoader();
    if (loader) loader.classList.remove('show');
}

// ============================================================================
// Auth transitions — show branded loader while logging in / logging out
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    const authForms = document.querySelectorAll('form[data-auth-loader]');
    authForms.forEach(function(form) {
        form.addEventListener('submit', function() {
            showAppLoader();
            const btn = form.querySelector('button[type="submit"]');
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>' +
                    (btn.dataset.loading || 'Please wait...');
            }
        });
    });

    const logoutLink = document.querySelector('.sidebar-logout');
    if (logoutLink) {
        logoutLink.addEventListener('click', function() {
            showAppLoader();
        });
    }
});

// ============================================================================
// DOM Ready
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Road Condition Prediction System loaded');

    // Auto-hide flash messages
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.transition = 'opacity 0.5s';
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 500);
        }, 4000);
    });
});

// ============================================================================
// Image Preview
// ============================================================================

function previewImage(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('imagePreview');
            if (preview) {
                preview.src = e.target.result;
                const container = document.getElementById('imagePreviewContainer');
                if (container) container.classList.remove('d-none');
            }
        };
        reader.readAsDataURL(input.files[0]);
    }
}

// ============================================================================
// File Size Validation
// ============================================================================

function validateFileSize(file, maxSizeMB = 16) {
    const maxBytes = maxSizeMB * 1024 * 1024;
    if (file.size > maxBytes) {
        alert(`File too large! Maximum size is ${maxSizeMB}MB.`);
        return false;
    }
    return true;
}

// ============================================================================
// Copy to Clipboard (for sharing results)
// ============================================================================

function copyResult(text) {
    navigator.clipboard.writeText(text).then(() => {
        const btn = event.target;
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-check me-1"></i>Copied!';
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 2000);
    }).catch(() => {
        alert('Could not copy. Please manually copy the text.');
    });
}

// ============================================================================
// Delete Confirmation
// ============================================================================

function confirmDelete(message = 'Are you sure you want to delete this?') {
    return confirm(message);
}

// ============================================================================
// Password visibility toggle (login / register)
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.password-toggle').forEach(function(btn) {
        btn.addEventListener('click', function() {
            const input = document.querySelector(this.dataset.target);
            if (!input) return;
            const isPassword = input.type === 'password';
            input.type = isPassword ? 'text' : 'password';
            const icon = this.querySelector('i');
            if (icon) icon.className = isPassword ? 'fas fa-eye-slash' : 'fas fa-eye';
            this.setAttribute('aria-label', isPassword ? 'Hide password' : 'Show password');
            input.focus();
        });
    });
});

// ============================================================================
// Auto-submit form with loading state
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    const forms = document.querySelectorAll('form[data-auto-submit]');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const btn = this.querySelector('button[type="submit"]');
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
            }
        });
    });
});
