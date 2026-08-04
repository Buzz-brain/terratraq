"""
Convert a Keras .h5 model (from the training notebook) to the .onnx format
used by the live app.

Why: the app serves predictions with ONNX Runtime, which is light enough to
run on Render's free 512MB instance (full TensorFlow is not).

Usage (on your own machine / Colab, NOT on the server):

    python tools/convert_to_onnx.py model_final.h5 model_final.onnx

Then upload the resulting .onnx file (plus class_names.pkl) on the site:
Admin -> Model -> Update CNN Model.
"""

import sys

import numpy as np
from tensorflow import keras


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]

    model = keras.models.load_model(src)
    # Warm-up call is required before keras native ONNX export
    model(np.zeros((1, 224, 224, 3), dtype='float32'))

    model.export(dst, format='onnx')
    print(f"Converted {src} -> {dst}")
    print(f"Input:  (batch, 224, 224, 3)")
    print(f"Output: {model.outputs[0].shape[-1]} classes")


if __name__ == '__main__':
    main()
