import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
import math
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "/tmp/uploads"
LOGS_FOLDER = "/tmp/logs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)

@app.route('/static/uploads/<filename>')
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/logs/<filename>')
def serve_logs(filename):
    return send_from_directory(LOGS_FOLDER, filename)

def safe_imread(path):
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        pil_img = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Could not read image '{path}': {e}")

# --- THE REAL FIX: Monkey-patch InputLayer before loading ---
# custom_objects is ignored for built-in layers during deserialization.
# Patching the class directly intercepts TF's own internal lookup.
_original_input_layer_init = tf.keras.layers.InputLayer.__init__

def _patched_input_layer_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs:
        batch_shape = kwargs.pop('batch_shape')
        kwargs['input_shape'] = batch_shape[1:]
    _original_input_layer_init(self, *args, **kwargs)

tf.keras.layers.InputLayer.__init__ = _patched_input_layer_init

MODEL_PATH = os.path.join("projects", "model5.h5") if os.path.exists("projects/model5.h5") else "model5.h5"

try:
    print(f"DEBUG: Attempting load from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("DEBUG: Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    model = None

def predict_polyp_risk(img_path):
    if model is None:
        raise ValueError("AI Model failed to load. Check server startup logs.")

    img = safe_imread(img_path)
    resized = cv2.resize(img, (256, 256))
    normalized = resized / 255.0

    preds = model.predict(np.expand_dims(normalized, axis=0))[0]
    if preds.ndim == 3:
        preds = preds[..., 0]

    mask = (preds > 0.5).astype(np.uint8)
    output_img = resized.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    px_to_mm = 0.2
    lengths = []
    for c in contours:
        if cv2.contourArea(c) > 20:
            cv2.drawContours(output_img, [c], -1, (0, 255, 0), 2)
            area_mm2 = cv2.contourArea(c) * (px_to_mm ** 2)
            lengths.append(2 * math.sqrt(area_mm2 / math.pi))

    chance = float(preds[mask == 1].mean()) * 100 if mask.sum() > 0 else 0.0
    size = max(lengths) if lengths else 0.0
    base_risk = 0.6 if size <= 5 else (2.1 if size <= 9 else 13.4)
    risk = (chance / 100) * base_risk

    out_name = f"Analyzed_{os.path.splitext(os.path.basename(img_path))[0]}.png"
    cv2.imwrite(os.path.join(LOGS_FOLDER, out_name), output_img)
    return chance, size, risk, out_name

@app.route("/", methods=["GET", "POST"])
def home():
    res, up_img, proc_img, data = None, None, None, None

    if request.method == "POST" and "file" in request.files:
        file = request.files["file"]
        if file.filename:
            try:
                filename = secure_filename(file.filename)
                orig_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(orig_path)

                base_name = os.path.splitext(filename)[0]
                png_path = os.path.join(UPLOAD_FOLDER, base_name + ".png")
                img_data = safe_imread(orig_path)
                if not cv2.imwrite(png_path, img_data):
                    raise ValueError("Failed to save converted PNG.")

                up_img = url_for("serve_uploads", filename=base_name + ".png")
                c, s, r, out = predict_polyp_risk(png_path)
                proc_img = url_for("serve_logs", filename=out)

                data = {
                    "polyp_chance": f"{c:.2f}",
                    "polyp_length_mm": f"{s:.2f}",
                    "cancer_risk": f"{r:.2f}"
                }
                res = "Analysis Completed Successfully ✅"
            except Exception as e:
                res = f"Error: {str(e)}"

    return render_template(
        "results.html",
        result=res,
        uploaded_image=up_img,
        processed_image=proc_img,
        result_data=data
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
