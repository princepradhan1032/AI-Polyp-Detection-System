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
LOGS_FOLDER   = "/tmp/logs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER,   exist_ok=True)

@app.route('/static/uploads/<filename>')
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/logs/<filename>')
def serve_logs(filename):
    return send_from_directory(LOGS_FOLDER, filename)

# ── Suppress TF logs to save memory/startup time ──
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"   # force CPU, skip GPU probe

model      = None
load_error = None

def get_model():
    """Lazy-load: only load the model on first real request."""
    global model, load_error
    if model is not None:
        return model
    if load_error is not None:
        raise RuntimeError(load_error)
    try:
        model_path = os.path.join("projects", "model5.h5")
        print(f"DEBUG: Loading model from {model_path} ...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("DEBUG: Model loaded successfully!")
        return model
    except Exception as e:
        load_error = str(e)
        print(f"CRITICAL: Model load failed: {e}")
        raise

def safe_imread(path):
    img = cv2.imread(path)
    if img is not None:
        return img
    pil = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def predict_polyp_risk(png_path):
    m = get_model()
    image   = safe_imread(png_path)
    resized = cv2.resize(image, (256, 256))
    norm    = resized / 255.0

    prob_map = m.predict(np.expand_dims(norm, axis=0))[0]
    if prob_map.ndim == 3 and prob_map.shape[-1] == 1:
        prob_map = prob_map[..., 0]

    mask = (prob_map > 0.5).astype(np.uint8)
    output = resized.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    px_to_mm = 0.2
    lengths  = []
    for c in contours:
        if cv2.contourArea(c) > 20:
            cv2.drawContours(output, [c], -1, (0, 255, 0), 2)
            area_mm2 = cv2.contourArea(c) * (px_to_mm ** 2)
            lengths.append(2 * math.sqrt(area_mm2 / math.pi))

    chance = float(prob_map[mask == 1].mean()) * 100 if mask.sum() > 0 else 0.0
    size   = max(lengths) if lengths else 0.0
    base_risk = 0.6 if size <= 5 else (2.1 if size <= 9 else 13.4)
    risk   = (chance / 100) * base_risk

    out_name = f"Predicted_{os.path.splitext(os.path.basename(png_path))[0]}.png"
    cv2.imwrite(os.path.join(LOGS_FOLDER, out_name), output)
    return chance, size, risk, out_name

@app.route("/", methods=["GET", "POST"])
def home():
    result, uploaded_image, processed_image, result_data = None, None, None, None

    if request.method == "POST":
        if "file" not in request.files:
            result = "No file part in the request."
        else:
            file = request.files["file"]
            if file.filename == "":
                result = "No file selected."
            else:
                filename  = secure_filename(file.filename)
                orig_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(orig_path)

                # Convert to PNG
                img_data = safe_imread(orig_path)
                base_name = os.path.splitext(filename)[0]
                png_path  = os.path.join(UPLOAD_FOLDER, base_name + ".png")
                cv2.imwrite(png_path, img_data)

                uploaded_image = url_for("serve_uploads", filename=base_name + ".png")

                try:
                    chance, size, risk, out_name = predict_polyp_risk(png_path)
                    processed_image = url_for("serve_logs", filename=out_name)
                    result_data = {
                        "polyp_chance":    f"{chance:.2f}",
                        "polyp_length_mm": f"{size:.2f}",
                        "cancer_risk":     f"{risk:.2f}",
                    }
                    result = "Prediction completed successfully ✅"
                except Exception as e:
                    result = f"Error processing image: {str(e)}"

    return render_template(
        "results.html",
        result=result,
        uploaded_image=uploaded_image,
        processed_image=processed_image,
        result_data=result_data,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
