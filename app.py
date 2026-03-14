import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
import math

app = Flask(__name__)

# ----------------- Folders (Render & Linux Friendly) -----------------
UPLOAD_FOLDER = "/tmp/uploads"
LOGS_FOLDER = "/tmp/logs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)

# ----------------- File Serving Routes -----------------
@app.route('/static/uploads/<filename>')
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/logs/<filename>')
def serve_logs(filename):
    return send_from_directory(LOGS_FOLDER, filename)

# ----------------- Model Loader Fix (Solves batch_shape Error) -----------------
class CompatibleInputLayer(tf.keras.layers.InputLayer):
    """Translates 'batch_shape' into 'shape' for newer Keras versions."""
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['shape'] = kwargs.pop('batch_shape')[1:]
        super().__init__(*args, **kwargs)

# Look for the model in 'projects/' as shown in your GitHub image
MODEL_PATH = os.path.join("projects", "model5.h5") if os.path.exists("projects/model5.h5") else "model5.h5"

try:
    print(f"DEBUG: Loading model from: {MODEL_PATH}")
    # custom_objects is used to 'swap' the broken InputLayer with our fixed one
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        compile=False, 
        custom_objects={'InputLayer': CompatibleInputLayer}
    )
    print("DEBUG: Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR LOADING MODEL: {e}")
    model = None

# ----------------- Prediction Logic -----------------
def predict_polyp_risk(png_image_path):
    if model is None:
        raise ValueError("AI Model is not loaded. Please check the server logs.")

    image = cv2.imread(png_image_path)
    if image is None:
        raise ValueError("Could not read image file.")

    # Image Pre-processing
    image_resized = cv2.resize(image, (256, 256))
    image_norm = image_resized / 255.0

    # AI Inference
    prob_map = model.predict(np.expand_dims(image_norm, axis=0))[0]
    
    if prob_map.ndim == 3:
        prob_map = prob_map[..., 0]
    
    # Post-processing: Generate Mask
    mask = (prob_map > 0.5).astype(np.uint8)
    output_image = image_resized.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Size Estimation
    pixel_to_mm = 0.2
    lengths = []
    for c in contours:
        if cv2.contourArea(c) > 20:
            cv2.drawContours(output_image, [c], -1, (0, 255, 0), 2)
            area_mm2 = cv2.contourArea(c) * (pixel_to_mm ** 2)
            lengths.append(2 * math.sqrt(area_mm2 / math.pi))

    chance = prob_map[mask == 1].mean() * 100 if mask.sum() > 0 else 0.0
    length_mm = max(lengths) if lengths else 0

    # Risk Calculation
    base_risk = 0.6 if length_mm <= 5 else (2.1 if length_mm <= 9 else 13.4)
    cancer_risk = (chance / 100) * base_risk

    # Save Output Image
    out_name = f"Predicted_{os.path.basename(png_image_path)}"
    cv2.imwrite(os.path.join(LOGS_FOLDER, out_name), output_image)
    
    return chance, length_mm, cancer_risk, out_name

# ----------------- Routes -----------------
@app.route("/", methods=["GET", "POST"])
def home():
    res, up_img, proc_img, data = None, None, None, None
    if request.method == "POST" and "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            try:
                # Save input
                filename = secure_filename(file.filename)
                path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(path)
                
                # Convert and predict
                img_path = path if path.lower().endswith('.png') else path + ".png"
                img = cv2.imread(path)
                cv2.imwrite(img_path, img) # Basic PNG conversion
                
                up_img = url_for("serve_uploads", filename=os.path.basename(img_path))
                
                c, l, r, out = predict_polyp_risk(img_path)
                
                proc_img = url_for("serve_logs", filename=out)
                data = {
                    "polyp_chance": f"{c:.2f}",
                    "polyp_length_mm": f"{l:.2f}",
                    "cancer_risk": f"{r:.2f}"
                }
                res = "Analysis successful! ✅"
            except Exception as e:
                res = f"Error: {str(e)}"
    
    return render_template("results.html", result=res, uploaded_image=up_img, processed_image=proc_img, result_data=data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
