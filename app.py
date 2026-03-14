import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from werkzeug.utils import secure_filename
import math

app = Flask(__name__)

# ----------------- Folders (Render Compatible) -----------------
UPLOAD_FOLDER = "/tmp/uploads"
LOGS_FOLDER = "/tmp/logs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)

# ----------------- File Serving Routes -----------------
# This allows the browser to see images stored in /tmp
@app.route('/static/uploads/<filename>')
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/logs/<filename>')
def serve_logs(filename):
    return send_from_directory(LOGS_FOLDER, filename)

# ----------------- Load Model (Version Fix) -----------------
# Your GitHub shows the model is in 'projects/model5.h5'
MODEL_PATH = os.path.join("projects", "model5.h5") if os.path.exists("projects/model5.h5") else "model5.h5"

try:
    print(f"DEBUG: Attempting to load model from: {MODEL_PATH}")
    # custom_objects fix solves the 'batch_shape' error caused by Keras versions
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        compile=False, 
        custom_objects={'InputLayer': InputLayer}
    )
    print("DEBUG: Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. {e}")
    model = None

# ----------------- Utility Functions -----------------
def convert_to_png(src_path, dest_folder):
    img = cv2.imread(src_path)
    if img is None:
        raise ValueError("Invalid image file.")
    base_name = os.path.splitext(os.path.basename(src_path))[0]
    dest_path = os.path.join(dest_folder, f"{base_name}.png")
    cv2.imwrite(dest_path, img)
    return dest_path

def predict_polyp_risk(png_image_path):
    if model is None:
        raise ValueError("AI Model is not loaded. Check server logs.")

    image = cv2.imread(png_image_path)
    if image is None:
        raise ValueError("Invalid image file.")

    image_resized = cv2.resize(image, (256, 256))
    image_norm = image_resized / 255.0

    # Predict
    prob_map = model.predict(np.expand_dims(image_norm, axis=0))[0]
    
    if prob_map.ndim == 3 and prob_map.shape[-1] == 1:
        prob_map = prob_map[..., 0]
    
    mask = (prob_map > 0.5).astype(np.uint8)
    output_image = image_resized.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pixel_to_mm = 0.2
    polyp_lengths = []

    for c in contours:
        if cv2.contourArea(c) > 20:
            cv2.drawContours(output_image, [c], -1, (0, 255, 0), 2)
            area_mm2 = cv2.contourArea(c) * (pixel_to_mm ** 2)
            length_mm = 2 * math.sqrt(area_mm2 / math.pi)
            polyp_lengths.append(length_mm)

    polyp_chance = prob_map[mask == 1].mean() * 100 if mask.sum() > 0 else 0.0
    polyp_length_mm = max(polyp_lengths) if polyp_lengths else 0

    # Risk Logic
    if polyp_length_mm <= 5:
        base_risk = 0.6
    elif polyp_length_mm <= 9:
        base_risk = 2.1
    else:
        base_risk = 13.4

    cancer_risk = (polyp_chance / 100) * base_risk

    output_filename = f"Predicted_{os.path.splitext(os.path.basename(png_image_path))[0]}.png"
    output_path = os.path.join(LOGS_FOLDER, output_filename)
    cv2.imwrite(output_path, output_image)

    return polyp_chance, polyp_length_mm, cancer_risk, output_filename

# ----------------- Main Route -----------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    uploaded_image = None
    processed_image = None
    result_data = None

    if request.method == "POST":
        if "file" not in request.files:
            result = "No file part in the request."
        else:
            file = request.files["file"]
            if file.filename == "":
                result = "No file selected."
            else:
                try:
                    original_file = secure_filename(file.filename)
                    original_path = os.path.join(UPLOAD_FOLDER, original_file)
                    file.save(original_path)

                    # Process
                    uploaded_png_path = convert_to_png(original_path, UPLOAD_FOLDER)
                    uploaded_png_filename = os.path.basename(uploaded_png_path)
                    
                    # This url_for now maps to our /tmp/ serving routes
                    uploaded_image = url_for("serve_uploads", filename=uploaded_png_filename)

                    p_chance, p_length, c_risk, out_file = predict_polyp_risk(uploaded_png_path)
                    
                    processed_image = url_for("serve_logs", filename=out_file)
                    
                    result_data = {
                        "polyp_chance": f"{p_chance:.2f}",
                        "polyp_length_mm": f"{p_length:.2f}",
                        "cancer_risk": f"{c_risk:.2f}"
                    }
                    result = "Analysis successful! ✅"
                except Exception as e:
                    result = f"Error: {str(e)}"
    
    return render_template(
        "results.html",
        result=result,
        uploaded_image=uploaded_image,
        processed_image=processed_image,
        result_data=result_data
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
