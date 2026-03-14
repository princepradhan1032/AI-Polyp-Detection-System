# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, render_template, url_for
# from keras.models import load_model
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
# RESULT_FOLDER = os.path.join(app.root_path, 'static', 'results')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)

# MODEL_PATH = r"C:\Users\pradh\OneDrive\Desktop\pythonprojects\projects\model5.h5"
# model = load_model(MODEL_PATH, custom_objects={"iou": tf.keras.metrics.MeanIoU(num_classes=2)})

# def process_image(img_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         raise ValueError("Invalid image.")

#     img_resized = cv2.resize(img, (256, 256))
#     img_norm = img_resized / 255.0

#     prob_map = model.predict(np.expand_dims(img_norm, axis=0))[0]
    
#     if prob_map.ndim == 3:
#         prob_map = prob_map[:, :, 0]  # ensure 2D

#     mask = (prob_map > 0.5).astype(np.uint8) * 255

#     output_img = img_resized.copy()
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for c in contours:
#         if cv2.contourArea(c) > 20:
#             cv2.drawContours(output_img, [c], -1, (0, 255, 0), 2)

#     polyp_chance = 0.0
#     if mask.sum() > 0:
#         polyp_chance = prob_map[mask==255].mean() * 100

#     color = (0, 0, 255) if polyp_chance > 50 else (0, 255, 0)
#     cv2.putText(output_img, f"Polyp Chance: {polyp_chance:.2f}%",
#                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#     output_filename = f"Predicted_{os.path.basename(img_path)}"
#     output_path = os.path.join(RESULT_FOLDER, output_filename)
#     cv2.imwrite(output_path, output_img)

#     return polyp_chance, output_filename

# @app.route("/", methods=["GET", "POST"])
# def home():
#     uploaded_image = None
#     processed_image = None
#     result = None

#     if request.method == "POST":
#         if "file" not in request.files:
#             result = "No file part in the request."
#         else:
#             file = request.files["file"]
#             if file.filename == "":
#                 result = "No file selected."
#             else:
#                 filename = secure_filename(file.filename)
#                 file_path = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(file_path)
#                 print(f"Saved file at: {file_path}")  # debug

#                 uploaded_image = url_for("static", filename=f"uploads/{filename}")

#                 try:
#                     polyp_chance, output_filename = process_image(file_path)
#                     processed_image = url_for("static", filename=f"results/{output_filename}")
#                     result = f"✅ Image uploaded successfully! ⚠️ Polyp Chance: {polyp_chance:.2f}%"
#                     print(f"Processed image saved at: {processed_image}")  # debug
#                 except Exception as e:
#                     result = f"Error processing image: {str(e)}"

#     return render_template("index.html",
#                            uploaded_image=uploaded_image,
#                            processed_image=processed_image,
#                            result=result)

# if __name__ == "__main__":
#     app.run(debug=True)



# import os
# from flask import Flask, request, render_template, url_for
# from werkzeug.utils import secure_filename
# from result import detect_polyp_cancer_risk  # Import function from result.py

# app = Flask(__name__)

# UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
# LOGS_FOLDER = os.path.join(app.root_path, 'projects', 'logs')

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(LOGS_FOLDER, exist_ok=True)

# @app.route("/", methods=["GET", "POST"])
# def home():
#     result = None
#     uploaded_image = None
#     processed_image = None

#     if request.method == "POST":
#         if "file" not in request.files:
#             result = "No file part in the request."
#         else:
#             file = request.files["file"]
#             if file.filename == "":
#                 result = "No file selected."
#             else:
#                 filename = secure_filename(file.filename)
#                 file_path = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(file_path)
#                 uploaded_image = url_for("static", filename=f"uploads/{filename}")

#                 try:
#                     polyp_chance, polyp_length_mm, cancer_risk, output_path = detect_polyp_cancer_risk(file_path)

#                     # Convert output_path to url relative to 'static'
#                     processed_image = url_for("static", filename=os.path.relpath(output_path, os.path.join(app.root_path, 'static')).replace('\\', '/'))

#                     result = f"Polyp Chance: {polyp_chance:.2f}%, Length: {polyp_length_mm:.2f} mm, Cancer Risk: {cancer_risk:.2f}%"
#                 except Exception as e:
#                     result = f"Error processing image: {str(e)}"
#                     processed_image = None

#     return render_template("index.html",
#                            uploaded_image=uploaded_image,
#                            processed_image=processed_image,
#                            result=result)


# if __name__ == "__main__":
#     app.run(debug=True)




# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, render_template, url_for
# from keras.models import load_model
# from werkzeug.utils import secure_filename
# import math

# app = Flask(__name__)

# UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
# LOGS_FOLDER = os.path.join(app.root_path, 'static', 'logs')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(LOGS_FOLDER, exist_ok=True)

# MODEL_PATH = r"projects\model5.h5"
# model = load_model(MODEL_PATH, custom_objects={"iou": tf.keras.metrics.MeanIoU(num_classes=2)})

# def convert_to_png(src_path, dest_folder):
#     """Converts image at src_path to PNG format in dest_folder, returns new file path."""
#     img = cv2.imread(src_path)
#     if img is None:
#         raise ValueError("Invalid image file.")
#     base_name = os.path.splitext(os.path.basename(src_path))[0]
#     dest_path = os.path.join(dest_folder, f"{base_name}.png")
#     cv2.imwrite(dest_path, img)
#     return dest_path

# def predict_polyp_risk(png_image_path):
#     image = cv2.imread(png_image_path)
#     if image is None:
#         raise ValueError("Invalid image file.")

#     image_resized = cv2.resize(image, (256, 256))
#     image_norm = image_resized / 255.0

#     prob_map = model.predict(np.expand_dims(image_norm, axis=0))[0]
#     mask = (prob_map > 0.5).astype(np.uint8)

#     output_image = image_resized.copy()
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     pixel_to_mm = 0.2
#     polyp_lengths = []

#     for c in contours:
#         if cv2.contourArea(c) > 20:
#             cv2.drawContours(output_image, [c], -1, (0, 255, 0), 2)
#             area_mm2 = cv2.contourArea(c) * (pixel_to_mm ** 2)
#             length_mm = 2 * math.sqrt(area_mm2 / math.pi)
#             polyp_lengths.append(length_mm)

#     polyp_chance = prob_map[mask == 1].mean() * 100 if mask.sum() > 0 else 0.0
#     polyp_length_mm = max(polyp_lengths) if polyp_lengths else 0

#     if polyp_length_mm <= 5:
#         base_risk = 0.6
#     elif polyp_length_mm <= 9:
#         base_risk = 2.1
#     else:
#         base_risk = 13.4

#     cancer_risk = polyp_chance / 100 * base_risk

#     color_polyp = (0, 255, 0) if polyp_chance < 50 else (0, 0, 255)
#     color_cancer = (0, 255, 255) if cancer_risk < 5 else (0, 0, 255)

#     cv2.putText(output_image,
#                 f"Polyp Chance: {polyp_chance:.2f}%",
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 color_polyp,
#                 2,
#                 cv2.LINE_AA)

#     cv2.putText(output_image,
#                 f"Length: {polyp_length_mm:.2f} mm",
#                 (10, 65),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 255, 0),
#                 2,
#                 cv2.LINE_AA)

#     cv2.putText(output_image,
#                 f"Cancer Risk: {cancer_risk:.2f}%",
#                 (10, 100),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 color_cancer,
#                 2,
#                 cv2.LINE_AA)

#     output_filename = f"Predicted_{os.path.splitext(os.path.basename(png_image_path))[0]}.png"
#     output_path = os.path.join(LOGS_FOLDER, output_filename)

#     cv2.imwrite(output_path, output_image)

#     return polyp_chance, polyp_length_mm, cancer_risk, output_filename

# @app.route("/", methods=["GET", "POST"])
# def home():
#     result = None
#     uploaded_image = None
#     processed_image = None
#     result_data = None

#     if request.method == "POST":
#         if "file" not in request.files:
#             result = "No file part in the request."
#         else:
#             file = request.files["file"]
#             if file.filename == "":
#                 result = "No file selected."
#             else:
#                 # Save original uploaded image
#                 original_filename = secure_filename(file.filename)
#                 original_file_path = os.path.join(UPLOAD_FOLDER, original_filename)
#                 file.save(original_file_path)

#                 # Convert to PNG for display
#                 uploaded_png_path = convert_to_png(original_file_path, UPLOAD_FOLDER)
#                 uploaded_png_filename = os.path.basename(uploaded_png_path)
#                 uploaded_image = url_for("static", filename=f"uploads/{uploaded_png_filename}")

#                 try:
#                     polyp_chance, polyp_length_mm, cancer_risk, output_filename = predict_polyp_risk(uploaded_png_path)
#                     processed_image = url_for("static", filename=f"logs/{output_filename}")

#                     # Debug prints
#                     print("Uploaded image URL:", uploaded_image)
#                     print("Processed image URL:", processed_image)

#                     result_data = {
#                         "polyp_chance": f"{polyp_chance:.2f}",
#                         "polyp_length_mm": f"{polyp_length_mm:.2f}",
#                         "cancer_risk": f"{cancer_risk:.2f}"
#                     }
#                 except Exception as e:
#                     result = f"Error processing image: {str(e)}"
#                     processed_image = None

#     return render_template("index.html",
#                            uploaded_image=uploaded_image,
#                            processed_image=processed_image,
#                            result=result,
#                            result_data=result_data)

# if __name__ == "__main__":
#     app.run(debug=True)



# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, render_template, url_for
# from keras.models import load_model
# from werkzeug.utils import secure_filename
# import math

# app = Flask(__name__)

# UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
# LOGS_FOLDER = os.path.join(app.root_path, 'static', 'logs')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(LOGS_FOLDER, exist_ok=True)

# MODEL_PATH = r"projects\model5.h5"
# model = load_model(MODEL_PATH, custom_objects={"iou": tf.keras.metrics.MeanIoU(num_classes=2)})

# def convert_to_png(src_path, dest_folder):
#     img = cv2.imread(src_path)
#     if img is None:
#         raise ValueError("Invalid image file.")
#     base_name = os.path.splitext(os.path.basename(src_path))[0]
#     dest_path = os.path.join(dest_folder, f"{base_name}.png")
#     cv2.imwrite(dest_path, img)
#     return dest_path

# def predict_polyp_risk(png_image_path):
#     image = cv2.imread(png_image_path)
#     if image is None:
#         raise ValueError("Invalid image file.")

#     image_resized = cv2.resize(image, (256, 256))
#     image_norm = image_resized / 255.0

#     prob_map = model.predict(np.expand_dims(image_norm, axis=0))[0]
#     mask = (prob_map > 0.5).astype(np.uint8)

#     output_image = image_resized.copy()
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     pixel_to_mm = 0.2
#     polyp_lengths = []

#     for c in contours:
#         if cv2.contourArea(c) > 20:
#             cv2.drawContours(output_image, [c], -1, (0, 255, 0), 2)
#             area_mm2 = cv2.contourArea(c) * (pixel_to_mm ** 2)
#             length_mm = 2 * math.sqrt(area_mm2 / math.pi)
#             polyp_lengths.append(length_mm)

#     polyp_chance = prob_map[mask == 1].mean() * 100 if mask.sum() > 0 else 0.0
#     polyp_length_mm = max(polyp_lengths) if polyp_lengths else 0

#     if polyp_length_mm <= 5:
#         base_risk = 0.6
#     elif polyp_length_mm <= 9:
#         base_risk = 2.1
#     else:
#         base_risk = 13.4

#     cancer_risk = polyp_chance / 100 * base_risk

#     color_polyp = (0, 255, 0) if polyp_chance < 50 else (0, 0, 255)
#     color_cancer = (0, 255, 255) if cancer_risk < 5 else (0, 0, 255)

#     cv2.putText(output_image,
#                 f"Polyp Chance: {polyp_chance:.2f}%",
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 color_polyp,
#                 2,
#                 cv2.LINE_AA)

#     cv2.putText(output_image,
#                 f"Length: {polyp_length_mm:.2f} mm",
#                 (10, 65),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 255, 0),
#                 2,
#                 cv2.LINE_AA)

#     cv2.putText(output_image,
#                 f"Cancer Risk: {cancer_risk:.2f}%",
#                 (10, 100),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 color_cancer,
#                 2,
#                 cv2.LINE_AA)

#     output_filename = f"Predicted_{os.path.splitext(os.path.basename(png_image_path))[0]}.png"
#     output_path = os.path.join(LOGS_FOLDER, output_filename)

#     cv2.imwrite(output_path, output_image)

#     return polyp_chance, polyp_length_mm, cancer_risk, output_filename

# @app.route("/", methods=["GET", "POST"])
# def home():
#     result = None
#     uploaded_image = None
#     processed_image = None
#     result_data = None

#     if request.method == "POST":
#         if "file" not in request.files:
#             result = "No file part in the request."
#         else:
#             file = request.files["file"]
#             if file.filename == "":
#                 result = "No file selected."
#             else:
#                 original_file = secure_filename(file.filename)
#                 original_path = os.path.join(UPLOAD_FOLDER, original_file)
#                 file.save(original_path)

#                 # Convert uploaded image to PNG for browser display
#                 uploaded_png_path = convert_to_png(original_path, UPLOAD_FOLDER)
#                 uploaded_png_filename = os.path.basename(uploaded_png_path)
#                 uploaded_image = url_for("static", filename=f"uploads/{uploaded_png_filename}")

#                 try:
#                     polyp_chance, polyp_length_mm, cancer_risk, output_filename = predict_polyp_risk(uploaded_png_path)
#                     processed_image = url_for("static", filename=f"logs/{output_filename}")

#                     result_data = {
#                         "polyp_chance": f"{polyp_chance:.2f}",
#                         "polyp_length_mm": f"{polyp_length_mm:.2f}",
#                         "cancer_risk": f"{cancer_risk:.2f}"
#                     }
#                 except Exception as e:
#                     result = f"Error processing image: {str(e)}"
#                     processed_image = None

#     return render_template("index.html",
#                            result=result,
#                            uploaded_image=uploaded_image,
#                            processed_image=processed_image,
#                            result_data=result_data)

# if __name__ == "__main__":
#     app.run(debug=True)




# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, render_template, url_for
# from keras.models import load_model
# from werkzeug.utils import secure_filename
# import math

# app = Flask(__name__)

# UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
# LOGS_FOLDER = os.path.join(app.root_path, 'static', 'logs')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(LOGS_FOLDER, exist_ok=True)

# MODEL_PATH = r"projects\model5.h5"
# model = load_model(MODEL_PATH, custom_objects={"iou": tf.keras.metrics.MeanIoU(num_classes=2)})

# def convert_to_png(src_path, dest_folder):
#     img = cv2.imread(src_path)
#     if img is None:
#         raise ValueError("Invalid image file.")
#     base_name = os.path.splitext(os.path.basename(src_path))[0]
#     dest_path = os.path.join(dest_folder, f"{base_name}.png")
#     cv2.imwrite(dest_path, img)
#     return dest_path

# def predict_polyp_risk(png_image_path):
#     image = cv2.imread(png_image_path)
#     if image is None:
#         raise ValueError("Invalid image file.")

#     image_resized = cv2.resize(image, (256, 256))
#     image_norm = image_resized / 255.0

#     prob_map = model.predict(np.expand_dims(image_norm, axis=0))[0]
#     mask = (prob_map > 0.5).astype(np.uint8)

#     output_image = image_resized.copy()
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     pixel_to_mm = 0.2
#     polyp_lengths = []

#     for c in contours:
#         if cv2.contourArea(c) > 20:
#             cv2.drawContours(output_image, [c], -1, (0, 255, 0), 2)
#             area_mm2 = cv2.contourArea(c) * (pixel_to_mm ** 2)
#             length_mm = 2 * math.sqrt(area_mm2 / math.pi)
#             polyp_lengths.append(length_mm)

#     polyp_chance = prob_map[mask == 1].mean() * 100 if mask.sum() > 0 else 0.0
#     polyp_length_mm = max(polyp_lengths) if polyp_lengths else 0

#     if polyp_length_mm <= 5:
#         base_risk = 0.6
#     elif polyp_length_mm <= 9:
#         base_risk = 2.1
#     else:
#         base_risk = 13.4

#     cancer_risk = polyp_chance / 100 * base_risk

#     cv2.putText(output_image,
#                 f"Polyp Chance: {polyp_chance:.2f}%",
#                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     cv2.putText(output_image,
#                 f"Length: {polyp_length_mm:.2f} mm",
#                 (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#     cv2.putText(output_image,
#                 f"Cancer Risk: {cancer_risk:.2f}%",
#                 (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     output_filename = f"Predicted_{os.path.splitext(os.path.basename(png_image_path))[0]}.png"
#     output_path = os.path.join(LOGS_FOLDER, output_filename)
#     cv2.imwrite(output_path, output_image)

#     return polyp_chance, polyp_length_mm, cancer_risk, output_filename

# @app.route("/", methods=["GET", "POST"])
# def home():
#     result = None
#     uploaded_image = None
#     processed_image = None
#     result_data = None

#     if request.method == "POST":
#         if "file" not in request.files:
#             result = "No file part in the request."
#         else:
#             file = request.files["file"]
#             if file.filename == "":
#                 result = "No file selected."
#             else:
#                 original_file = secure_filename(file.filename)
#                 original_path = os.path.join(UPLOAD_FOLDER, original_file)
#                 file.save(original_path)

#                 uploaded_png_path = convert_to_png(original_path, UPLOAD_FOLDER)
#                 uploaded_png_filename = os.path.basename(uploaded_png_path)
#                 uploaded_image = url_for("static", filename=f"uploads/{uploaded_png_filename}")

#                 try:
#                     polyp_chance, polyp_length_mm, cancer_risk, output_filename = predict_polyp_risk(uploaded_png_path)
#                     processed_image = url_for("static", filename=f"logs/{output_filename}")
#                     result_data = {
#                         "polyp_chance": f"{polyp_chance:.2f}",
#                         "polyp_length_mm": f"{polyp_length_mm:.2f}",
#                         "cancer_risk": f"{cancer_risk:.2f}"
#                     }
#                 except Exception as e:
#                     result = f"Error processing image: {str(e)}"

#     return render_template(
#         "index.html",
#         result=result,
#         uploaded_image=uploaded_image,
#         processed_image=processed_image,
#         result_data=result_data
#     )

# if __name__ == "__main__":
#     app.run(debug=True)



# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, render_template, url_for
# from keras.models import load_model
# from werkzeug.utils import secure_filename

# app = Flask(_name_)

# UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
# RESULT_FOLDER = os.path.join(app.root_path, 'static', 'results')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)

# MODEL_PATH = r"C:\Users\pradh\OneDrive\Desktop\pythonprojects\projects\model5.h5"
# model = load_model(MODEL_PATH, custom_objects={"iou": tf.keras.metrics.MeanIoU(num_classes=2)})

# def process_image(img_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         raise ValueError("Invalid image.")

#     img_resized = cv2.resize(img, (256, 256))
#     img_norm = img_resized / 255.0

#     prob_map = model.predict(np.expand_dims(img_norm, axis=0))[0]
    
#     if prob_map.ndim == 3:
#         prob_map = prob_map[:, :, 0]  # ensure 2D

#     mask = (prob_map > 0.5).astype(np.uint8) * 255

#     output_img = img_resized.copy()
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for c in contours:
#         if cv2.contourArea(c) > 20:
#             cv2.drawContours(output_img, [c], -1, (0, 255, 0), 2)

#     polyp_chance = 0.0
#     if mask.sum() > 0:
#         polyp_chance = prob_map[mask==255].mean() * 100

#     color = (0, 0, 255) if polyp_chance > 50 else (0, 255, 0)
#     cv2.putText(output_img, f"Polyp Chance: {polyp_chance:.2f}%",
#                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#     output_filename = f"Predicted_{os.path.basename(img_path)}"
#     output_path = os.path.join(RESULT_FOLDER, output_filename)
#     cv2.imwrite(output_path, output_img)

#     return polyp_chance, output_filename

# @app.route("/", methods=["GET", "POST"])
# def home():
#     uploaded_image = None
#     processed_image = None
#     result = None

#     if request.method == "POST":
#         if "file" not in request.files:
#             result = "No file part in the request."
#         else:
#             file = request.files["file"]
#             if file.filename == "":
#                 result = "No file selected."
#             else:
#                 filename = secure_filename(file.filename)
#                 file_path = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(file_path)
#                 print(f"Saved file at: {file_path}")  # debug

#                 uploaded_image = url_for("static", filename=f"uploads/{filename}")

#                 try:
#                     polyp_chance, output_filename = process_image(file_path)
#                     processed_image = url_for("static", filename=f"results/{output_filename}")
#                     result = f"✅ Image uploaded successfully! ⚠ Polyp Chance: {polyp_chance:.2f}%"
#                     print(f"Processed image saved at: {processed_image}")  # debug
#                 except Exception as e:
#                     result = f"Error processing image: {str(e)}"

#     return render_template("index.html",
#                            uploaded_image=uploaded_image,
#                            processed_image=processed_image,
#                            result=result)

# if _name_ == "_main_":
#     app.run(debug=True)



# import os
# from flask import Flask, request, render_template, url_for
# from werkzeug.utils import secure_filename
# from result import detect_polyp_cancer_risk  # Import function from result.py

# app = Flask(_name_)

# UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
# LOGS_FOLDER = os.path.join(app.root_path, 'projects', 'logs')

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(LOGS_FOLDER, exist_ok=True)

# @app.route("/", methods=["GET", "POST"])
# def home():
#     result = None
#     uploaded_image = None
#     processed_image = None

#     if request.method == "POST":
#         if "file" not in request.files:
#             result = "No file part in the request."
#         else:
#             file = request.files["file"]
#             if file.filename == "":
#                 result = "No file selected."
#             else:
#                 filename = secure_filename(file.filename)
#                 file_path = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(file_path)
#                 uploaded_image = url_for("static", filename=f"uploads/{filename}")

#                 try:
#                     polyp_chance, polyp_length_mm, cancer_risk, output_path = detect_polyp_cancer_risk(file_path)

#                     # Convert output_path to url relative to 'static'
#                     processed_image = url_for("static", filename=os.path.relpath(output_path, os.path.join(app.root_path, 'static')).replace('\\', '/'))

#                     result = f"Polyp Chance: {polyp_chance:.2f}%, Length: {polyp_length_mm:.2f} mm, Cancer Risk: {cancer_risk:.2f}%"
#                 except Exception as e:
#                     result = f"Error processing image: {str(e)}"
#                     processed_image = None

#     return render_template("index.html",
#                            uploaded_image=uploaded_image,
#                            processed_image=processed_image,
#                            result=result)


# if _name_ == "_main_":
#     app.run(debug=True)




# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, render_template, url_for
# from keras.models import load_model
# from werkzeug.utils import secure_filename
# import math

# app = Flask(_name_)

# UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
# LOGS_FOLDER = os.path.join(app.root_path, 'static', 'logs')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(LOGS_FOLDER, exist_ok=True)

# MODEL_PATH = r"projects\model5.h5"
# model = load_model(MODEL_PATH, custom_objects={"iou": tf.keras.metrics.MeanIoU(num_classes=2)})

# def convert_to_png(src_path, dest_folder):
#     """Converts image at src_path to PNG format in dest_folder, returns new file path."""
#     img = cv2.imread(src_path)
#     if img is None:
#         raise ValueError("Invalid image file.")
#     base_name = os.path.splitext(os.path.basename(src_path))[0]
#     dest_path = os.path.join(dest_folder, f"{base_name}.png")
#     cv2.imwrite(dest_path, img)
#     return dest_path

# def predict_polyp_risk(png_image_path):
#     image = cv2.imread(png_image_path)
#     if image is None:
#         raise ValueError("Invalid image file.")

#     image_resized = cv2.resize(image, (256, 256))
#     image_norm = image_resized / 255.0

#     prob_map = model.predict(np.expand_dims(image_norm, axis=0))[0]
#     mask = (prob_map > 0.5).astype(np.uint8)

#     output_image = image_resized.copy()
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     pixel_to_mm = 0.2
#     polyp_lengths = []

#     for c in contours:
#         if cv2.contourArea(c) > 20:
#             cv2.drawContours(output_image, [c], -1, (0, 255, 0), 2)
#             area_mm2 = cv2.contourArea(c) * (pixel_to_mm ** 2)
#             length_mm = 2 * math.sqrt(area_mm2 / math.pi)
#             polyp_lengths.append(length_mm)

#     polyp_chance = prob_map[mask == 1].mean() * 100 if mask.sum() > 0 else 0.0
#     polyp_length_mm = max(polyp_lengths) if polyp_lengths else 0

#     if polyp_length_mm <= 5:
#         base_risk = 0.6
#     elif polyp_length_mm <= 9:
#         base_risk = 2.1
#     else:
#         base_risk = 13.4

#     cancer_risk = polyp_chance / 100 * base_risk

#     color_polyp = (0, 255, 0) if polyp_chance < 50 else (0, 0, 255)
#     color_cancer = (0, 255, 255) if cancer_risk < 5 else (0, 0, 255)

#     cv2.putText(output_image,
#                 f"Polyp Chance: {polyp_chance:.2f}%",
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 color_polyp,
#                 2,
#                 cv2.LINE_AA)

#     cv2.putText(output_image,
#                 f"Length: {polyp_length_mm:.2f} mm",
#                 (10, 65),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 255, 0),
#                 2,
#                 cv2.LINE_AA)

#     cv2.putText(output_image,
#                 f"Cancer Risk: {cancer_risk:.2f}%",
#                 (10, 100),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 color_cancer,
#                 2,
#                 cv2.LINE_AA)

#     output_filename = f"Predicted_{os.path.splitext(os.path.basename(png_image_path))[0]}.png"
#     output_path = os.path.join(LOGS_FOLDER, output_filename)

#     cv2.imwrite(output_path, output_image)

#     return polyp_chance, polyp_length_mm, cancer_risk, output_filename

# @app.route("/", methods=["GET", "POST"])
# def home():
#     result = None
#     uploaded_image = None
#     processed_image = None
#     result_data = None

#     if request.method == "POST":
#         if "file" not in request.files:
#             result = "No file part in the request."
#         else:
#             file = request.files["file"]
#             if file.filename == "":
#                 result = "No file selected."
#             else:
#                 # Save original uploaded image
#                 original_filename = secure_filename(file.filename)
#                 original_file_path = os.path.join(UPLOAD_FOLDER, original_filename)
#                 file.save(original_file_path)

#                 # Convert to PNG for display
#                 uploaded_png_path = convert_to_png(original_file_path, UPLOAD_FOLDER)
#                 uploaded_png_filename = os.path.basename(uploaded_png_path)
#                 uploaded_image = url_for("static", filename=f"uploads/{uploaded_png_filename}")

#                 try:
#                     polyp_chance, polyp_length_mm, cancer_risk, output_filename = predict_polyp_risk(uploaded_png_path)
#                     processed_image = url_for("static", filename=f"logs/{output_filename}")

#                     # Debug prints
#                     print("Uploaded image URL:", uploaded_image)
#                     print("Processed image URL:", processed_image)

#                     result_data = {
#                         "polyp_chance": f"{polyp_chance:.2f}",
#                         "polyp_length_mm": f"{polyp_length_mm:.2f}",
#                         "cancer_risk": f"{cancer_risk:.2f}"
#                     }
#                 except Exception as e:
#                     result = f"Error processing image: {str(e)}"
#                     processed_image = None

#     return render_template("index.html",
#                            uploaded_image=uploaded_image,
#                            processed_image=processed_image,
#                            result=result,
#                            result_data=result_data)

# if _name_ == "_main_":
#     app.run(debug=True)



# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, render_template, url_for
# from keras.models import load_model
# from werkzeug.utils import secure_filename
# import math

# app = Flask(_name_)

# UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
# LOGS_FOLDER = os.path.join(app.root_path, 'static', 'logs')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(LOGS_FOLDER, exist_ok=True)

# MODEL_PATH = r"projects\model5.h5"
# model = load_model(MODEL_PATH, custom_objects={"iou": tf.keras.metrics.MeanIoU(num_classes=2)})

# def convert_to_png(src_path, dest_folder):
#     img = cv2.imread(src_path)
#     if img is None:
#         raise ValueError("Invalid image file.")
#     base_name = os.path.splitext(os.path.basename(src_path))[0]
#     dest_path = os.path.join(dest_folder, f"{base_name}.png")
#     cv2.imwrite(dest_path, img)
#     return dest_path

# def predict_polyp_risk(png_image_path):
#     image = cv2.imread(png_image_path)
#     if image is None:
#         raise ValueError("Invalid image file.")

#     image_resized = cv2.resize(image, (256, 256))
#     image_norm = image_resized / 255.0

#     prob_map = model.predict(np.expand_dims(image_norm, axis=0))[0]
#     mask = (prob_map > 0.5).astype(np.uint8)

#     output_image = image_resized.copy()
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     pixel_to_mm = 0.2
#     polyp_lengths = []

#     for c in contours:
#         if cv2.contourArea(c) > 20:
#             cv2.drawContours(output_image, [c], -1, (0, 255, 0), 2)
#             area_mm2 = cv2.contourArea(c) * (pixel_to_mm ** 2)
#             length_mm = 2 * math.sqrt(area_mm2 / math.pi)
#             polyp_lengths.append(length_mm)

#     polyp_chance = prob_map[mask == 1].mean() * 100 if mask.sum() > 0 else 0.0
#     polyp_length_mm = max(polyp_lengths) if polyp_lengths else 0

#     if polyp_length_mm <= 5:
#         base_risk = 0.6
#     elif polyp_length_mm <= 9:
#         base_risk = 2.1
#     else:
#         base_risk = 13.4

#     cancer_risk = polyp_chance / 100 * base_risk

#     color_polyp = (0, 255, 0) if polyp_chance < 50 else (0, 0, 255)
#     color_cancer = (0, 255, 255) if cancer_risk < 5 else (0, 0, 255)

#     cv2.putText(output_image,
#                 f"Polyp Chance: {polyp_chance:.2f}%",
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 color_polyp,
#                 2,
#                 cv2.LINE_AA)

#     cv2.putText(output_image,
#                 f"Length: {polyp_length_mm:.2f} mm",
#                 (10, 65),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 255, 0),
#                 2,
#                 cv2.LINE_AA)

#     cv2.putText(output_image,
#                 f"Cancer Risk: {cancer_risk:.2f}%",
#                 (10, 100),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 color_cancer,
#                 2,
#                 cv2.LINE_AA)

#     output_filename = f"Predicted_{os.path.splitext(os.path.basename(png_image_path))[0]}.png"
#     output_path = os.path.join(LOGS_FOLDER, output_filename)

#     cv2.imwrite(output_path, output_image)

#     return polyp_chance, polyp_length_mm, cancer_risk, output_filename

# @app.route("/", methods=["GET", "POST"])
# def home():
#     result = None
#     uploaded_image = None
#     processed_image = None
#     result_data = None

#     if request.method == "POST":
#         if "file" not in request.files:
#             result = "No file part in the request."
#         else:
#             file = request.files["file"]
#             if file.filename == "":
#                 result = "No file selected."
#             else:
#                 original_file = secure_filename(file.filename)
#                 original_path = os.path.join(UPLOAD_FOLDER, original_file)
#                 file.save(original_path)

#                 # Convert uploaded image to PNG for browser display
#                 uploaded_png_path = convert_to_png(original_path, UPLOAD_FOLDER)
#                 uploaded_png_filename = os.path.basename(uploaded_png_path)
#                 uploaded_image = url_for("static", filename=f"uploads/{uploaded_png_filename}")

#                 try:
#                     polyp_chance, polyp_length_mm, cancer_risk, output_filename = predict_polyp_risk(uploaded_png_path)
#                     processed_image = url_for("static", filename=f"logs/{output_filename}")

#                     result_data = {
#                         "polyp_chance": f"{polyp_chance:.2f}",
#                         "polyp_length_mm": f"{polyp_length_mm:.2f}",
#                         "cancer_risk": f"{cancer_risk:.2f}"
#                     }
#                 except Exception as e:
#                     result = f"Error processing image: {str(e)}"
#                     processed_image = None

#     return render_template("index.html",
#                            result=result,
#                            uploaded_image=uploaded_image,
#                            processed_image=processed_image,
#                            result_data=result_data)

# if _name_ == "_main_":
#     app.run(debug=True)




# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, render_template, url_for
# from keras.models import load_model
# from werkzeug.utils import secure_filename
# import math

# app = Flask(__name__)

# UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
# LOGS_FOLDER = os.path.join(app.root_path, 'static', 'logs')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(LOGS_FOLDER, exist_ok=True)

# MODEL_PATH = r"projects\model5.h5"
# model = load_model(MODEL_PATH, custom_objects={"iou": tf.keras.metrics.MeanIoU(num_classes=2)})

# def convert_to_png(src_path, dest_folder):
#     img = cv2.imread(src_path)
#     if img is None:
#         raise ValueError("Invalid image file.")
#     base_name = os.path.splitext(os.path.basename(src_path))[0]
#     dest_path = os.path.join(dest_folder, f"{base_name}.png")
#     cv2.imwrite(dest_path, img)
#     return dest_path

# def predict_polyp_risk(png_image_path):
#     image = cv2.imread(png_image_path)
#     if image is None:
#         raise ValueError("Invalid image file.")

#     image_resized = cv2.resize(image, (256, 256))
#     image_norm = image_resized / 255.0

#     prob_map = model.predict(np.expand_dims(image_norm, axis=0))[0]
#     mask = (prob_map > 0.5).astype(np.uint8)

#     output_image = image_resized.copy()
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     pixel_to_mm = 0.2
#     polyp_lengths = []

#     for c in contours:
#         if cv2.contourArea(c) > 20:
#             cv2.drawContours(output_image, [c], -1, (0, 255, 0), 2)
#             area_mm2 = cv2.contourArea(c) * (pixel_to_mm ** 2)
#             length_mm = 2 * math.sqrt(area_mm2 / math.pi)
#             polyp_lengths.append(length_mm)

#     polyp_chance = prob_map[mask == 1].mean() * 100 if mask.sum() > 0 else 0.0
#     polyp_length_mm = max(polyp_lengths) if polyp_lengths else 0

#     if polyp_length_mm <= 5:
#         base_risk = 0.6
#     elif polyp_length_mm <= 9:
#         base_risk = 2.1
#     else:
#         base_risk = 13.4

#     cancer_risk = polyp_chance / 100 * base_risk

#     cv2.putText(output_image,
#                 f"Polyp Chance: {polyp_chance:.2f}%",
#                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     cv2.putText(output_image,
#                 f"Length: {polyp_length_mm:.2f} mm",
#                 (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#     cv2.putText(output_image,
#                 f"Cancer Risk: {cancer_risk:.2f}%",
#                 (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     output_filename = f"Predicted_{os.path.splitext(os.path.basename(png_image_path))[0]}.png"
#     output_path = os.path.join(LOGS_FOLDER, output_filename)
#     cv2.imwrite(output_path, output_image)

#     return polyp_chance, polyp_length_mm, cancer_risk, output_filename

# @app.route("/", methods=["GET", "POST"])
# def home():
#     result = None
#     uploaded_image = None
#     processed_image = None
#     result_data = None

#     if request.method == "POST":
#         if "file" not in request.files:
#             result = "No file part in the request."
#         else:
#             file = request.files["file"]
#             if file.filename == "":
#                 result = "No file selected."
#             else:
#                 original_file = secure_filename(file.filename)
#                 original_path = os.path.join(UPLOAD_FOLDER, original_file)
#                 file.save(original_path)

#                 uploaded_png_path = convert_to_png(original_path, UPLOAD_FOLDER)
#                 uploaded_png_filename = os.path.basename(uploaded_png_path)
#                 uploaded_image = url_for("static", filename=f"uploads/{uploaded_png_filename}")

#                 try:
#                     polyp_chance, polyp_length_mm, cancer_risk, output_filename = predict_polyp_risk(uploaded_png_path)
#                     processed_image = url_for("static", filename=f"logs/{output_filename}")
#                     result_data = {
#                         "polyp_chance": f"{polyp_chance:.2f}",
#                         "polyp_length_mm": f"{polyp_length_mm:.2f}",
#                         "cancer_risk": f"{cancer_risk:.2f}"
#                     }
#                 except Exception as e:
#                     result = f"Error processing image: {str(e)}"

#     return render_template(
#         "index.html",
#         result=result,
#         uploaded_image=uploaded_image,
#         processed_image=processed_image,
#         result_data=result_data
#     )

# if __name__ == "_main_":
#     app.run(debug=True)



# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, render_template, url_for
# from keras.models import load_model
# from werkzeug.utils import secure_filename
# import math

# app = Flask(__name__)

# # ----------------- Folders -----------------
# UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
# LOGS_FOLDER = os.path.join(app.root_path, 'static', 'logs')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(LOGS_FOLDER, exist_ok=True)

# # ----------------- Load Model -----------------
# MODEL_PATH = os.path.join("projects", "model5.h5")
# # Use compile=False if custom metrics are not needed
# model = load_model(MODEL_PATH, compile=False)

# # ----------------- Utility Functions -----------------
# def convert_to_png(src_path, dest_folder):
#     img = cv2.imread(src_path)
#     if img is None:
#         raise ValueError("Invalid image file.")
#     base_name = os.path.splitext(os.path.basename(src_path))[0]
#     dest_path = os.path.join(dest_folder, f"{base_name}.png")
#     cv2.imwrite(dest_path, img)
#     return dest_path

# def predict_polyp_risk(png_image_path):
#     image = cv2.imread(png_image_path)
#     if image is None:
#         raise ValueError("Invalid image file.")

#     image_resized = cv2.resize(image, (256, 256))
#     image_norm = image_resized / 255.0

#     prob_map = model.predict(np.expand_dims(image_norm, axis=0))[0]
#     # Ensure correct mask extraction (handles 3D output)
#     if prob_map.ndim == 3 and prob_map.shape[-1] == 1:
#         prob_map = prob_map[..., 0]
#     mask = (prob_map > 0.5).astype(np.uint8)

#     output_image = image_resized.copy()
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     pixel_to_mm = 0.2
#     polyp_lengths = []

#     for c in contours:
#         if cv2.contourArea(c) > 20:
#             cv2.drawContours(output_image, [c], -1, (0, 255, 0), 2)
#             area_mm2 = cv2.contourArea(c) * (pixel_to_mm ** 2)
#             length_mm = 2 * math.sqrt(area_mm2 / math.pi)
#             polyp_lengths.append(length_mm)

#     polyp_chance = prob_map[mask == 1].mean() * 100 if mask.sum() > 0 else 0.0
#     polyp_length_mm = max(polyp_lengths) if polyp_lengths else 0

#     if polyp_length_mm <= 5:
#         base_risk = 0.6
#     elif polyp_length_mm <= 9:
#         base_risk = 2.1
#     else:
#         base_risk = 13.4

#     cancer_risk = polyp_chance / 100 * base_risk

#     # cv2.putText(output_image,
#     #             f"Polyp Chance: {polyp_chance:.2f}%",
#     #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     # cv2.putText(output_image,
#     #             f"Length: {polyp_length_mm:.2f} mm",
#     #             (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#     # cv2.putText(output_image,
#     #             f"Cancer Risk: {cancer_risk:.2f}%",
#     #             (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     output_filename = f"Predicted_{os.path.splitext(os.path.basename(png_image_path))[0]}.png"
#     output_path = os.path.join(LOGS_FOLDER, output_filename)
#     cv2.imwrite(output_path, output_image)

#     return polyp_chance, polyp_length_mm, cancer_risk, output_filename

# # ----------------- Flask Routes -----------------
# @app.route("/", methods=["GET", "POST"])
# def home():
#     result = None
#     uploaded_image = None
#     processed_image = None
#     result_data = None

#     if request.method == "POST":
#         if "file" not in request.files:
#             result = "No file part in the request."
#         else:
#             file = request.files["file"]
#             if file.filename == "":
#                 result = "No file selected."
#             else:
#                 original_file = secure_filename(file.filename)
#                 original_path = os.path.join(UPLOAD_FOLDER, original_file)
#                 file.save(original_path)

#                 uploaded_png_path = convert_to_png(original_path, UPLOAD_FOLDER)
#                 uploaded_png_filename = os.path.basename(uploaded_png_path)
#                 uploaded_image = url_for("static", filename=f"uploads/{uploaded_png_filename}")

#                 try:
#                     polyp_chance, polyp_length_mm, cancer_risk, output_filename = predict_polyp_risk(uploaded_png_path)
#                     processed_image = url_for("static", filename=f"logs/{output_filename}")
#                     result_data = {
#                         "polyp_chance": f"{polyp_chance:.2f}",
#                         "polyp_length_mm": f"{polyp_length_mm:.2f}",
#                         "cancer_risk": f"{cancer_risk:.2f}"
#                     }
#                     result = "Prediction completed successfully ✅"
#                 except Exception as e:
     
#                     result = f"Error processing image: {str(e)}"
#     print("DEBUG => result:", result)
#     print("DEBUG => result_data:", result_data)
#     print("DEBUG => uploaded_image:", uploaded_image)
#     print("DEBUG => processed_image:", processed_image)
                
                    

#     return render_template(
#         "index.html",
#         result=result,
#         uploaded_image=uploaded_image,
#         processed_image=processed_image,
#         result_data=result_data
#     )

# if __name__ == "__main__":
#     app.run(debug=True)










import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, url_for
from keras.models import load_model
from werkzeug.utils import secure_filename
import math
import time  # <--- NEW IMPORT

app = Flask(__name__)

# ----------------- Folders -----------------
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
LOGS_FOLDER = os.path.join(app.root_path, "static", "logs")

if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.isdir(LOGS_FOLDER):
    os.makedirs(LOGS_FOLDER)
# ----------------- Load Model -----------------
MODEL_PATH = os.path.join("projects", "model5.h5")
# Use compile=False if custom metrics are not needed
model = load_model(MODEL_PATH, compile=False)

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
    image = cv2.imread(png_image_path)
    if image is None:
        raise ValueError("Invalid image file.")

    image_resized = cv2.resize(image, (256, 256))
    image_norm = image_resized / 255.0

    prob_map = model.predict(np.expand_dims(image_norm, axis=0))[0]
    
    # Ensure correct mask extraction (handles 3D output)
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

    if polyp_length_mm <= 5:
        base_risk = 0.6
    elif polyp_length_mm <= 9:
        base_risk = 2.1
    else:
        base_risk = 13.4

    cancer_risk = polyp_chance / 100 * base_risk

    # cv2.putText(output_image,
    #             f"Polyp Chance: {polyp_chance:.2f}%",
    #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # cv2.putText(output_image,
    #             f"Length: {polyp_length_mm:.2f} mm",
    #             (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    # cv2.putText(output_image,
    #             f"Cancer Risk: {cancer_risk:.2f}%",
    #             (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    output_filename = f"Predicted_{os.path.splitext(os.path.basename(png_image_path))[0]}.png"
    output_path = os.path.join(LOGS_FOLDER, output_filename)
    cv2.imwrite(output_path, output_image)

    return polyp_chance, polyp_length_mm, cancer_risk, output_filename

# ----------------- Flask Routes -----------------
@app.route("/", methods=["GET", "POST"])
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
                original_file = secure_filename(file.filename)
                original_path = os.path.join(UPLOAD_FOLDER, original_file)
                file.save(original_path)

                uploaded_png_path = convert_to_png(original_path, UPLOAD_FOLDER)
                uploaded_png_filename = os.path.basename(uploaded_png_path)
                uploaded_image = url_for("static", filename=f"uploads/{uploaded_png_filename}")

                try:
                    polyp_chance, polyp_length_mm, cancer_risk, output_filename = predict_polyp_risk(uploaded_png_path)
                    processed_image = url_for("static", filename=f"logs/{output_filename}")
                    result_data = {
                        "polyp_chance": f"{polyp_chance:.2f}",
                        "polyp_length_mm": f"{polyp_length_mm:.2f}",
                        "cancer_risk": f"{cancer_risk:.2f}"
                    }
                    result = "Prediction completed successfully ✅"
                except Exception as e:
     
                    result = f"Error processing image: {str(e)}"
    print("DEBUG => result:", result)
    print("DEBUG => result_data:", result_data)
    print("DEBUG => uploaded_image:", uploaded_image)
    print("DEBUG => processed_image:", processed_image)
                
                    

    return render_template(
        "results.html",
        result=result,
        uploaded_image=uploaded_image,
        processed_image=processed_image,
        result_data=result_data
    )

if __name__ == "__main__":
    app.run(debug=True)
