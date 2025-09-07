from flask import Flask, render_template, request, send_from_directory
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ===============================
# Konfigurasi Upload
# ===============================
UPLOAD_FOLDER = "images_upload"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ===============================
# Load Dua Model
# ===============================
models = {
    "anedet": YOLO("Anedet AI/best2.pt"),
    "diadet": YOLO("Diadet AI/best.pt")
}

# ===============================
# Fungsi bantu: cek ekstensi
# ===============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===============================
# Halaman Utama
# ===============================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ===============================
# Tampilkan Gambar Upload
# ===============================
@app.route("/images_upload/<filename>")
def display_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ===============================
# Prediksi
# ===============================
@app.route("/", methods=["POST"])
def predict():
    if 'imagefile' not in request.files:
        return render_template("index.html", error="❌ No file uploaded!")

    imagefile = request.files["imagefile"]
    if imagefile.filename == "":
        return render_template("index.html", error="❌ Empty filename!")

    if not allowed_file(imagefile.filename):
        return render_template("index.html", error="❌ Unsupported file format! Use .png, .jpg, .jpeg, or .webp")

    # Pilih model dari dropdown (default anedet)
    model_choice = request.form.get("model", "anedet")
    if model_choice not in models:
        return render_template("index.html", error="❌ Invalid model choice!")

    # Simpan file
    filename = secure_filename(imagefile.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    imagefile.save(image_path)

    try:
        # Prediksi
        results = models[model_choice](image_path)

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            pred_class = results[0].names[int(results[0].boxes.cls[0])]
            percent = f"{float(results[0].boxes.conf[0]) * 100:.2f}"

            return render_template("index.html", 
                                   prediction=pred_class, 
                                   percent=percent, 
                                   image_file=filename,
                                   model_used=model_choice)
        else:
            return render_template("index.html", error="⚠️ No object detected.", image_file=filename)

    except Exception as e:
        return render_template("index.html", error=f"❌ Error during prediction: {e}")

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(port=5005, debug=False)
