from flask import Flask, render_template, request, send_from_directory, url_for
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ===============================
# Konfigurasi Upload & Model
# ===============================
UPLOAD_FOLDER = "images_upload2"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model sekali saja saat server mulai
model = YOLO("Anedet AI/best2.pt")

# ===============================
# Fungsi bantu: Cek ekstensi
# ===============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===============================
# Halaman Utama
# ===============================
@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

# ===============================
# Tampilkan Gambar yang Sudah Di-upload
# ===============================
@app.route('/images_upload2/<filename>')
def display_image(filename):
    abs_path = os.path.abspath(app.config['UPLOAD_FOLDER'])
    return send_from_directory(abs_path, filename)

# ===============================
# Endpoint POST untuk Prediksi
# ===============================
@app.route("/", methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return render_template("index.html", error="❌ Tidak ada file yang diunggah!")

    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return render_template("index.html", error="❌ Nama file kosong!")

    if not allowed_file(imagefile.filename):
        return render_template("index.html", error="❌ Format file tidak didukung! Harus .png, .jpg, .jpeg, atau .webp")

    # Amankan nama file
    filename = secure_filename(imagefile.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    imagefile.save(image_path)

    try:
        # Jalankan prediksi
        results = model(image_path)

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            pred_class = results[0].names[int(results[0].boxes.cls[0])]
            percent = f"{float(results[0].boxes.conf[0]) * 100:.2f}"

            return render_template("index.html", prediction=pred_class, percent=percent, image_file=filename)
        else:
            return render_template("index.html", error="⚠️ Tidak ada objek yang terdeteksi.", image_file=filename)

    except Exception as e:
        return render_template("index.html", error=f"❌ Terjadi error saat prediksi: {e}")

# ===============================
# Jalankan Aplikasi
# ===============================
if __name__ == "__main__":
    app.run(debug=True)