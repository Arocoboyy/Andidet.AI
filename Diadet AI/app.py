from flask import Flask, render_template, request, send_from_directory
import os
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "images_upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model sekali saja saat server mulai
model = YOLO("Diadet AI/best.pt")  # Pastikan file best.pt tidak corrupted

@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")  # Flask otomatis cari di folder templates/

@app.route('/images_upload/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/", methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return render_template("index.html", error="Tidak ada foto yang Anda upload!")

    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return render_template("index.html", error="Tidak ada foto yang Anda upload!")

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
    imagefile.save(image_path)

    try:
        results = model(image_path)

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Ambil class dan confidence dari prediksi pertama
            pred_class = results[0].names[int(results[0].boxes.cls[0])]
            percent = f"{float(results[0].boxes.conf[0]) * 100:.2f}"

            return render_template("index.html", prediction=pred_class, percent=percent, image_file=imagefile.filename)
        else:
            return render_template("index.html", error="Tidak ada objek terdeteksi.")
    except Exception as e:
        return render_template("index.html", error=f"Terjadi error saat prediksi: {e}")

if __name__ == "__main__":
    app.run(debug=True)