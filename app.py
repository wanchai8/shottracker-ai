from flask import Flask, render_template, request, url_for
import os
import cv2
from ultralytics import YOLO

# กำหนดตำแหน่ง config ของ YOLO สำหรับ Render
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

app = Flask(__name__)

# -------------------------------
# โหลดโมเดล YOLO (Lazy Loading)
# -------------------------------
model = None

def get_model():
    global model
    if model is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "best.pt")
        model = YOLO(model_path)
    return model

# -------------------------------
# หน้าแรกของเว็บไซต์
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# -------------------------------
# Health Check (ใช้ทดสอบว่าเซิร์ฟเวอร์ทำงาน)
# -------------------------------
@app.route('/health')
def health():
    return {"status": "running"}

# -------------------------------
# อัปโหลดและประมวลผลวิดีโอ
# -------------------------------
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400

    upload_folder = os.path.join("static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    input_path = os.path.join(upload_folder, file.filename)
    output_path = os.path.join(upload_folder, f"processed_{file.filename}")
    file.save(input_path)

    model = get_model()

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    made_counter, miss_counter = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    return render_template(
        'index.html',
        video_url=url_for('static', filename=f'uploads/processed_{file.filename}'),
        result={
            "made": made_counter,
            "missed": miss_counter,
            "accuracy": 0
        }
    )

# -------------------------------
# สำหรับการรันในเครื่อง
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
