from flask import Flask, render_template, Response
import os
import cv2
from YOLOProject import process_frame

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(0)  # ใช้สำหรับทดสอบในเครื่อง
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

# หมายเหตุ: สำหรับ Render ไม่ควรใช้ webcam โดยตรง
@app.route('/video_feed')
def video_feed():
    return "Webcam streaming is not supported on cloud deployment."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)