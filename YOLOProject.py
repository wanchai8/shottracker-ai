import cv2
from ultralytics import YOLO
import numpy as np
import time

# ==============================
# โหลดโมเดล YOLO
# ==============================
model = YOLO("best.pt")

# ==============================
# ตัวแปรสำหรับการนับคะแนน
# ==============================
class_label_map = {
    0: "basketball",
    1: "rim"
}

made_counter = 0
miss_counter = 0
prev_ball_y = None
rim_center_x, rim_center_y = None, None
rim_tolerance = 40
cooldown = 2
last_event_time = time.time()

# ==============================
# ฟังก์ชันประมวลผลแต่ละเฟรม
# ==============================
def process_frame(frame):
    global made_counter, miss_counter
    global prev_ball_y, rim_center_x, rim_center_y, last_event_time

    height, width, _ = frame.shape

    # ตรวจจับวัตถุด้วย YOLO
    results = model(frame)

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()

        for cls, bbox, score in zip(classes, boxes, scores):
            if score < 0.7:
                continue

            x1, y1, x2, y2 = bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            label = class_label_map.get(cls, "unknown")

            # ตรวจจับห่วง
            if label == "rim":
                rim_center_x = center_x
                rim_center_y = center_y
                color = (255, 0, 0)

            # ตรวจจับลูกบาส
            elif label == "basketball" and rim_center_y is not None:
                color = (0, 255, 0)

                if prev_ball_y is not None:
                    if (prev_ball_y < rim_center_y) and (center_y >= rim_center_y):
                        if time.time() - last_event_time > cooldown:
                            if abs(center_x - rim_center_x) <= rim_tolerance:
                                made_counter += 1
                                result_text = "IN"
                                text_color = (0, 255, 0)
                            else:
                                miss_counter += 1
                                result_text = "OUT"
                                text_color = (0, 0, 255)

                            last_event_time = time.time()

                            cv2.putText(frame, result_text,
                                        (width // 2 - 50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        2, text_color, 4)

                prev_ball_y = center_y
            else:
                color = (0, 255, 255)

            # วาดกรอบ
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            cv2.putText(frame, f"{label} {score:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

    # แสดงผลคะแนน
    total_shots = made_counter + miss_counter
    percentage = (made_counter / total_shots * 100) if total_shots > 0 else 0

    cv2.putText(frame, f"Made: {made_counter}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Missed: {miss_counter}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Accuracy: {percentage:.2f}%", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    return frame

# ==============================
# ฟังก์ชันรีเซ็ตคะแนน
# ==============================
def reset_scores():
    global made_counter, miss_counter, prev_ball_y
    made_counter = 0
    miss_counter = 0
    prev_ball_y = None