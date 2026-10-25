
import cv2
import time
import winsound
from ultralytics import YOLO

print("Starting BlindAssistAI obstacle alert...")

model = YOLO("yolov8n.pt")
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not open.")
    raise SystemExit

last_beep = 0

print("Move objects only in a controlled test.")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Unable to read camera.")
            break

        results = model.predict(
            frame,
            imgsz=320,
            conf=0.4,
            verbose=False
        )

        height, width = frame.shape[:2]
        largest_ratio = 0
        largest_label = None

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            box_area = max(0, x2 - x1) * max(0, y2 - y1)
            ratio = box_area / (width * height)

            class_id = int(box.cls[0])
            label = model.names[class_id]

            if ratio > largest_ratio:
                largest_ratio = ratio
                largest_label = label

        # Rough image-size-based alert intervals.
        if largest_ratio > 0.20:
            interval = 0.30
            status = "LARGE OBJECT IN VIEW"
        elif largest_ratio > 0.08:
            interval = 0.65
            status = "OBJECT IN VIEW"
        elif largest_ratio > 0.025:
            interval = 1.20
            status = "OBJECT DETECTED"
        else:
            interval = None
            status = "NO LARGE OBJECT DETECTED"

        now = time.time()

        if interval and now - last_beep >= interval:
            winsound.Beep(900, 80)
            last_beep = now

        output = results[0].plot()

        cv2.putText(
            output,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        if largest_label:
            print(
                f"\rDetected: {largest_label} | "
                f"Image area: {largest_ratio:.1%}",
                end=""
            )

        cv2.imshow("BlindAssistAI Alerts", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("\nBlindAssistAI test stopped.")

