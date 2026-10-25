
import cv2
from ultralytics import YOLO

print("Loading AI model...")

model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not open.")
    exit()

print("Camera started. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    results = model.predict(
        frame,
        imgsz=320,
        conf=0.4,
        verbose=False
    )

    output = results[0].plot()
    cv2.imshow("BlindAssistAI - Object Detection", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
print("Test completed.")
