
import cv2
import time
import threading
import pyttsx3
from queue import Queue
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
camera = cv2.VideoCapture(0)

speech_queue = Queue()
announced_tracks = {}
last_seen = {}

TRACK_TIMEOUT = 2.0

def speech_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)

    while True:
        message = speech_queue.get()
        try:
            if message is None:
                break
            engine.say(message)
            engine.runAndWait()
        finally:
            speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

if not camera.isOpened():
    print("Camera not available.")
    raise SystemExit

print("BlindAssistAI running. Press Q to stop.")

try:
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=416,
            conf=0.35,
            verbose=False
        )[0]

        now = time.time()
        current_tracks = set()

        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()

            for track_id, class_id in zip(track_ids, classes):
                current_tracks.add(track_id)
                label = model.names[class_id]
                last_seen[track_id] = now

                if track_id not in announced_tracks:
                    message = f"New {label} detected"
                    print("\n" + message)

                    speech_queue.put(message)
                    announced_tracks[track_id] = label

        # Purane, gayab tracks ko clear karo
        for track_id in list(last_seen):
            if now - last_seen[track_id] > TRACK_TIMEOUT:
                last_seen.pop(track_id, None)
                announced_tracks.pop(track_id, None)

        cv2.imshow("BlindAssistAI", result.plot())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)

print("Assistant stopped.")
