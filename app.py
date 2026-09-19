import os
import cv2
import csv
import pickle
import threading
from datetime import datetime

from flask import (
    Flask, render_template, Response,
    request, redirect, url_for, flash
)
import numpy as np
import face_recognition

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pickle")
ATTENDANCE_PATH = os.path.join(BASE_DIR, "attendance.csv")
ATTENDANCE_PHOTO_DIR = os.path.join(BASE_DIR, "attendance_photos")
os.makedirs(ATTENDANCE_PHOTO_DIR, exist_ok=True)

# Globals
video_capture = cv2.VideoCapture(0)
known_encodings = []
known_names = []
attendance_lock = threading.Lock()
last_frame = None
last_mark = None


def save_encodings(encodings, names):
    data = {"encodings": encodings, "names": names}
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)


def load_encodings():
    global known_encodings, known_names
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
            known_encodings = data.get("encodings", [])
            known_names = data.get("names", [])
        print(f"[INFO] Loaded {len(known_names)} known faces.")
    else:
        known_encodings = []
        known_names = []
        print("[INFO] No encodings file found, starting fresh.")


def train_from_dataset():
    """Rebuild encodings from all dataset images."""
    print("[INFO] Training from dataset...")
    encodings = []
    names = []

    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    for person_name in os.listdir(DATASET_DIR):
        person_folder = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        for filename in os.listdir(person_folder):
            path = os.path.join(person_folder, filename)
            if not path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(img)

            if len(face_locations) == 0:
                print(f"[WARN] No face in {filename}, skipping.")
                continue

            encs = face_recognition.face_encodings(img, face_locations)
            if len(encs) == 0:
                print(f"[WARN] Cannot encode {filename}.")
                continue

            encodings.append(encs[0])
            names.append(person_name)
            print(f"[INFO] Encoded {person_name} from {filename}")

    save_encodings(encodings, names)
    load_encodings()

    print(f"[INFO] Training complete. Total: {len(encodings)} encodings.")


def mark_attendance(name, frame=None):
    """Always store attendance + save photo.""" 
    global last_mark

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    file_exists = os.path.exists(ATTENDANCE_PATH)

    with attendance_lock:
        try:
            with open(ATTENDANCE_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Name", "Date", "Time"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({"Name": name, "Date": date_str, "Time": time_str})

            if frame is not None:
                photo_name = f"{name}_{date_str}_{time_str}".replace(":", "-")
                photo_path = os.path.join(ATTENDANCE_PHOTO_DIR, photo_name + ".jpg")
                cv2.imwrite(photo_path, frame)

            last_mark = (name, f"{date_str} {time_str}")
            print(f"[MARKED] {name} at {time_str}")

        except Exception as e:
            print("Error marking attendance:", e)


def generate_frames():
    """Live camera feed + recognition."""
    global video_capture, last_frame

    while True:
        if not video_capture.isOpened():
            video_capture.open(0)

        success, frame = video_capture.read()
        if not success:
            continue

        # store last frame for registration
        last_frame = frame.copy()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for face_encoding, loc in zip(face_encodings, face_locations):
            name = "Unknown"

            if len(known_encodings) > 0:
                matches = face_recognition.compare_faces(
                    known_encodings, face_encoding, tolerance=0.45
                )
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_index = np.argmin(distances)

                if matches[best_index]:
                    name = known_names[best_index]
                    mark_attendance(name, frame)

            # Draw
            top, right, bottom, left = [x * 4 for x in loc]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, -1)
            cv2.putText(
                frame, name, (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

            if name != "Unknown":
                cv2.putText(
                    frame, "Present", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/attendance")
def attendance():
    return render_template("attendance.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/last_mark")
def last_mark_api():
    if last_mark is None:
        return {"status": "no_record"}
    name, ts = last_mark
    return {"status": "ok", "name": name, "timestamp": ts}


# --------------------------------------------------------
# ✅ UPDATED REGISTER FUNCTION – USES last_frame
# --------------------------------------------------------
@app.route("/register", methods=["POST"])
def register():
    """
    Register a new user:
    - Use the latest frame from the camera (last_frame)
    - Save multiple copies into dataset/<name>/
    - Retrain encodings
    - Mark attendance once
    """
    global last_frame

    name = request.form.get("name", "").strip()
    print("[REGISTER] Request for name:", name)

    if not name:
        flash("Name cannot be empty.", "danger")
        return redirect(url_for("attendance"))

    # Make sure we have at least one frame from the camera
    if last_frame is None:
        flash("No camera frame yet. Wait until your face appears, then try again.", "danger")
        print("[REGISTER] last_frame is None")
        return redirect(url_for("attendance"))

    # Create folder dataset/<name>
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    print("[REGISTER] Saving images to:", person_dir)

    # Save 5 copies of the current frame (good enough for project demo)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(5):
        filename = f"{name}_{timestamp}_{i}.jpg"
        filepath = os.path.join(person_dir, filename)
        cv2.imwrite(filepath, last_frame)
        print("[REGISTER] Saved", filepath)

    # Retrain encodings and mark attendance
    try:
        print("[REGISTER] Starting training...")
        train_from_dataset()
        print("[REGISTER] Training done. Marking attendance...")
        mark_attendance(name, last_frame)
        flash(f"{name} registered successfully and attendance marked.", "success")
        print("[REGISTER] Completed for", name)
    except Exception as e:
        print("[REGISTER] Error during training:", e)
        flash("Error while updating encodings. Check terminal logs.", "danger")

    return redirect(url_for("attendance"))


@app.route("/history")
def history():
    records = []
    if os.path.exists(ATTENDANCE_PATH):
        try:
            with open(ATTENDANCE_PATH, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader, start=1):
                    records.append({
                        "sno": idx,
                        "name": row["Name"],
                        "date": row["Date"],
                        "time": row["Time"]
                    })
            records.reverse()
        except Exception as e:
            print("History read error:", e)

    return render_template("history.html", records=records)


@app.teardown_appcontext
def release_camera(exception):
    if video_capture is not None and video_capture.isOpened():
        video_capture.release()


if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    if os.path.exists(ENCODINGS_PATH):
        load_encodings()
    else:
        train_from_dataset()

    app.run(host="0.0.0.0", port=5000, debug=True)
