from flask import Flask, Response, jsonify, render_template, request, redirect, flash, session
from flask_cors import CORS
import sqlite3
import os
import cv2
import datetime
import mediapipe as mp
from models.fall_detector import FallDetector
import math

# --- Flask app setup ---
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'templates')
app = Flask(__name__, template_folder=template_dir)
CORS(app)
app.secret_key = "supersecretkey"
DB_NAME = "users.db"

# --- Initialize DB ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT UNIQUE,
        password TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        name TEXT,
        email TEXT UNIQUE,
        role TEXT,
        number TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    c.execute("PRAGMA table_info(contacts)")
    columns = [row[1] for row in c.fetchall()]
    if 'number' not in columns:
        c.execute("ALTER TABLE contacts ADD COLUMN number TEXT")
    conn.commit()
    conn.close()

init_db()

# --- Fall detector setup ---
MODEL_PATH = "yolov12n1.pt"
CONFIDENCE = 0.5 # lower for far detection
FALL_THRESHOLD = 0.7
ANGLE_THRESHOLD = 110 # Adjusted for horizontal fall

fall_detector = FallDetector(model_path=MODEL_PATH, confidence=CONFIDENCE)
if hasattr(fall_detector, "fall_threshold"):
    fall_detector.fall_threshold = FALL_THRESHOLD
else:
    fall_detector.fallthreshold = FALL_THRESHOLD
if hasattr(fall_detector, "angle_threshold"):
    fall_detector.angle_threshold = ANGLE_THRESHOLD
else:
    fall_detector.anglethreshold = ANGLE_THRESHOLD

camera_src = 0  # Webcam
camera = cv2.VideoCapture(camera_src)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fall_detected_flag = False
last_fall_time = None
fall_history = []

# --- Mediapipe pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Preprocessing ---
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame

# --- Helper function for angle ---
def angle(p1, p2):
    return abs(math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x)))

# --- Frame generator ---
# --- Frame generator with reliable fall detection ---
def generate_frames():
    global fall_detected_flag, last_fall_time, fall_history
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- YOLO detection ---
        output_frame, yolo_fall, fall_data = fall_detector.process_frame(frame)

        # --- Mediapipe pose detection ---
        results = pose.process(frame_rgb)
        hip_angle_fall = False

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            # Hip-to-shoulder vector angle
            hip_angle = (math.degrees(math.atan2(left_shoulder.y - left_hip.y,
                                                 left_shoulder.x - left_hip.x)) +
                         math.degrees(math.atan2(right_shoulder.y - right_hip.y,
                                                 right_shoulder.x - right_hip.x))) / 2

            if hip_angle > ANGLE_THRESHOLD:
                hip_angle_fall = True

        # --- Combine YOLO + Mediapipe ---
        fall_detected_bool = bool(yolo_fall) or hip_angle_fall

        if fall_detected_bool and not fall_detected_flag:
            curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fall_detected_flag = True
            last_fall_time = curr_time
            fall_history.append(curr_time)
            print(f"Fall recorded at {curr_time}")
        elif not fall_detected_bool:
            fall_detected_flag = False

        # --- Overlay info ---
        person_count = 1 if results.pose_landmarks else 0
        cv2.putText(output_frame, f"Persons: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        if fall_detected_flag:
            cv2.putText(output_frame, "FALL DETECTED!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ------------------ Flask Routes ------------------
@app.route("/")
def root():
    return redirect("/splash")

@app.route("/splash")
def splash():
    return render_template("splash.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
        user = c.fetchone()
        conn.close()
        if user:
            session["user_id"] = user[0]
            session["username"] = user[1]
            session["password"] = password
            return redirect("/about")
        flash("Invalid email or password.", "danger")
        return redirect("/login")
    return render_template("login.html")

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                      (username, email, password))
            conn.commit()
            flash("Signup successful! You can now login.", "success")
            conn.close()
            return redirect("/login")
        except sqlite3.IntegrityError:
            conn.close()
            flash("Email already exists!", "danger")
            return redirect("/signup")
    return render_template("signup.html")

@app.route("/about")
def about():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("about.html", username=session["username"])

@app.route("/live_monitor")
def live_monitor():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("live_monitor.html", username=session["username"])

@app.route("/settings")
def settings():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("settings.html", username=session["username"])

@app.route("/contacts")
def contacts():
    if "user_id" not in session:
        return redirect("/login")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM contacts WHERE user_id=?", (session["user_id"],))
    contacts = c.fetchall()
    conn.close()
    return render_template("contacts.html", username=session["username"], contacts=contacts)

@app.route("/contacts/add", methods=["GET","POST"])
def add_contact():
    if "user_id" not in session:
        return redirect("/login")
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        role = request.form.get("role")
        number = request.form.get("number")
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("INSERT INTO contacts (user_id, name, email, role, number) VALUES (?, ?, ?, ?, ?)",
                      (session["user_id"], name, email, role, number))
            conn.commit()
            conn.close()
            flash("Contact added successfully!", "success")
            return redirect("/contacts")
        except sqlite3.IntegrityError:
            flash("Error: Duplicate email or invalid data", "danger")
            return render_template("add_contact.html", username=session["username"], name=name, email=email, role=role, number=number)
    return render_template("add_contact.html", username=session["username"])

@app.route("/contacts/edit/<int:id>", methods=["GET","POST"])
def edit_contact(id):
    if "user_id" not in session:
        return redirect("/login")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        role = request.form.get("role")
        number = request.form.get("number")
        c.execute("UPDATE contacts SET name=?, email=?, role=?, number=? WHERE id=? AND user_id=?",
                  (name, email, role, number, id, session["user_id"]))
        conn.commit()
        conn.close()
        flash("Contact updated successfully!", "info")
        return redirect("/contacts")
    else:
        c.execute("SELECT * FROM contacts WHERE id=? AND user_id=?", (id, session["user_id"]))
        contact = c.fetchone()
        conn.close()
        if not contact:
            flash("Contact not found", "danger")
            return redirect("/contacts")
        return render_template("edit_contact.html", username=session["username"], contact=contact)

@app.route("/contacts/delete/<int:id>")
def delete_contact(id):
    if "user_id" not in session:
        return redirect("/login")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM contacts WHERE id=? AND user_id=?", (id, session["user_id"]))
    conn.commit()
    conn.close()
    flash("Contact deleted!", "danger")
    return redirect("/contacts")

@app.route("/logout", methods=["GET","POST"])
def logout():
    if request.method == "POST":
        session.clear()
        return render_template("logout.html", username=None, password=None, logged_out=True)
    if "user_id" not in session:
        return redirect("/login")
    return render_template("logout.html", username=session.get("username"), password=session.get("password"), logged_out=False)
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({
        "fall_detected": fall_detected_flag,
        "timestamp": last_fall_time,
        "history": fall_history
    })

@app.route('/api/stats')
def stats():
    return jsonify({
        "falls_detected": len(fall_history),
        "detection_confidence": CONFIDENCE,
        "person_count": 1
    })

@app.route('/api/falls')
def falls_chart():
    labels = ['T1', 'T2', 'T3', 'T4']
    fall_counts = [0, 1, 0, len(fall_history)]
    return jsonify({"labels": labels, "data": fall_counts})

@app.route('/api/notifications')
def notifications():
    notif = [{"type": "fall", "msg": f"Fall detected at {t}", "timestamp": t} for t in fall_history]
    return jsonify(notif)

# --- Run app ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
