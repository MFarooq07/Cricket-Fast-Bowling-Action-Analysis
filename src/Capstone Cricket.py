import cv
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
from threading import Thread
import threading
import queue
import shelve
import os
import pandas as pd
from datetime import datetime
import warnings
from functools import partial


# ----------------------------
# Processing & Utility Functions
# ----------------------------


# Initialize DataFrame (not critical if you mainly store data to CSV)
data_df = pd.DataFrame(columns=['Timestamp', 'Username', 'Pixel Distance', 'Real Distance'])
frame_queue = queue.Queue(maxsize=30)


# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Global variables for processing
current_frame = None
cap = None
running = False
recording = False
current_user = None
user_data_path = "user_data"  # Path to store user data
if not os.path.exists(user_data_path):
    os.makedirs(user_data_path)


warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")


def save_user(username, password, shoe_size):
    """Save user credentials with shoe size."""
    with shelve.open(os.path.join(user_data_path, 'user_credentials'), writeback=True) as db:
        db[username] = {
            'password': password,
            'shoe_size': shoe_size
        }


def get_user_info(username):
    """Retrieve user information."""
    with shelve.open(os.path.join(user_data_path, 'user_credentials')) as db:
        if username in db:
            return db[username]
    return None


def check_user(username, password):
    """Check user credentials and retrieve user info if exists."""
    user_info = get_user_info(username)
    if user_info and user_info['password'] == password:
        return user_info
    return None


def log_data(username, pixel_distance, real_distance):
    """(Optional) Example function that used to log foot distances.
       If you no longer need it, you can remove it entirely."""
    global data_df
    new_entry = pd.DataFrame({
        'Timestamp': [datetime.now()],
        'Username': [username],
        'Pixel Distance': [pixel_distance],
        'Real Distance': [real_distance]
    })
    data_df = pd.concat([data_df, new_entry], ignore_index=True)
    data_df.to_excel('user_data.xlsx', index=False)


def initialize_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: The camera could not be opened.")
    else:
        print("Camera initialized successfully.")


def calculate_angle(pointA, pointB, pointC):
    """Generic angle calculation in degrees between points A-B-C."""
    a = np.array(pointA)
    b = np.array(pointB)
    c = np.array(pointC)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def calculate_distance(ptA, ptB):
    """Helper to compute Euclidian distance between two points (x,y)."""
    return np.linalg.norm(np.array(ptA) - np.array(ptB))


def calculate_biomechanical_data(landmarks, shoe_size=10):
    """
    Computes the same measurements as process_images():
      - Elbow Angle
      - Shoulder Angle
      - Left Knee Angle
      - Stride Length (cm)
      - Right Arm Length (cm)


    Scaling is derived from left_heel -> left_foot_index to convert pixels to cm.
    """


    data = {}


    # 1) Calculate foot_length_pixels (LEFT_HEEL ? LEFT_FOOT_INDEX)
    left_heel = (
        landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y
    )
    left_foot_index = (
        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
    )
    foot_length_pixels = calculate_distance(left_heel, left_foot_index)


    # If foot_length_pixels is valid, compute a scaling factor
    if foot_length_pixels > 0:
        # This part mirrors your process_images code that reuses 'shoe_size'
        real_foot_size = 2.5 * ((22 + shoe_size) / 3.0)  # cm
        scaling_factor = real_foot_size / foot_length_pixels
    else:
        scaling_factor = 0.0


    # 2) Stride Length (cm) = distance(left_ankle, right_ankle) * scaling_factor
    left_ankle = (
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    )
    right_ankle = (
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    )
    stride_length_pixels = calculate_distance(left_ankle, right_ankle)
    stride_length_cm = stride_length_pixels * scaling_factor


    # 3) Elbow Angle (right_shoulder, right_elbow, right_wrist)
    right_shoulder = (
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    )
    right_elbow = (
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
    )
    right_wrist = (
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
    )
    angle_elbow = calculate_angle(right_shoulder, right_elbow, right_wrist)


    # 4) Shoulder Angle (right_elbow, right_shoulder, left_shoulder)
    left_shoulder = (
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    )
    angle_shoulder = calculate_angle(right_elbow, right_shoulder, left_shoulder)


    # 5) Left Knee Angle (left_ankle, left_knee, left_hip)
    left_knee = (
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    )
    left_hip = (
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    )
    angle_knee = calculate_angle(left_ankle, left_knee, left_hip)


    # 6) Right Arm Length (cm)
    #    = distance(right_shoulder ? right_elbow) + distance(right_elbow ? right_wrist), then scaled
    right_arm_length_pixels = (calculate_distance(right_shoulder, right_elbow)
                               + calculate_distance(right_elbow, right_wrist))
    right_arm_length_cm = right_arm_length_pixels * scaling_factor


    # Store results in the same dictionary keys as your CSV output
    data["Elbow Angle"] = angle_elbow
    data["Shoulder Angle"] = angle_shoulder
    data["Left Knee Angle"] = angle_knee
    data["Stride Length (cm)"] = stride_length_cm
    data["Right Arm Length (cm)"] = right_arm_length_cm


    return data




def estimate_arm_rotation_speed(elbow_angle, shoulder_angle, knee_angle, stride_length, ball_speed_km_h):
    """
    Estimate arm rotation speed (in degrees/second) based on a simplified
    research-informed approach for cricket fast bowling.


    :param elbow_angle:      Right elbow angle in degrees
    :param shoulder_angle:   A measure of shoulder angle in degrees
    :param knee_angle:       Left (or right) knee angle in degrees
    :param stride_length:    Distance (in cm) between the two feet
    :param ball_speed_km_h:  Approximate ball speed (km/h)


    :return: A float indicating estimated arm rotation speed (deg/s)
    """
    # 1) Convert ball speed from km/h to m/s
    ball_speed_m_s = ball_speed_km_h * 1000.0 / 3600.0


    # 2) Convert stride length from cm to meters
    stride_length_m = stride_length / 100.0


    # 3) Arbitrary weighting logic (fictional example)
    #    Weighted sum of angles multiplied by stride length and ball speed
    #    More emphasis on elbow + shoulder, partial on knee.
    base_angle_factor = (0.4 * elbow_angle) + (0.4 * shoulder_angle) + (0.2 * knee_angle)


    # 4) Scale by stride length and ball speed
    k = 1.2  # A scaling constant (tweak with real data)
    arm_rotation_speed = k * base_angle_factor * stride_length_m * ball_speed_m_s


    return arm_rotation_speed




def append_data_to_csv(data_row, username, base_dir='user_data'):
    """Write a single row (dict) to CSV in the user's folder."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    user_dir = os.path.join(base_dir, username)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    filepath = os.path.join(user_dir, f"{username}_biomechanical_measurements.csv")
    df = pd.DataFrame([data_row])
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', header=header, index=False)
    print(f"Data for {username} saved to {filepath}")


def process_frame(annotated_image_name, image_path, username, shoe_size=10):
    """
    Loads a single image in static mode, calculates angles & stride length,
    writes them to CSV, saves an annotated copy of the image.
    Removed references to user_height because we now do foot-based scaling.
    """
    mp_drawing_local = mp.solutions.drawing_utils
    mp_pose_local = mp.solutions.pose
    pose = mp_pose_local.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True
    )
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return


    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        annotated_image = image_rgb.copy()
        mp_drawing_local.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose_local.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_local.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing_local.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
        )
        landmarks = results.pose_landmarks.landmark


        # Build a data dict with the new function
        data_row = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Image': image_path
        }
        # Add new metrics
        data_row.update(calculate_biomechanical_data(landmarks, shoe_size=shoe_size))


        # Append to user's CSV
        append_data_to_csv(data_row, username)


        # Save annotated version
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(annotated_image_name, annotated_image_bgr)
        print(f"Annotated image saved to {annotated_image_name}")
    pose.close()


# Camera processing is unchanged, though you might remove references to "real_height" if not used
def process_video():
    global running, cap
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to RGB for processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            # Example: we do not log or scale by "real_height" anymore.
            # If you still want real-time angles, you can call the new function here.
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


            # Convert for display in Tkinter
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image=pil_image)
            if 'video_label' in globals():
                video_label.config(image=image_tk)
                video_label.image = image_tk


            root.update()
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()


def start_video():
    global running
    initialize_camera()  # Ensure the camera is initialized
    if not running:
        running = True
        Thread(target=process_video, daemon=True).start()


def stop_video():
    global running
    running = False


def start_recording(username):
    global out, recording, cap
    initialize_camera()
    if not cap.isOpened():
        print("Camera is not available.")
        return
    recording = True
    user_base_dir = os.path.join("user_data", username)  
    user_video_dir = os.path.join(user_base_dir, "videos")
    if not os.path.exists(user_video_dir):
        os.makedirs(user_video_dir)
    video_path = os.path.join(
        user_video_dir, f"{datetime.now().strftime('%Y%m%d%H%M%S')}.avi"
    )
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 120.0, (640, 480))
    if not out.isOpened():
        print("Failed to start recording.")
        return
    print("Recording started.")
    start_time = datetime.now()
    while (datetime.now() - start_time).seconds < 10 and recording:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            print("Failed to capture frame from camera.")
            break
    out.release()
    toggle_recording(username)
    print("Recording stopped.")


def stop_recording():
    global recording, out, cap, running
    recording = False
    if cap and not running:
        cap.release()
        cap = None
        print("Camera released.")
    print("Recording stopped.")


def toggle_recording(username):
    global recording_thread, recording
    if not recording:
        recording_thread = threading.Thread(target=start_recording, args=(username,))
        recording_thread.start()
    else:
        stop_recording()


# ----------------------------
# UI Functions (Single-Window, Frame-based)
# ----------------------------


root = tk.Tk()
root.title("Cricket Analysis")
root.geometry("850x650")
container = tk.Frame(root)
container.pack(fill="both", expand=True)


def clear_container():
    for widget in container.winfo_children():
        widget.destroy()


def show_page(page_function, *args, **kwargs):
    clear_container()
    page_function(*args, **kwargs)


def login_page():
    frame = tk.Frame(container)
    frame.pack(expand=True, fill="both")
    tk.Label(frame, text="Cricket Analysis - Login", font=("Helvetica", 16)).pack(pady=10)
   
    tk.Label(frame, text="Username:").pack(pady=5)
    username_entry = tk.Entry(frame)
    username_entry.pack(pady=5)
   
    tk.Label(frame, text="Password:").pack(pady=5)
    password_entry = tk.Entry(frame, show="*")
    password_entry.pack(pady=5)
   
    def do_login():
        user = username_entry.get()
        pwd = password_entry.get()
        user_info = check_user(user, pwd)
        if user_info:
            global current_user
            current_user = user
            # We keep user_info['height'] but not using it anymore
            messagebox.showinfo("Login Success", f"Welcome, {current_user}!")
            show_page(main_menu_page)
        else:
            messagebox.showerror("Login Failed", "Invalid username or password")
   
    def do_register():
        """Register a new user, asking for shoe size instead of height."""
        user = username_entry.get()
        pwd = password_entry.get()
        if user and pwd:
            # Prompt for shoe size
            shoe_size = simpledialog.askfloat("Shoe Size", "Enter your shoe size (e.g., 10.5):")
            if shoe_size is not None:
                # Create the user folder structure
                user_base_dir = os.path.join("user_data", user)
                user_video_dir = os.path.join(user_base_dir, "videos")
                user_pictures_dir = os.path.join(user_base_dir, "pictures")
                for dir_path in [user_video_dir, user_pictures_dir]:
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
           
                # Save user credentials (with shoe size) to shelve
                save_user(user, pwd, shoe_size)
                messagebox.showinfo("Register Success", f"User '{user}' registered successfully with shoe size {shoe_size}.")
            else:
                messagebox.showerror("Register Failed", "Shoe size entry is required.")
        else:
            messagebox.showerror("Register Failed", "Please enter a username and password")
   
    btn_frame = tk.Frame(frame)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Login", command=do_login).grid(row=0, column=0, padx=5)
    tk.Button(btn_frame, text="Register", command=do_register).grid(row=0, column=1, padx=5)
    tk.Button(frame, text="Exit App", command=exit_app).pack(pady=5)


def main_menu_page():
    frame = tk.Frame(container)
    frame.pack(expand=True, fill="both")
    tk.Label(frame, text=f"Welcome {current_user}", font=("Helvetica", 20)).pack(pady=20)
    btn_frame = tk.Frame(frame)
    btn_frame.pack(pady=20)


    tk.Button(btn_frame, text="Start Video", width=20, command=lambda: show_page(video_page)).grid(row=0, column=0, padx=10, pady=10)
    tk.Button(btn_frame, text="Record Video Clip", width=20, command=lambda: toggle_recording(current_user)).grid(row=0, column=1, padx=10, pady=10)
    tk.Button(btn_frame, text="Stop Video", width=20, command=stop_video).grid(row=0, column=2, padx=10, pady=10)
    tk.Button(btn_frame, text="View Recordings", width=20, command=lambda: show_page(recordings_page)).grid(row=1, column=0, padx=10, pady=10)
    tk.Button(btn_frame, text="View Pictures", width=20, command=lambda: show_page(pictures_page)).grid(row=1, column=1, padx=10, pady=10)


    tk.Button(btn_frame, text="Bowling Analysis", width=20, command=lambda: show_page(bowling_analysis_page)).grid(row=1, column=2, padx=10, pady=10)


    tk.Button(btn_frame, text="Exit App", width=20, command=exit_app).grid(row=2, column=1, padx=10, pady=10)


def video_page():
    frame = tk.Frame(container)
    frame.pack(expand=True, fill="both")
    global video_label
    video_label = tk.Label(frame)
    video_label.pack(pady=10)
   
    # Add a control frame for buttons under the video display
    control_frame = tk.Frame(frame)
    control_frame.pack(pady=5)
   
    # Record button integrated into the video window
    record_button = tk.Button(control_frame, text="Record Video Clip",
                              command=lambda: toggle_recording(current_user))
    record_button.grid(row=0, column=0, padx=5)
   
    tk.Button(control_frame, text="Stop Video", command=stop_video).grid(row=0, column=1, padx=5)
    tk.Button(control_frame, text="Back", command=lambda: show_page(main_menu_page)).grid(row=0, column=2, padx=5)
   
    start_video()




def recordings_page():
    frame = tk.Frame(container)
    frame.pack(expand=True, fill="both")
    tk.Label(frame, text="Recorded Videos", font=("Helvetica", 16)).pack(pady=10)
    user_video_dir = os.path.join("user_data", current_user, "videos")
    if not os.path.exists(user_video_dir):
        tk.Label(frame, text="No recordings found.").pack(pady=5)
    else:
        listbox = tk.Listbox(frame)
        listbox.pack(fill="both", expand=True, padx=20, pady=10)
        videos = os.listdir(user_video_dir)
        for video in videos:
            listbox.insert(tk.END, video)
       
        def play_video():
            selected = listbox.curselection()
            if selected:
                video_file = listbox.get(selected[0])
                video_path = os.path.join(user_video_dir, video_file)
                show_page(video_playback_page, video_path)
       
        tk.Button(frame, text="Play Video", command=play_video).pack(pady=5)
    tk.Button(frame, text="Back", command=lambda: show_page(main_menu_page)).pack(pady=10)


def pictures_page():
    frame = tk.Frame(container)
    frame.pack(expand=True, fill="both")
    tk.Label(frame, text="Stored Pictures", font=("Helvetica", 16)).pack(pady=10)


    user_pictures_dir = os.path.join("user_data", current_user, "pictures")
    if not os.path.exists(user_pictures_dir):
        tk.Label(frame, text="No pictures found.").pack(pady=5)
    else:
        left_frame = tk.Frame(frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        right_frame = tk.Frame(frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)


        listbox = tk.Listbox(left_frame)
        listbox.pack(fill=tk.BOTH, expand=True)


        pictures = os.listdir(user_pictures_dir)
        for picture in pictures:
            listbox.insert(tk.END, picture)


        image_label = tk.Label(right_frame)
        image_label.pack(fill=tk.BOTH, expand=True)
       
        def display_picture(event):
            selected = listbox.curselection()
            if selected:
                picture_file = listbox.get(selected[0])
                picture_path = os.path.join(user_pictures_dir, picture_file)
                img = Image.open(picture_path)
                imgtk = ImageTk.PhotoImage(image=img)
                image_label.config(image=imgtk)
                image_label.image = imgtk
       
        listbox.bind('<<ListboxSelect>>', display_picture)
       
        def annotate_picture(listbox):
            selected = listbox.curselection()
            if selected:
                picture_file = listbox.get(selected[0])
                user_base_dir = os.path.join("user_data", current_user)
                user_pictures_dir = os.path.join(user_base_dir, "pictures")
                user_annotated_pictures_dir = os.path.join(user_base_dir, "annotated_pictures")
                if not os.path.exists(user_annotated_pictures_dir):
                    os.makedirs(user_annotated_pictures_dir)


                picture_path = os.path.join(user_pictures_dir, picture_file)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                original_filename = os.path.splitext(picture_file)[0]
                file_extension = os.path.splitext(picture_file)[1]
                annotated_filename = f"{original_filename}_annotated{file_extension}"
                save_path = os.path.join(user_annotated_pictures_dir, annotated_filename)


                # 1) Retrieve the user�s shoe size from stored credentials
                user_info = get_user_info(current_user)
                shoe_size = user_info["shoe_size"]  # if you stored it under that key


                # 2) Pass shoe_size to process_frame
                process_frame(save_path, picture_path, current_user, shoe_size=shoe_size)
       
                messagebox.showinfo("Success", f"Frame saved as {annotated_filename}")




        def annotate_all_pictures():
            user_base_dir = os.path.join("user_data", current_user)
            user_annotated_pictures_dir = os.path.join(user_base_dir, "annotated_pictures")
            if not os.path.exists(user_annotated_pictures_dir):
                os.makedirs(user_annotated_pictures_dir)


            if not pictures:
                messagebox.showinfo("No Images", "No pictures found to annotate.")
                return
   
            # Retrieve shoe size here, too
            user_info = get_user_info(current_user)
            shoe_size = user_info["shoe_size"]


            for picture_file in pictures:
                picture_path = os.path.join(user_pictures_dir, picture_file)
       
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                original_filename = os.path.splitext(picture_file)[0]
                file_extension = os.path.splitext(picture_file)[1]
                annotated_filename = f"{original_filename}_annotated{file_extension}"
                save_path = os.path.join(user_annotated_pictures_dir, annotated_filename)


                # Use the stored shoe size
                process_frame(save_path, picture_path, current_user, shoe_size=shoe_size)
   
            messagebox.showinfo("Success", "All pictures have been annotated and saved!")




        # NEW: "Study Picture" button
        def study_picture(listbox):
            selected = listbox.curselection()
            if selected:
                picture_file = listbox.get(selected[0])
                # We'll define a function below for analyzing the selected picture
                analyze_single_picture(picture_file)


        tk.Button(frame, text="Annotate and Save", command=lambda: annotate_picture(listbox)).pack(pady=5)
        tk.Button(frame, text="Annotate ALL Pictures", command=annotate_all_pictures).pack(pady=5)
        tk.Button(frame, text="Study Picture", command=lambda: study_picture(listbox)).pack(pady=5)


    tk.Button(frame, text="Back", command=lambda: show_page(main_menu_page)).pack(pady=10)




def analyze_single_picture(picture_file):
    """
    Loads the row in {current_user}_biomechanical_measurements.csv for the given picture_file,
    then:
      - Loads the trained classification model (random forest .pkl).
      - Builds a single-row feature vector from the CSV data (skipping Timestamp and Image).
      - Predicts whether this delivery is Good (1) or Bad (0), plus the probability.
      - Also compares to feature importances (CSV) and provides custom feedback.
    """
    user_base_dir = os.path.join("user_data", current_user)
    csv_path = os.path.join(user_base_dir, f"{current_user}_biomechanical_measurements.csv")
    fi_path = os.path.join(user_base_dir, f"{current_user}_feature_importances.csv")
    model_path = os.path.join(user_base_dir, "trained_rf_classifier_85good.pkl")  # or your correct model file


    if not os.path.exists(csv_path):
        messagebox.showerror("Error", f"No measurements CSV found for {current_user}.")
        return


    # Check if the trained model file exists
    if not os.path.exists(model_path):
        messagebox.showerror("Error", f"No trained classifier found. Please run a Bowling Analysis first.")
        return


    try:
        import joblib
        from sklearn.exceptions import NotFittedError


        # 1) Load the classification model
        rf_clf = joblib.load(model_path)


        # 2) Read the CSV and find the row for the given picture
        #    (If you have lines like '########' above the header, use skiprows or remove them.)
        data = pd.read_csv(csv_path, encoding='cp1252', on_bad_lines='skip', sep=',')


        # Match the row where the "Image" column ends with picture_file
        row = data.loc[data["Image"].str.endswith(picture_file)]
        if row.empty:
            messagebox.showerror("Not Found", f"No data found for {picture_file}")
            return


        # 3) Make a copy for processing
        single_row = row.copy()


        # 3a) Rename columns if your training script used different names
        rename_cols = {
            "Right Elbow Angle": "Elbow Angle",
            "Knee Flexion Angle": "Left Knee Angle",
            "Arm Rotation Speed (deg/s)": "arm rotation speed (deg/sec)"
        }
        single_row.rename(columns=rename_cols, inplace=True)


        # 3b) Drop or ignore the first two columns (Timestamp, Image) so they won't break numeric processing
        #     (We already used "Image" for matching above, so now we don't need it in X_single.)
        #     If your first two columns are literally named "Timestamp" and "Image", you can do:
        non_numeric_cols = ["Timestamp", "Image"]
        single_row.drop(columns=non_numeric_cols, axis=1, inplace=True, errors="ignore")


        # 4) Define exactly which columns your model needs
        feature_cols = [
            "Elbow Angle",
            "Shoulder Angle",
            "Left Knee Angle",
            "Stride Length (cm)",
            "Right arm length (cm)",
            "arm rotation speed (deg/sec)"
        ]


        # Make sure those columns exist
        for c in feature_cols:
            if c not in single_row.columns:
                messagebox.showerror("Error", f"Missing column '{c}' in the CSV for {picture_file}")
                return


        # 5) Build the numeric feature vector
        X_single = single_row[feature_cols]


        # Fill numeric NaNs with the mean of each column
        numeric_cols = X_single.select_dtypes(include=[np.number]).columns
        X_single[numeric_cols] = X_single[numeric_cols].fillna(X_single[numeric_cols].mean())


        # 6) Predict Good/Bad
        y_pred_single = rf_clf.predict(X_single)
        y_prob_single = rf_clf.predict_proba(X_single)[:, 1]
        pred_label = y_pred_single[0]
        pred_prob_good = y_prob_single[0]


        if pred_label == 1:
            classification_feedback = (
                f"Model Classification: GOOD bowl.\n"
                f"Confidence (prob good): {pred_prob_good:.2f}\n"
            )
        else:
            classification_feedback = (
                f"Model Classification: BAD bowl.\n"
                f"Confidence (prob good): {pred_prob_good:.2f}\n"
            )


        # 7) Load feature importances for context
        if not os.path.exists(fi_path):
            messagebox.showinfo("No Analysis Yet", "No feature importances found. Please run a Bowling Analysis first.")
            return
        importances_df = pd.read_csv(fi_path)
        fi_dict = dict(zip(importances_df["Feature"], importances_df["Importance"]))
        sorted_fi = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)


        # 8) Construct feedback message
        feedback = "Analysis for single picture:\n"
        feedback += classification_feedback
        feedback += "\nMost important features (from last analysis):\n"
        for feat, imp in sorted_fi:
            feedback += f"  {feat}: {imp:.3f}\n"


        # 8a) Angle-based suggestions
        elbow_angle = X_single["Elbow Angle"].values[0]
        shoulder_angle = X_single["Shoulder Angle"].values[0]
        knee_angle = X_single["Left Knee Angle"].values[0]
        stride_length = X_single["Stride Length (cm)"].values[0]
        arm_speed = X_single["arm rotation speed (deg/sec)"].values[0]


        feedback += "\nMeasured angles/distances:\n"
        feedback += f"  Elbow Angle: {elbow_angle:.2f}\n"
        feedback += f"  Shoulder Angle: {shoulder_angle:.2f}\n"
        feedback += f"  Left Knee Angle: {knee_angle:.2f}\n"
        feedback += f"  Stride Length (cm): {stride_length:.2f}\n"
        feedback += f"  Arm Rotation Speed (deg/sec): {arm_speed:.2f}\n\n"


        suggestions = []
        if elbow_angle > 160:
            suggestions.append("Try to reduce elbow angle; keep it more flexible for pace.")
        elif elbow_angle < 100:
            suggestions.append("Your elbow angle might be too small, limiting your release swing.")


        if arm_speed < 1000:
            suggestions.append("Try to rotate your arm faster for more pace.")
        elif arm_speed > 2850:
            suggestions.append("Great arm rotation speed! Keep it consistent.")


        if stride_length > 150:
            suggestions.append("Your stride length seems quite large; aim for more controlled stride.")
        elif stride_length < 100:
            suggestions.append("A longer stride can help generate momentum.")


        if suggestions:
            feedback += "Feedback / Suggestions:\n" + "\n".join(f" - {s}" for s in suggestions)
        else:
            feedback += "Your form on this image looks fairly balanced.\n"


        # 9) Show in a popup
        messagebox.showinfo("Picture Analysis", feedback)


    except NotFittedError:
        messagebox.showerror("Model Error", "The loaded classifier is not fitted. Please re-train.")
    except Exception as e:
        messagebox.showerror("Error", f"Could not analyze {picture_file}.\nError: {e}")






def video_playback_page(video_path):
    frame = tk.Frame(container)
    frame.pack(expand=True, fill="both")
    tk.Label(frame, text="Video Playback", font=("Helvetica", 16)).pack(pady=10)
    lbl_video = tk.Label(frame)
    lbl_video.pack(pady=10)
    cap_playback = cv2.VideoCapture(video_path)
    if not cap_playback.isOpened():
        tk.Label(frame, text="Error opening video.").pack(pady=5)
        tk.Button(frame, text="Back", command=lambda: show_page(recordings_page)).pack(pady=10)
        return
    total_frames = int(cap_playback.get(cv2.CAP_PROP_FRAME_COUNT))
    slider = tk.Scale(frame, from_=0, to=total_frames, orient=tk.HORIZONTAL, length=400)
    slider.pack(pady=10)
   
    def update_frame(value):
        frame_no = int(value)
        cap_playback.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame_img = cap_playback.read()
        if ret:
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_img)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl_video.config(image=imgtk)
            lbl_video.image = imgtk
   
    slider.config(command=update_frame)
    def take_picture_from_video(cap_obj):
        ret, frame_img = cap_obj.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            user_base_dir = os.path.join("user_data", current_user)
            user_pictures_dir = os.path.join(user_base_dir, "pictures")
            if not os.path.exists(user_pictures_dir):
                os.makedirs(user_pictures_dir)
            filename = os.path.join(user_pictures_dir, f"frame_{timestamp}.png")
            cv2.imwrite(filename, frame_img)
            messagebox.showinfo("Saved", f"Frame saved as {filename}")


    tk.Button(frame, text="Take Picture", command=lambda: take_picture_from_video(cap_playback)).pack(pady=5)
    tk.Button(frame, text="Back", command=lambda: show_page(recordings_page)).pack(pady=10)


def exit_app():
    stop_video()
    root.quit()
    root.destroy()


def bowling_analysis_page():
    """
    ML analysis page that:
      1) Shows a preview of the user's data.
      2) Trains a Random Forest to predict Speed (km/h).
      3) Prints MAE, cross-validation results, feature importances, and feedback.
      4) Overwrites the user's feature importances CSV (instead of appending).
      5) Displays the text results and the feature-importance chart side by side.
    """
    frame = tk.Frame(container)
    frame.pack(expand=True, fill="both")


    tk.Label(frame, text="Bowling Analysis with Machine Learning", font=("Helvetica", 16)).pack(pady=10)


    # 1) Create a container for left (text) and right (chart)
    content_frame = tk.Frame(frame)
    content_frame.pack(expand=True, fill="both")


    # Left frame for text
    left_frame = tk.Frame(content_frame)
    left_frame.pack(side=tk.LEFT, expand=True, fill="both", padx=10, pady=10)


    # Right frame for chart
    right_frame = tk.Frame(content_frame)
    right_frame.pack(side=tk.RIGHT, expand=True, fill="both", padx=10, pady=10)


    # Text widget goes to the left
    results_text = tk.Text(left_frame, height=20, width=60)
    results_text.pack(expand=True, fill="both")


    # Chart label on the right
    chart_label = tk.Label(right_frame)
    chart_label.pack(expand=True, fill="both")


    def run_analysis():
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import mean_absolute_error
        from io import BytesIO
        from PIL import Image, ImageTk
        import matplotlib
        matplotlib.use("Agg")  # Non-GUI backend
        import matplotlib.pyplot as plt
        from datetime import datetime
        import os, traceback


        try:
            # ------------------------------------------------
            # 1) Build path to the user's CSV
            # ------------------------------------------------
            user_base_dir = os.path.join("user_data", current_user)
            csv_path = os.path.join(user_base_dir, f"{current_user}_biomechanical_measurements.csv")


            # ------------------------------------------------
            # 2) Load CSV and show preview
            # ------------------------------------------------
            data = pd.read_csv(csv_path, encoding='cp1252')
            preview_str = "Data Preview:\n" + data.head().to_string(index=False) + "\n\n"


            # ------------------------------------------------
            # 3) Define target & features
            # ------------------------------------------------
            y = data["Speed (km/h)"]
            X = data[[
                "Elbow Angle",
                "Shoulder Angle",
                "Left Knee Angle",
                "Stride Length (cm)",
                "Arm rotation speed"
            ]]


            # Handle missing
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())


            # Check data size
            if len(data) < 2:
                raise ValueError("Not enough data to train/test split. Please record or annotate more measurements.")


            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


            # ------------------------------------------------
            # 4) Build & train Random Forest
            # ------------------------------------------------
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)


            # ------------------------------------------------
            # 5) Predict, Evaluate
            # ------------------------------------------------
            y_pred = rf.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            cv_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -np.mean(cv_scores)


            # ------------------------------------------------
            # 6) Feature Importances
            # ------------------------------------------------
            importances = rf.feature_importances_
            feature_names = X.columns


            # 6a) Overwrite these weights for each user run (instead of appending)
            importances_df = pd.DataFrame({
                "Timestamp": [datetime.now()] * len(feature_names),
                "Feature": feature_names,
                "Importance": importances
            })
            weightings_path = os.path.join(user_base_dir, f"{current_user}_feature_importances.csv")
            # Use mode='w' so we overwrite the file each time
            importances_df.to_csv(weightings_path, mode='w', index=False, header=True)


            # 6b) Provide feedback
            sorted_idx = np.argsort(importances)[::-1]
            sorted_features = feature_names[sorted_idx]
            sorted_importances = importances[sorted_idx]


            feedback_str = "\nFeedback:\n"
            feedback_str += "These features have the greatest impact on ball speed:\n"


            top_n = 2
            top_feats = sorted_features[:top_n]
            top_imps = sorted_importances[:top_n]


            suggestions_dict = {
                "Elbow Angle": "Keep elbow angle consistent to maximize control.",
                "Shoulder Angle": "Work on shoulder rotation and alignment.",
                "Left Knee Angle": "Front knee stability helps in power transfer.",
                "Stride Length (cm)": "Proper stride builds momentum.",
                "Arm rotation speed": "Increasing arm rotation speed adds extra pace."
            }


            for f, imp_val in zip(top_feats, top_imps):
                feedback_str += f"  * {f} ({imp_val:.2f} importance)\n"
                if f in suggestions_dict:
                    feedback_str += f"    Suggestion: {suggestions_dict[f]}\n"


            # ------------------------------------------------
            # 7) Build textual results to display
            # ------------------------------------------------
            results_str = preview_str
            results_str += f"Mean Absolute Error on Test Set: {mae:.2f} km/h\n"
            results_str += f"Cross-Validated MAE (5-fold): {cv_mae:.2f} km/h\n\n"
            results_str += "Feature Importances (Weights):\n"
            for name, imp in zip(feature_names, importances):
                results_str += f"{name}: {imp:.4f}\n"
            results_str += feedback_str


            # ------------------------------------------------
            # 8) Show textual results in the text widget
            # ------------------------------------------------
            results_text.delete("1.0", tk.END)
            results_text.insert(tk.END, results_str)


            # ------------------------------------------------
            # 9) Plot a bar chart of feature importances
            # ------------------------------------------------
            fig, ax = plt.subplots()
            ax.barh(feature_names, importances)
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importances")


            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            im = Image.open(buf)
            imgtk = ImageTk.PhotoImage(im)
            chart_label.config(image=imgtk)
            chart_label.image = imgtk
            plt.close(fig)


        except Exception as e:
            error_message = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            results_text.delete("1.0", tk.END)
            results_text.insert(tk.END, f"Error during analysis:\n{error_message}")
            chart_label.config(image='')


    # Buttons at the bottom
    tk.Button(frame, text="Run Analysis", command=run_analysis).pack(pady=5)
    tk.Button(frame, text="Back to Main Menu", command=lambda: show_page(main_menu_page)).pack(pady=5)


# ----------------------------
# Start the Application
# ----------------------------
show_page(login_page)
root.mainloop()