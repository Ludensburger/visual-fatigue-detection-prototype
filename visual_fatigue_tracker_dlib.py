import cv2
import dlib
import time
import numpy as np
import pandas as pd
import datetime
import os
import statistics
import argparse
from scipy.spatial import distance as dist  # Import for EAR calculation

# --- Configuration & Constants ---

#  Face and Eye Detection (dlib) -  These are relatively fast.
FACE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  # Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

#  Thresholds -  EXPERIMENT AND ADJUST THESE!  They are crucial for accurate detection.
EAR_THRESHOLD = 0.22  # Eye Aspect Ratio threshold. Lower = more closed.  Start around 0.2-0.3
BLINK_CONSECUTIVE_FRAMES = 2  # Number of consecutive frames below EAR_THRESHOLD to count as a blink.
BLINK_DURATION_THRESHOLD = 0.1  # Minimum blink duration in seconds.
PERCLOS_THRESHOLD = 0.7  # Percentage of time eyes are closed (or nearly closed) to indicate fatigue
PERCLOS_DURATION = 60  # Calculate PERCLOS over the last 60 seconds
SACCADE_THRESHOLD = 5  # Minimum eye movement in pixels to be considered a saccade
HEAD_MOVEMENT_THRESHOLD = 15  # Minimum head movement in pixels to be considered significant
YAWN_ASPECT_RATIO_THRESHOLD = 0.4 # Adjust this value based on your observations, higher means a wider mouth

# --- Feature Extraction Functions ---

def eye_aspect_ratio(eye):
    """Calculates the Eye Aspect Ratio (EAR)."""
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

def mouth_aspect_ratio(mouth):
    """Calculates the Mouth Aspect Ratio (MAR) for yawn detection."""
    mouth = np.array(mouth)  # Convert to NumPy array
    # Vertical distances (lip corners)
    A = np.linalg.norm(mouth[13] - mouth[19])  # 61 - 67 in dlib's 68-point model
    B = np.linalg.norm(mouth[14] - mouth[18])  # 62 - 66
    C = np.linalg.norm(mouth[15] - mouth[17])  # 63 - 65

    # Horizontal distance (mouth corners)
    D = np.linalg.norm(mouth[12] - mouth[16])  # 60 - 64

    mar = (A + B + C) / (3.0 * D)
    return mar


def get_perclos(ear_history, frame_rate):
    """Calculates PERCLOS (percentage of eye closure) over a specified duration."""
    if len(ear_history) < PERCLOS_DURATION * frame_rate:
        return 0.0  # Not enough data yet

    # Get the EAR values from the specified duration
    recent_ears = ear_history[-int(PERCLOS_DURATION * frame_rate):]
    closed_frames = sum(1 for ear in recent_ears if ear < EAR_THRESHOLD)
    perclos = (closed_frames / len(recent_ears))
    return perclos

def calculate_saccades(eye_positions):
    """Counts saccades (rapid eye movements) based on position changes."""
    saccade_count = 0
    if len(eye_positions) < 2:
        return 0

    for i in range(1, len(eye_positions)):
        distance = np.linalg.norm(np.array(eye_positions[i]) - np.array(eye_positions[i-1]))
        if distance > SACCADE_THRESHOLD:
            saccade_count += 1
    return saccade_count

def calculate_head_movements(head_positions):
    """Counts significant head movements based on position changes."""
    head_movement_count = 0
    if len(head_positions) < 2:
        return 0

    for i in range(1, len(head_positions)):
        distance = np.linalg.norm(np.array(head_positions[i]) - np.array(head_positions[i-1]))
        if distance > HEAD_MOVEMENT_THRESHOLD:
            head_movement_count += 1
    return head_movement_count

# --- Main Application Class (dlib Only) ---

class DlibFatigueTracker:
    def __init__(self, output_dir="fatigue_data_dlib", show_video=True):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(FACE_PREDICTOR_PATH)

        # Find a working camera index
        self.cap = None
        for i in range(3):  # Try 0, 1, 2
            temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if temp_cap.read()[0]:
                self.cap = temp_cap
                print(f"Camera index {i} is working.")
                break
            temp_cap.release()

        if not self.cap:
            print("No working camera found.")
            exit()

        # Lower resolution for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.output_dir = output_dir
        self.show_video = show_video
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure output directory exists

        self.frame_count = 0
        self.start_time = time.time()
        self.ear_history = []
        self.blink_count = 0
        self.blink_durations = []
        self.last_blink_end_time = 0
        self.left_eye_positions = []  # Store (x, y) of left eye centroid
        self.right_eye_positions = [] # Store (x, y) of right eye centroid
        self.head_positions = []       # Store (x, y) of head (nose tip)
        self.yawn_count = 0
        self.last_frame_had_face = False  # To handle face disappearing/reappearing
        self.current_session_data = []  # Store data for the current session
        self.session_start_time = datetime.datetime.now()
        self.blink_frame_counter = 0  # Initialize blink frame counter
        self.yawn_frame_counter = 0  # Initialize yawn frame counter

    def create_timestamped_filename(self, base_name, extension=".csv"):
        """Creates a unique, timestamped filename."""
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"{base_name}_{timestamp}{extension}")

    def save_session_data(self):
        """Saves the collected data for the current session to a CSV file."""
        if not self.current_session_data:
            return  # No data to save

        df = pd.DataFrame(self.current_session_data)
        filename = self.create_timestamped_filename("fatigue_data")
        df.to_csv(filename, index=False)
        print(f"Session data saved to: {filename}")

        # Calculate and print summary statistics *after* saving the raw data
        self.print_summary_statistics(df)

    def print_summary_statistics(self, df):
        """Calculates and prints summary statistics from a DataFrame."""

        print("\n--- Session Summary Statistics (dlib) ---")

        # Basic Statistics
        print(f"Total Session Duration: {df['timestamp'].max() - df['timestamp'].min():.2f} seconds")
        print(f"Total Blinks: {df['blink_count'].iloc[-1]}")
        print(f"Total Yawns: {df['yawn_count'].iloc[-1]}")
        if (time.time() - self.start_time) > 0:
            print(f"Average Frame Rate: {self.frame_count / (time.time() - self.start_time):.2f} FPS")
        else:
            print("Average Frame Rate: 0.00 FPS")

        # Blink Duration Statistics
        blink_durations = df[df['blink_duration'] > 0]['blink_duration']
        if not blink_durations.empty:
            print(f"Average Blink Duration: {blink_durations.mean():.3f} seconds")
            print(f"Median Blink Duration: {blink_durations.median():.3f} seconds")
            print(f"Max Blink Duration: {blink_durations.max():.3f} seconds")
            print(f"Standard Deviation of Blink Duration: {blink_durations.std():.3f} seconds")
        else:
            print("No complete blinks detected in this session.")

        # PERCLOS Statistics
        perclos_values = df['perclos']
        if not perclos_values.empty:
            print(f"Average PERCLOS: {perclos_values.mean():.3f}")
            print(f"Max PERCLOS: {perclos_values.max():.3f}")
            if len(perclos_values) > 0:
                print(f"Time spent above PERCLOS threshold ({PERCLOS_THRESHOLD:.2f}): {len(perclos_values[perclos_values > PERCLOS_THRESHOLD]) / len(perclos_values) * 100:.2f}%")
            else:
                print(f"Time spent above PERCLOS threshold ({PERCLOS_THRESHOLD:.2f}): 0.00%")

        # Saccade Statistics
        saccade_counts = df['saccade_count']
        if not saccade_counts.empty and (df['timestamp'].max() - df['timestamp'].min()) > 0:
            print(f"Total Saccades: {saccade_counts.sum()}")
            print(f"Average Saccades per Minute: {saccade_counts.sum() / (df['timestamp'].max() - df['timestamp'].min()) * 60:.2f}")
        else:
            print("Saccade statistics not available or session too short.")

        # Head Movement Statistics
        head_movement_counts = df['head_movement_count']
        if not head_movement_counts.empty and (df['timestamp'].max() - df['timestamp'].min()) > 0:
            print(f"Total Head Movements: {head_movement_counts.sum()}")
            print(f"Average Head Movements per Minute: {head_movement_counts.sum() / (df['timestamp'].max() - df['timestamp'].min()) * 60:.2f}")
        else:
            print("Head movement statistics not available or session too short.")


    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray) # Apply Histogram Equalization

        rects = self.detector(gray, 0)  # Detect faces

        if rects:
            shape = self.predictor(gray, rects[0]) # Assuming only one face
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            leftEye = shape[36:42]
            rightEye = shape[42:48]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0
            self.ear_history.append(ear)

            # Blink detection logic
            if ear < EAR_THRESHOLD:
                self.blink_frame_counter += 1
            else:
                if self.blink_frame_counter >= BLINK_CONSECUTIVE_FRAMES:
                    current_time = time.time()
                    blink_duration = current_time - self.last_blink_end_time
                    if blink_duration > BLINK_DURATION_THRESHOLD:
                        self.blink_count += 1
                        self.blink_durations.append(blink_duration)
                    self.last_blink_end_time = current_time
                self.blink_frame_counter = 0

            # --- Eye Position and Saccade Detection ---
            left_eye_center = np.mean(leftEye, axis=0).astype(int)
            right_eye_center = np.mean(rightEye, axis=0).astype(int)
            self.left_eye_positions.append(left_eye_center)
            self.right_eye_positions.append(right_eye_center)
            saccade_count = calculate_saccades(self.left_eye_positions) + calculate_saccades(self.right_eye_positions)

            # --- Head Position ---
            nose_tip = shape[30]  # Nose tip landmark
            self.head_positions.append(nose_tip)
            head_movement_count = calculate_head_movements(self.head_positions)

            # --- Yawn Detection ---
            mouth = shape[48:68]  # Mouth landmarks
            mar = mouth_aspect_ratio(shape[48:68])
            if mar > YAWN_ASPECT_RATIO_THRESHOLD:
                self.yawn_frame_counter +=1
            else:
                if self.yawn_frame_counter > 10: #Consider yawn as true after 10 frames
                    self.yawn_count += 1
                self.yawn_frame_counter = 0

            # --- PERCLOS Calculation ---
            frame_rate = self.frame_count / (time.time() - self.start_time) if (time.time() - self.start_time) > 0 else 1.0 # Avoid division by zero
            perclos = get_perclos(self.ear_history, frame_rate)

            # --- Data Logging ---
            current_time = time.time() - self.start_time
            self.current_session_data.append({
                'timestamp': current_time,
                'ear': ear,
                'perclos': perclos,
                'blink_count': self.blink_count,  # Cumulative blink count
                'blink_duration': self.blink_durations[-1] if self.blink_durations else 0, #Last blink duration
                'saccade_count': saccade_count, #Incremental saccade count this frame
                'head_movement_count': head_movement_count, #Incremental Head Movement
                'yawn_count': self.yawn_count,
                'mar': mar
            })

            # --- Visualization ---
            if self.show_video:
                # Draw landmarks and display metrics
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1) # Red for dlib

                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yawns: {self.yawn_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Saccades: {saccade_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Head Movements: {head_movement_count}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if (time.time() - self.start_time) > 0:
                    cv2.putText(frame, f"FPS: {self.frame_count / (time.time() - self.start_time):.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def run(self):
        """Main application loop."""
        # FPS tracking
        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Mirror the frame
            frame = cv2.flip(frame, 1)

            processed_frame = self.process_frame(frame)

            if self.show_video:
                cv2.imshow("Dlib Visual Fatigue Tracker", processed_frame)

            # OPTIONAL: Display FPS
            frame_count += 1
            if time.time() - start_time >= 1:
                print(f"FPS: {frame_count}")
                frame_count = 0
                start_time = time.time()
            # OPTIONAL: Display FPS

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# --- Main ---
if __name__ == "__main__":
    tracker = DlibFatigueTracker()
    tracker.run()