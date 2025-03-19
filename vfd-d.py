import cv2
import dlib
import time
import numpy as np
import pandas as pd
import datetime
import os
import statistics
import argparse

# --- Configuration & Constants ---

#   Face and Eye Detection (dlib) -  These are relatively fast.
FACE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  # Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

#   Thresholds -  EXPERIMENT AND ADJUST THESE!  They are crucial for accurate detection.
EAR_THRESHOLD = 0.22  # Eye Aspect Ratio threshold. Lower = more closed.  Start around 0.2-0.3
BLINK_CONSECUTIVE_FRAMES = 2  # Number of consecutive frames below EAR_THRESHOLD to count as a blink.
BLINK_DURATION_THRESHOLD = 0.1  # Minimum blink duration in seconds.
PERCLOS_THRESHOLD = 0.7   # Percentage of time eyes are closed (or nearly closed) to indicate fatigue
PERCLOS_DURATION = 60 # Calculate PERCLOS over the last 60 seconds
SACCADE_THRESHOLD = 5  # Minimum eye movement in pixels to be considered a saccade
HEAD_MOVEMENT_THRESHOLD = 15  # Minimum head movement in pixels to be considered significant
YAWN_ASPECT_RATIO_THRESHOLD = 0.6 # Adjust this value based on your observations, higher means a wider mouth

# --- Feature Extraction Functions ---

def eye_aspect_ratio(eye):
    """Calculates the Eye Aspect Ratio (EAR)."""
    # Vertical distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Horizontal distance
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """Calculates the Mouth Aspect Ratio (MAR) for yawn detection."""
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

# --- Main Application Class ---

class VisualFatigueTracker:
    def __init__(self, output_dir="fatigue_data", show_video=True):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(FACE_PREDICTOR_PATH)
        self.cap = cv2.VideoCapture(0)  # Use webcam (change index if you have multiple cameras)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

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
        self.current_session_data = []
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

        print("\n--- Session Summary Statistics ---")

        # Basic Statistics
        print(f"Total Session Duration: {df['timestamp'].max() - df['timestamp'].min():.2f} seconds")
        print(f"Total Blinks: {df['blink_count'].iloc[-1]}")
        print(f"Total Yawns: {df['yawn_count'].iloc[-1]}")
        print(f"Average Frame Rate: {self.frame_count / (time.time() - self.start_time):.2f} FPS")

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
            print(f"Time spent above PERCLOS threshold ({PERCLOS_THRESHOLD:.2f}): {len(perclos_values[perclos_values > PERCLOS_THRESHOLD]) / len(perclos_values) *100:.2f}%") #Check for divide by zero

        # Saccade Statistics
        saccade_counts = df['saccade_count']
        if not saccade_counts.empty:
          print(f"Total Saccades: {saccade_counts.sum()}")  # Use sum, as each row has incremental saccades
          print(f"Average Saccades per Minute: {saccade_counts.sum() / (df['timestamp'].max() - df['timestamp'].min()) * 60:.2f}")

        # Head Movement Statistics
        head_movement_counts = df['head_movement_count']
        if not head_movement_counts.empty:
          print(f"Total Head Movements: {head_movement_counts.sum()}")
          print(f"Average Head Movements per Minute: {head_movement_counts.sum() / (df['timestamp'].max() - df['timestamp'].min()) * 60:.2f}")


    def process_frame(self, frame):
        """Processes a single frame for fatigue detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if faces:
            self.last_frame_had_face = True
            rect = faces[0]  # Assume only one face for simplicity
            shape = self.predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            # --- Eye Tracking and Blink Detection ---
            left_eye = shape[42:48]  # Left eye landmarks
            right_eye = shape[36:42]  # Right eye landmarks
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            self.ear_history.append(ear)

            # Blink detection logic
            if ear < EAR_THRESHOLD:
                self.blink_frame_counter += 1
            else:
                if self.blink_frame_counter >= BLINK_CONSECUTIVE_FRAMES:
                    blink_duration = time.time() - self.last_blink_end_time
                    if blink_duration > BLINK_DURATION_THRESHOLD:
                      self.blink_count += 1
                      self.blink_durations.append(blink_duration)
                    self.last_blink_end_time = time.time()
                self.blink_frame_counter = 0

            # --- Eye Position and Saccade Detection ---
            left_eye_center = np.mean(left_eye, axis=0).astype(int)
            right_eye_center = np.mean(right_eye, axis=0).astype(int)
            self.left_eye_positions.append(left_eye_center)
            self.right_eye_positions.append(right_eye_center)
            saccade_count = calculate_saccades(self.left_eye_positions) + calculate_saccades(self.right_eye_positions)

            # --- Head Position ---
            nose_tip = shape[30]  # Nose tip landmark
            self.head_positions.append(nose_tip)
            head_movement_count = calculate_head_movements(self.head_positions)


            # --- Yawn Detection ---
            mouth = shape[48:68]  # Mouth landmarks
            mar = mouth_aspect_ratio(mouth)
            if mar > YAWN_ASPECT_RATIO_THRESHOLD:
                self.yawn_frame_counter +=1
            else:
              if self.yawn_frame_counter > 10: #Consider yawn as true after 10 frames
                self.yawn_count += 1
              self.yawn_frame_counter = 0


            # --- PERCLOS Calculation ---
            frame_rate = self.frame_count / (time.time() - self.start_time)
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
                'mar' : mar
            })


            # --- Visualization (optional, for debugging) ---
            if self.show_video:
                # Draw landmarks and display metrics
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yawns: {self.yawn_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Saccades: {saccade_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Head Movements: {head_movement_count}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {frame_rate:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        elif self.last_frame_had_face:
            # Face was detected in the previous frame, but not in this one.
            #  Log this as missing data.
            self.last_frame_had_face = False  # Reset for next detection
            current_time = time.time() - self.start_time
            self.current_session_data.append({
                'timestamp': current_time,
                'ear': np.nan,  # Use NaN for missing values
                'perclos': np.nan,
                'blink_count': self.blink_count,
                'blink_duration': 0,
                'saccade_count': 0,
                'head_movement_count' : 0,
                'yawn_count': self.yawn_count,
                'mar': np.nan
            })

        return frame

    def run(self):
        """Main application loop."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame_count += 1
                processed_frame = self.process_frame(frame)

                if self.show_video:
                    cv2.imshow("Visual Fatigue Tracker", processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):  # Quit on 'q' press
                    break
                elif key == ord("s"):  # Saves data in the middle of the session
                    self.save_session_data()

            # --- End of Main Loop ---

        finally:
            # Release the webcam and close windows
            self.cap.release()
            cv2.destroyAllWindows()
            self.save_session_data() #Save any remaining data.