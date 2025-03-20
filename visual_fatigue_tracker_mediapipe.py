import cv2
import time
import numpy as np
import pandas as pd
import datetime
import os
import statistics
import argparse

# --- Configuration & Constants ---

# Thresholds - EXPERIMENT AND ADJUST THESE! They are crucial for accurate detection.
PERCLOS_THRESHOLD = 0.6  # Percentage of time eyes are closed (or nearly closed) to indicate fatigue
PERCLOS_DURATION = 60  # Calculate PERCLOS over the last 60 seconds
YAWN_ASPECT_RATIO_THRESHOLD = 0.5  # Adjust this value based on your observations, higher means a wider mouth

# --- Feature Extraction Functions (Approximate with MediaPipe) ---

def get_perclos_mediapipe(ear_history, frame_rate, ear_threshold=0.2): # You might need to adjust the EAR threshold for MediaPipe
    """Calculates PERCLOS (percentage of eye closure) over a specified duration."""
    if len(ear_history) < PERCLOS_DURATION * frame_rate:
        return 0.0  # Not enough data yet

    # Get the EAR values from the specified duration
    recent_ears = ear_history[-int(PERCLOS_DURATION * frame_rate):]
    closed_frames = sum(1 for ear in recent_ears if ear < ear_threshold)
    perclos = (closed_frames / len(recent_ears))
    return perclos

# --- Main Application Class (MediaPipe Only) ---

class MediaPipeFatigueTracker:
    def __init__(self, output_dir="fatigue_data_mediapipe", show_video=True):
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            print("MediaPipe Face Mesh initialized.")
        except ImportError:
            print("MediaPipe not installed. Please install it (pip install mediapipe).")
            self.mp_face_mesh = None
            self.face_mesh = None

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
        self.ear_history =  []  # You'll need to approximate EAR with MediaPipe
        self.blink_count = 0 # Blink detection might be different with MediaPipe
        self.yawn_count = 0
        self.current_session_data = []  # Data for the current session
        self.session_start_time = datetime.datetime.now()
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

        print("\n--- Session Summary Statistics (MediaPipe) ---")

        # Basic Statistics
        print(f"Total Session Duration: {df['timestamp'].max() - df['timestamp'].min():.2f} seconds")
        print(f"Total Yawns: {df['yawn_count'].iloc[-1]}")
        if (time.time() - self.start_time) > 0:
            print(f"Average Frame Rate: {self.frame_count / (time.time() - self.start_time):.2f} FPS")
        else:
            print("Average Frame Rate: 0.00 FPS")

        # PERCLOS Statistics (if you manage to approximate EAR)
        if 'perclos' in df.columns and not df['perclos'].empty:
            perclos_values = df['perclos']
            print(f"Average PERCLOS: {perclos_values.mean():.3f}")
            print(f"Max PERCLOS: {perclos_values.max():.3f}")
            if len(perclos_values) > 0:
                print(f"Time spent above PERCLOS threshold ({PERCLOS_THRESHOLD:.2f}): {len(perclos_values[perclos_values > PERCLOS_THRESHOLD]) / len(perclos_values) * 100:.2f}%")
            else:
                print(f"Time spent above PERCLOS threshold ({PERCLOS_THRESHOLD:.2f}): 0.00%")

        # MAR Statistics
        if 'mar' in df.columns and not df['mar'].empty:
            mar_values = df['mar']
            print(f"Average MAR: {mar_values.mean():.3f}")
            print(f"Max MAR: {mar_values.max():.3f}")

        # Note: Blink and saccade detection with MediaPipe requires different approaches based on landmark movements.
        print("Blink and saccade detection might require different logic with MediaPipe.")

    def process_frame(self, frame):
        img_height, img_width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if self.show_video:
                    for landmark in face_landmarks.landmark:
                        x, y = int(landmark.x * img_width), int(landmark.y * img_height)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1) # Green for MediaPipe

                    # --- Approximate MAR using MediaPipe landmarks ---
                    try:
                        # Approximate mouth landmarks (adjust indices as needed)
                        upper_lip = face_landmarks.landmark[13]
                        lower_lip = face_landmarks.landmark[14]
                        left_corner = face_landmarks.landmark[61]
                        right_corner = face_landmarks.landmark[291]

                        A = np.linalg.norm(np.array([upper_lip.x, upper_lip.y]) - np.array([lower_lip.x, lower_lip.y]))
                        D = np.linalg.norm(np.array([left_corner.x, left_corner.y]) - np.array([right_corner.x, right_corner.y]))
                        mar = A / D if D > 0 else 0  # Avoid division by zero

                        if mar > YAWN_ASPECT_RATIO_THRESHOLD:
                            self.yawn_frame_counter += 1
                        else:
                            if self.yawn_frame_counter > 10: #Consider yawn as true after 10 frames
                                self.yawn_count += 1
                            self.yawn_frame_counter = 0

                        if self.show_video:
                            cv2.putText(frame, f"MAR (MediaPipe): {mar:.2f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Green to match dlib
                    except Exception as e:
                        print(f"Error calculating MAR with MediaPipe: {e}")
                        mar = 0.0

                    # --- Approximate PERCLOS (requires approximating EAR) ---
                    perclos = 0.0 # Placeholder

                    # --- Approximate Saccades and Head Movements ---
                    saccade_count = 0 # Placeholder
                    head_movement_count = 0 # Placeholder

                    # --- Display Counters ---
                    cv2.putText(frame, f"EAR: 0.00", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Placeholder
                    cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Yawns: {self.yawn_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Saccades: {saccade_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Head Movements: {head_movement_count}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if (time.time() - self.start_time) > 0:
                        fps = self.frame_count / (time.time() - self.start_time)
                        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # --- Data Logging ---
                    current_time = time.time() - self.start_time
                    self.current_session_data.append({
                        'timestamp': current_time,
                        'perclos': perclos, # Placeholder
                        'yawn_count': self.yawn_count,
                        'mar': mar
                        # Add other relevant data you can extract from MediaPipe
                    })

        return frame
    
    def run(self):
        """Main application loop."""
        if not self.face_mesh:
            print("MediaPipe Face Mesh not initialized. Exiting.")
            return

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
                cv2.imshow("MediaPipe Visual Fatigue Tracker", processed_frame)

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
    tracker = MediaPipeFatigueTracker()
    tracker.run()