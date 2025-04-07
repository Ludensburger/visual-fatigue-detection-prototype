import cv2
import time
import numpy as np
import pandas as pd
import datetime
import os
import statistics
# Removed unused import
from collections import deque

# --- Configuration & Constants ---
PERCLOS_THRESHOLD = 0.6  # Percentage of time eyes are closed to indicate fatigue
PERCLOS_DURATION = 10    # Calculate PERCLOS over last 60 seconds
YAWN_ASPECT_RATIO_THRESHOLD = 0.5
BLINK_CONSEC_FRAMES = 2  # Minimum frames for blink detection
CALIBRATION_DURATION = 3 # Seconds for calibration
YAWN_MIN_FRAMES = 10     # Minimum frames to count as a yawn

class MediaPipeFatigueTracker:
    def __init__(self, output_dir="fatigue_data_mediapipe", show_video=True):
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe Face Mesh initialized.")
        except ImportError as e:
            print(f"MediaPipe not installed. Please install it (pip install mediapipe). Error: {e}")
            exit()

        # Camera setup
        self.cap = self._init_camera()
        if not self.cap:
            print("No working camera found.")
            exit()

        # Video settings
        self.show_video = show_video
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Tracking variables
        self.frame_count = 0
        self.start_time = time.time()
        self.ear_history = deque(maxlen=1800)  # 60 sec * 30 fps
        self.ear_smoothed = deque(maxlen=5)    # For smoothing
        self.blink_count = 0
        self.yawn_count = 0
        self.session_start_time = datetime.datetime.now()
        self.current_session_data = []

        # Blink detection
        self.blink_threshold = None
        self.blink_frame_counter = 0
        self.eyes_closed = False
        self.calibrated = False

        # Yawn detection
        self.yawn_frame_counter = 0
        self.yawn_detected = False

        # Get actual frame rate
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        if self.frame_rate <= 0:
            self.frame_rate = 30  # Fallback

    def _init_camera(self):
        """Initialize camera with auto-detection of working index."""
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"Using camera index {i}")
                return cap
        return None

    def calculate_ear(self, landmarks):
        """Improved EAR calculation with MediaPipe landmarks."""
        # Left eye landmarks
        left_eye = [
            landmarks[33], landmarks[160], landmarks[158],
            landmarks[133], landmarks[153], landmarks[144]
        ]
        
        # Right eye landmarks
        right_eye = [
            landmarks[362], landmarks[385], landmarks[387],
            landmarks[263], landmarks[380], landmarks[373]
        ]
        
        def eye_aspect_ratio(eye):
            # Vertical distances
            A = np.linalg.norm([eye[1].x - eye[5].x, eye[1].y - eye[5].y])
            B = np.linalg.norm([eye[2].x - eye[4].x, eye[2].y - eye[4].y])
            # Horizontal distance
            C = np.linalg.norm([eye[0].x - eye[3].x, eye[0].y - eye[3].y])
            return (A + B) / (2.0 * C) if C > 0 else 0

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        return (left_ear + right_ear) / 2.0

    def calculate_mar(self, landmarks):
        """Calculate Mouth Aspect Ratio for yawn detection."""
        try:
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            
            A = np.linalg.norm([upper_lip.x - lower_lip.x, upper_lip.y - lower_lip.y])
            D = np.linalg.norm([left_corner.x - right_corner.x, left_corner.y - right_corner.y])
            return A / D if D > 0 else 0
        except Exception as e:
            print(f"Error calculating MAR: {e}")
            return 0

    def calibrate_blink_threshold(self):
        """Calibrate EAR threshold for the current user."""
        print("Calibrating blink detection... Please keep eyes open")
        ear_values = []
        start_time = time.time()
        
        while time.time() - start_time < CALIBRATION_DURATION:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                ear = self.calculate_ear(results.multi_face_landmarks[0].landmark)
                ear_values.append(ear)
                
                # Show calibration countdown
                remaining = CALIBRATION_DURATION - (time.time() - start_time)
                cv2.putText(frame, f"Calibrating... {int(remaining)}s", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("Calibration", frame)
                cv2.waitKey(1)
        
        if ear_values:
            # Set threshold to 85% of normal open-eye EAR
            self.blink_threshold = statistics.mean(ear_values) * 0.85
            print(f"Calibration complete. EAR threshold: {self.blink_threshold:.3f}")
            self.calibrated = True
        else:
            print("Calibration failed - using default threshold")
            self.blink_threshold = 0.2  # Fallback
            
        cv2.destroyWindow("Calibration")

    def detect_blink(self, ear):
        """State machine for accurate blink detection."""
        if not self.calibrated:
            self.calibrate_blink_threshold()
            
        # Smooth EAR values
        self.ear_smoothed.append(ear)
        smoothed_ear = statistics.mean(self.ear_smoothed) if self.ear_smoothed else ear
        
        # State machine logic
        if smoothed_ear < self.blink_threshold:
            self.blink_frame_counter += 1
            
            if not self.eyes_closed and self.blink_frame_counter >= BLINK_CONSEC_FRAMES:
                self.eyes_closed = True
                
        else:
            if self.eyes_closed:
                self.blink_count += 1
                self.eyes_closed = False
                
            self.blink_frame_counter = 0
            
        return self.eyes_closed

    def detect_yawn(self, mar):
        """Detect yawns using Mouth Aspect Ratio."""
        if mar > YAWN_ASPECT_RATIO_THRESHOLD:
            self.yawn_frame_counter += 1
            
            if not self.yawn_detected and self.yawn_frame_counter >= YAWN_MIN_FRAMES:
                self.yawn_detected = True
                self.yawn_count += 1
        else:
            self.yawn_frame_counter = 0
            self.yawn_detected = False
            
        return self.yawn_detected

    def get_perclos(self):
        """Calculate PERCLOS over the specified duration."""
        if len(self.ear_history) < PERCLOS_DURATION * self.frame_rate:
            return 0.0
            
        recent_ears = list(self.ear_history)[-int(PERCLOS_DURATION * self.frame_rate):]
        closed_frames = sum(1 for ear in recent_ears if ear < self.blink_threshold)
        return closed_frames / len(recent_ears)
    
    def print_analysis(self, df):
        """Prints statistical analysis and generates visualizations."""
        print("\n=== Fatigue Analysis Report ===")

        # Basic Statistics
        print(f"\n[1] Total Session Duration: {df['minutes_elapsed'].max()+1} mins")
        print(f"[2] Average PERCLOS: {df['perclos'].mean():.2f}")
        print(f"[3] Total Yawns: {df['yawn_count'].iloc[-1]}")

        # Correlation Analysis
        correlation_matrix = df.corr()
        print("\n[4] Correlation Matrix:")
        print(correlation_matrix)

        # Visualization (will save plots automatically)
        self.generate_visualizations(df)

    def generate_visualizations(self, df):
        """Generates matplotlib visualizations."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        # PERCLOS Trend
        plt.subplot(2, 2, 1)
        plt.plot(df['minutes_elapsed'], df['perclos'], 'b-')
        plt.axhline(y=PERCLOS_THRESHOLD, color='r', linestyle='--')
        plt.title('PERCLOS Over Time')
        plt.xlabel('Minutes')
        plt.ylabel('PERCLOS')

        # Yawn Frequency
        plt.subplot(2, 2, 2)
        df['yawns_per_min'] = df['yawn_count'].diff().fillna(0)
        plt.bar(df['minutes_elapsed'], df['yawns_per_min'])
        plt.title('Yawns per Minute')
        plt.xlabel('Minutes')

        # MAR Distribution
        plt.subplot(2, 2, 3)
        plt.hist(df['mar'], bins=20, edgecolor='black')
        plt.axvline(x=YAWN_ASPECT_RATIO_THRESHOLD, color='r', linestyle='--')
        plt.title('Mouth Aspect Ratio Distribution')

        # Correlation Heatmap
        plt.subplot(2, 2, 4)
        correlation_matrix = df.corr()
        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
        plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
        plt.title('Feature Correlations')

        plt.tight_layout()

        plot_filename = os.path.join(self.output_dir,
                                f"analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_filename)
        print(f"\nVisualizations saved to: {plot_filename}")
        plt.close()

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
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mar = 0.0
        ear = 0.0
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            img_h, img_w = frame.shape[:2]
            
            # Calculate EAR and detect blinks
            ear = self.calculate_ear(landmarks)
            self.ear_history.append(ear)
            blink_detected = self.detect_blink(ear)
            
            # Calculate MAR and detect yawns
            mar = self.calculate_mar(landmarks)
            yawn_detected = self.detect_yawn(mar)
            
            # Draw landmarks and info
            if self.show_video:
                for landmark in landmarks:
                    x, y = int(landmark.x * img_w), int(landmark.y * img_h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Eye status
                eye_status = "EYES CLOSED" if blink_detected else "EYES OPEN"
                eye_color = (0, 0, 255) if blink_detected else (0, 255, 0)
                cv2.putText(frame, eye_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
                
                # Mouth status
                mouth_status = "YAWNING" if yawn_detected else ""
                mouth_color = (0, 0, 255) if yawn_detected else (0, 255, 0)
                cv2.putText(frame, mouth_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_color, 2)
                
                # Display metrics
                info_text = [
                    f"EAR: {ear:.2f} (Thresh: {self.blink_threshold:.2f})",
                    f"PERCLOS: {self.get_perclos():.2f}",
                    f"Blinks: {self.blink_count}",
                    f"Yawns: {self.yawn_count}",
                    f"MAR: {mar:.2f}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(frame, text, (10, 90 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Log data every frame (even if no face detected)
        self.current_session_data.append({
            'timestamp': time.time() - self.start_time,
            'ear': ear,
            'perclos': self.get_perclos(),
            'blink_count': self.blink_count,
            'yawn_count': self.yawn_count,
            'mar': mar,
            'eyes_closed': int(self.eyes_closed),
            'yawn_detected': int(self.yawn_detected)
        })
        
        return frame

    def save_session_data(self):
        """Save collected data with analysis and visualizations."""
        if not self.current_session_data:
            print("No data to save")
            return
            
        df = pd.DataFrame(self.current_session_data)
        
        # Add time-based aggregations needed for visualizations
        df['minutes_elapsed'] = df['timestamp'] // 60
        df['seconds_elapsed'] = df['timestamp'] // 10  # 10-second intervals
        
        # Save raw data
        raw_filename = os.path.join(self.output_dir, 
                                  f"fatigue_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(raw_filename, index=False)
        print(f"Raw data saved to: {raw_filename}")
        
        # Save aggregated data
        agg_df = df.groupby('minutes_elapsed').agg({
            'perclos': 'mean',
            'yawn_count': 'max',
            'mar': 'max',
            'blink_count': 'max',
            'eyes_closed': 'sum'
        }).reset_index()
        
        agg_filename = os.path.join(self.output_dir,
                                  f"aggregated_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        agg_df.to_csv(agg_filename, index=False)
        print(f"Aggregated data saved to: {agg_filename}")
        
        # Generate and save visualizations
        self.generate_visualizations(agg_df)
        
        # Print analysis and summary
        self.print_analysis(agg_df)
        self.print_summary_statistics(df)

    def run(self):
        """Main processing loop."""
        self.calibrate_blink_threshold()
        last_fps_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame)
            self.frame_count += 1
            frame_count += 1
            
            # Show FPS periodically
            if time.time() - last_fps_time >= 1:
                fps = frame_count / (time.time() - last_fps_time)
                print(f"FPS: {fps:.1f}")
                frame_count = 0
                last_fps_time = time.time()
            
            if self.show_video:
                cv2.imshow("Fatigue Detection", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        self.save_session_data()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = MediaPipeFatigueTracker()
    tracker.run()