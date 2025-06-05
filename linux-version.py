import cv2
import time
import numpy as np
import pandas as pd
import datetime
import os
import statistics
import tkinter as tk
from collections import deque

# Fedora-specific optimizations
PERCLOS_THRESHOLD = 0.8
PERCLOS_DURATION = 20
YAWN_ASPECT_RATIO_THRESHOLD = 0.6
BLINK_CONSEC_FRAMES = 3
CALIBRATION_DURATION = 3
YAWN_MIN_FRAMES = 15
CAMERA_INDEX = 0
THRESHOLD_SCALING = 0.80
SACCADE_THRESHOLD = 0.003
SACCADE_SMOOTHING_WINDOW = 2

class FedoraFatigueTracker:
    def __init__(self, output_dir="fatigue_data_fedora", show_video=True):
        self.show_landmarks = True
        
        # Fedora-specific mediapipe import with reduced precision
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                refine_landmarks=False,  # Disabled for performance
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe Face Mesh initialized (Fedora optimized)")
        except ImportError as e:
            print(f"Error: {e}\nInstall MediaPipe with: pip install mediapipe --no-cache-dir")
            exit()

        # Optimized camera setup for Fedora
        self.cap = self._init_fedora_camera()
        if not self.cap:
            print("No working camera found. Try v4l2:")
            print("sudo dnf install v4l-utils")
            print("v4l2-ctl --list-devices")
            exit()

        self.show_video = show_video
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Tracking variables with optimized sizes
        self.frame_count = 0
        self.start_time = time.time()
        self.ear_history = deque(maxlen=900)  # 30 sec * 30 fps
        self.ear_smoothed = deque(maxlen=3)   # Smaller window for smoothing
        self.blink_count = 0
        self.yawn_count = 0
        self.session_start_time = datetime.datetime.now()
        self.current_session_data = []

        # Detection variables
        self.blink_threshold = None
        self.blink_frame_counter = 0
        self.eyes_closed = False
        self.calibrated = False
        self.yawn_frame_counter = 0
        self.yawn_detected = False
        self.prev_iris_pos = None
        self.saccade_count = 0
        self.saccade_detected = False
        self.iris_history = deque(maxlen=3)

        # Frame rate control
        self.frame_rate = 30  # Target FPS
        self.frame_interval = 1.0 / self.frame_rate
        self.last_frame_time = time.time()

    def _init_fedora_camera(self):
        """Initialize camera with v4l2 backend and optimized settings for Fedora"""
        # Try v4l2 first (common on Linux)
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        
        if not cap.isOpened():
            # Fallback to any available backend
            cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if cap.isOpened():
            # Fedora typically has better performance with these resolutions
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Optimize camera settings for Linux
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            print(f"Fedora camera initialized at 640x480")
            return cap
        return None

    def calculate_ear(self, landmarks):
        """Optimized EAR calculation with reduced precision"""
        try:
            # Use only 4 points per eye for faster calculation
            left_eye = [landmarks[33], landmarks[160], landmarks[158], landmarks[133]]
            right_eye = [landmarks[362], landmarks[385], landmarks[387], landmarks[263]]
            
            def fast_ear(eye):
                A = np.linalg.norm([eye[1].x - eye[3].x, eye[1].y - eye[3].y])
                B = np.linalg.norm([eye[0].x - eye[2].x, eye[0].y - eye[2].y])
                return A / B if B > 0 else 0
            
            left_ear = fast_ear(left_eye)
            right_ear = fast_ear(right_eye)
            return (left_ear + right_ear) / 2.0
        except:
            return 0

    def calculate_mar(self, landmarks):
        """Optimized MAR calculation"""
        try:
            # Use only 3 points for mouth detection
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            left_corner = landmarks[61]
            
            A = abs(upper_lip.y - lower_lip.y)  # Vertical distance only
            D = abs(left_corner.x - landmarks[291].x)
            return A / D if D > 0 else 0
        except:
            return 0

    def calibrate_blink_threshold(self):
        """Faster calibration with reduced frames"""
        print("Calibrating... Keep eyes open")
        ear_values = []
        start_time = time.time()
        
        while time.time() - start_time < CALIBRATION_DURATION:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (320, 240))  # Smaller frame for calibration
            
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                try:
                    ear = self.calculate_ear(results.multi_face_landmarks[0].landmark)
                    ear_values.append(ear)
                except:
                    continue
                
            cv2.putText(frame, f"Calibrating... {int(CALIBRATION_DURATION - (time.time() - start_time))}s", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Calibration", frame)
            cv2.waitKey(1)

        if ear_values:
            self.blink_threshold = statistics.mean(ear_values) * THRESHOLD_SCALING
            print(f"Calibration complete. EAR threshold: {self.blink_threshold:.3f}")
            self.calibrated = True
        else:
            print("Calibration failed - using default threshold")
            self.blink_threshold = 0.2
            
        cv2.destroyAllWindows()

    def detect_blink(self, ear):
        if not self.calibrated:
            self.calibrate_blink_threshold()
            
        # Simplified smoothing
        self.ear_smoothed.append(ear)
        smoothed_ear = statistics.mean(self.ear_smoothed) if self.ear_smoothed else ear
        
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
        if len(self.ear_history) < PERCLOS_DURATION * self.frame_rate:
            return 0.0
            
        recent_ears = list(self.ear_history)[-int(PERCLOS_DURATION * self.frame_rate):]
        closed_frames = sum(1 for ear in recent_ears if ear < self.blink_threshold)
        return closed_frames / len(recent_ears)

    def process_frame(self, frame):
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return frame  # Skip processing to maintain frame rate
            
        self.last_frame_time = current_time
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (480, 360))
        
        results = self.face_mesh.process(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
        mar = 0.0
        ear = 0.0
        saccade_detected = False 
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            img_h, img_w = frame.shape[:2]
            
            ear = self.calculate_ear(landmarks)
            self.ear_history.append(ear)
            blink_detected = self.detect_blink(ear)
            
            mar = self.calculate_mar(landmarks)
            yawn_detected = self.detect_yawn(mar)

            # Saccade detection
            saccade_detected = self.detect_saccade(landmarks)
            
            fatigue = self.is_fatigued(window_sec=60)
            
            if self.show_video:
                if self.show_landmarks:
                    for landmark in landmarks:
                        x, y = int(landmark.x * img_w), int(landmark.y * img_h)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Simplified status display
                status_text = [
                    f"EAR: {ear:.2f}",
                    f"Blinks: {self.blink_count}",
                    f"Yawns: {self.yawn_count}",
                    f"FPS: {1/(time.time()-current_time):.1f}"
                ]
                
                for i, text in enumerate(status_text):
                    cv2.putText(frame, text, (10, 30 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                if fatigue:
                    cv2.putText(frame, "FATIGUE WARNING!", (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        self.current_session_data.append({
            'timestamp': time.time() - self.start_time,
            'ear': ear,
            'perclos': self.get_perclos(),
            'blink_count': self.blink_count,
            'yawn_count': self.yawn_count,
            'mar': mar,
            'eyes_closed': int(self.eyes_closed),
            'yawn_detected': int(self.yawn_detected),
            'saccade_detected': int(saccade_detected)
        })
        
        return frame

    def run(self):
        """Main loop with frame rate control"""
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
                cv2.imshow("Fatigue Detection (Fedora)", processed_frame)
                cv2.setWindowProperty("Fatigue Detection (Fedora)", cv2.WND_PROP_TOPMOST, 1)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.show_landmarks = not self.show_landmarks
                elif key == ord('r'):
                    self.__init__(output_dir=self.output_dir, show_video=self.show_video)
                    print("Reset complete")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Fedora Fatigue Tracker - Optimized for Linux")
    tracker = FedoraFatigueTracker()
    tracker.run()