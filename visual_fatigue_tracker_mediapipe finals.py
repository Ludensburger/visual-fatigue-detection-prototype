import cv2
import time
import numpy as np
import pandas as pd
import datetime
import os
import statistics
# Removed unused import
from collections import deque

# LEGEND:
# - STAR ⭐: Important logic or calculations or something to Study further
# - BOX 📦: Newly added Feature tags: optional, new feature, TWIST
# - WORK IN PROGRESS 🚧: Feature that is not fully implemented yet (optional during presentation)

# --- Configuration & Constants ---
PERCLOS_THRESHOLD = 0.6  # Percentage of time eyes are closed to indicate fatigue, Ca be adjusted since the program on initialization calibrates based on the user's eye aspect ratio
PERCLOS_DURATION = 10    # Calculate PERCLOS over last 60 seconds
YAWN_ASPECT_RATIO_THRESHOLD = 0.5 # Threshold for yawn detection, Can be adjusted since the program on initialization calibrates based on the user's mouth aspect ratio
BLINK_CONSEC_FRAMES = 2  # Minimum frames for blink detection
CALIBRATION_DURATION = 3 # Seconds for calibration
YAWN_MIN_FRAMES = 10     # Minimum frames to count as a yawn
CAMERA_INDEX = 0        # Default camera index, can be changed if multiple cameras are available
THRESHOLD_SCALING = 0.85 # Scaling factor for blink threshold, can be adjusted
SACCADE_THRESHOLD = 0.01  # You might lower this to 0.005 for low-res/low-FPS setups
SACCADE_SMOOTHING_WINDOW = 3  # ✅ Try 3 to 5 for noise reduction


class MediaPipeFatigueTracker:


    # Initialize the MediaPipeFatigueTracker class
    # This is the constructor for the class — runs when you create an instance.
    # self refers to the instance of the class.
    # output_dir is a default parameter to define where to save data.
    # show_video toggles whether to display the video feed live.
    def __init__(self, output_dir="fatigue_data_mediapipe", show_video=True):


        # Handle import error gracefully
        try:
            # Import MediaPipe
            # Then use the Face Mesh module/model
            import mediapipe as mp


            # Access face_mesh from mediapipe, which is a pre-trained model for facial landmark detection.
            # then we store it in self.mp_face_mesh
            self.mp_face_mesh = mp.solutions.face_mesh

            # then we create an instance of the FaceMesh, which is our face tracker
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False, # Set to True for static images
                # refine_landmarks=True,   # Refine landmarks for better accuracy (Optional)
                max_num_faces=1, # Maximum number of faces to detect, for this case just 1 user
                min_detection_confidence=0.5, # Threshold for detection confidence
                min_tracking_confidence=0.5 # Threshold for tracking confidence
            )

            # Then we confirm that the MediaPipe Face Mesh is initialized
            print("MediaPipe Face Mesh initialized.")

        # If the import fails (e.g., you didn't install MediaPipe), it prints the error and exits the program.
        except ImportError as e:
            print(f"MediaPipe not installed. Please install it (pip install mediapipe). Error: {e}")
            exit()

        # Camera setup
        # Calls your internal method _init_camera() to find and open a webcam.
        # But since we have the option to select the camera index, we can set it to 0 or 1, later on we can discuss this feature.
        self.cap = self._init_camera()

        # Simply exit the camea is not initialized properly
        if not self.cap:
            print("No working camera found.")
            exit()

        # Video settings
        self.show_video = show_video # Stores wheter to display the video feed in a class attribute, so other methods can check this setting.
        self.output_dir = output_dir # Stores the output directory for saving data.

        # Ensures the output directory exists
        os.makedirs(self.output_dir, exist_ok=True) 
        # If the directory already exists, nothing happens (exist_ok=True prevents an error).
        # If it doesn't exist, it is created.

        # Tracking variables
        self.frame_count = 0 # Counts the number of frames processed.
        self.start_time = time.time() # measure the elapsed/running time
        self.ear_history = deque(maxlen=1800)  # 60 sec * 30 fps # STAR ⭐
        self.ear_smoothed = deque(maxlen=5)    # For smoothing
        self.blink_count = 0 # for storing the number of blinks detected
        self.yawn_count = 0 # for storing the number of yawns detected
        self.session_start_time = datetime.datetime.now() # for storing the session start time
        self.current_session_data = [] # for storing the current session data (currently just an optional feature and not always accurate)

        # Blink detection
        self.blink_threshold = None # Threshold for blink detection, will be set during calibration(when u run the program for the first time) [Eyes: Opened or Closed]
        self.blink_frame_counter = 0 # Frame counter for blink detection
        self.eyes_closed = False # Flag to indicate if eyes are closed 
        self.calibrated = False # Flag to indicate if calibration is done

        # Yawn detection
        # Used to track MAR-related activity over frames to detect yawns.
        self.yawn_frame_counter = 0 # Frame counter for yawn detection
        self.yawn_detected = False # Flag to indicate if yawn is detected
        # Used to track MAR-related activity over frames to detect yawns.

        # Get actual frame rate
        # This is the FPS value which is also printed in the console every second.
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        if self.frame_rate <= 0:
            self.frame_rate = 30  # Fallback


        # Saccade detection
        self.prev_iris_pos = None
        self.saccade_count = 0
        self.saccade_detected = False
        self.iris_history = deque(maxlen=3)  # Or 5 for heavier smoothing

    # Initialize camera HELPER METHOD
    def _init_camera(self):
        """Initialize camera using the selected index."""

        # cap is a variable that will hold the camera object
        # cv2.VideoCapture is a function from OpenCV to access the camera
            # Try to open the webcam using the specified index and backend
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        # Checks if the camera is opened successfully
        if cap.isOpened():

            # Set camera resolution
            # This is the most optimal resolution for the camera, to get the best performance and FPS (peaks at 20-30 FPS)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Next optional settings can be
            # but they are not recommended since they can cause performance issues, since laptop only has integrated GPU

            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Set width to 1280 pixels
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # Set height to 720 pixels

            # Indicate successful camera initialization
            print(f"Using camera index {CAMERA_INDEX}")
            return cap # Return the VideoCapture object
        return None # Return if camera can't be opened

    def calculate_ear(self, landmarks):
        """Calculates the average Eye Aspect Ratio (EAR) using MediaPipe facial landmarks."""

        # Handle gracefully if the landmarks are not available
        try:
        # Define the landmark indices for both eyes based on the MediaPipe Face Mesh model.
        # Each eye is represented by 6 key points.
        
        # Left eye landmark indices: 33, 160, 158, 133, 153, 144
        # as P1, P2, P3, P4, P5, P6

            left_eye = [
                landmarks[33], landmarks[160], landmarks[158],
                landmarks[133], landmarks[153], landmarks[144]
            ]

            # Right eye landmark indices: 362, 385, 387, 263, 380, 373
            right_eye = [
                landmarks[362], landmarks[385], landmarks[387],
                landmarks[263], landmarks[380], landmarks[373]
            ]
            
        # Helper function to calculate EAR for a given eye using its 6 landmarks.
        # EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        # p1–p6 refer to specific points around the eye.
            def eye_aspect_ratio(eye):
                # Compute vertical distances between eye landmarks
                A = np.linalg.norm([eye[1].x - eye[5].x, eye[1].y - eye[5].y])
                B = np.linalg.norm([eye[2].x - eye[4].x, eye[2].y - eye[4].y])
                # Compute horizontal distance
                C = np.linalg.norm([eye[0].x - eye[3].x, eye[0].y - eye[3].y])

                # Return EAR if denominator is non-zero, else return 0
                return (A + B) / (2.0 * C) if C > 0 else 0

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # Return the average EAR of both eyes
            # EAR = (left_ear + right_ear) / 2 eyes
            return (left_ear + right_ear) / 2.0
        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return 0

    # The MAR (Mouth Aspect Ratio) is similar in concept to EAR (Eye Aspect Ratio),
    # but it measures mouth openness — useful for detecting yawns.
    # This version uses 4 landmark points:
    # - Upper lip:     point 13  → acts as p2
    # - Lower lip:     point 14  → acts as p6
    # - Left corner:   point 61  → acts as p1
    # - Right corner:  point 291 → acts as p4

    # Formula:
    #   MAR = ||p2 - p6|| / ||p1 - p4||
    #        = vertical distance (A) / horizontal distance (D)

    # If the vertical distance increases (mouth opens), MAR gets higher — indicating a possible yawn.

    def calculate_mar(self, landmarks):
        """Calculate the Mouth Aspect Ratio (MAR) using selected facial landmarks for yawn detection."""

        # Handle gracefully if the landmarks are not available
        try:
            # Landmark positions from MediaPipe face mesh
            upper_lip = landmarks[13] # Upper lip P2
            lower_lip = landmarks[14] # Lower lip P6
            left_corner = landmarks[61] # Left corner P1
            right_corner = landmarks[291] # Right corner P4

            # Vertical distance between upper and lower lip (mouth opening)
            A = np.linalg.norm([upper_lip.x - lower_lip.x, upper_lip.y - lower_lip.y]) 

            # Horizontal distance between mouth corners (mouth width)
            D = np.linalg.norm([left_corner.x - right_corner.x, left_corner.y - right_corner.y])

            # Return the MAR value, or 0 if the horizontal distance is 0 to avoid division error
            # MAR = A / D
            # Where A is the vertical distance and D is the horizontal distance
            # If D is 0, return 0 to avoid division by zero
            return A / D if D > 0 else 0 

        except Exception as e:
            # Fallback in case of invalid/missing landmarks
            print(f"Error calculating MAR: {e}")
            return 0

    # Runs as soon as the program starts
    # This method calibrates the EAR threshold for the current user. (Reads it from the camera)
    # This makes it DYNAMIC and ADAPTABLE to different users.

    # Because not everyone has the same facial features, more specifically the same EAR and MAR.
    # This is done by asking the user to keep their eyes open for a few seconds. Or their neutral facial expression.

    # Then the program will just SCAN the user's face and calculate the possible Threshold for blink Threshold via EAR.

    def calibrate_blink_threshold(self):
        """Calibrate EAR threshold for the current user."""
        print("Calibrating blink detection... Please keep eyes open")

        
        ear_values = [] # Initializes an empty list to store EAR values measured over the calibration period.
        start_time = time.time() # Records the current time to track how long calibration has been running.
        
        while time.time() - start_time < CALIBRATION_DURATION: # Runs a loop repeatedly until the calibration duration (in seconds) has passed.
           
            # Reads a frame from the camera (self.cap). ret is a boolean showing if the frame was successfully captured; frame is the image data.
            ret, frame = self.cap.read()
            # If frame capture failed, skip this iteration and try again (don’t crash or process empty data).
            if not ret:
                continue
                
            # Just Mirror the frame for a better user experience
            frame = cv2.flip(frame, 1)

            # then Convert the frame to RGB format 
            # Because MediaPipe works with RGB images, not BGR (OpenCV default).
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Checks if any face landmarks were detected in the current frame.
            # If landmarks are detected, calculate EAR and append to ear_values.
            # If no face is detected, it skips the EAR calculation.
            if results.multi_face_landmarks:

                try:
                    # We finally calculate the EAR using the detected landmarks. For a single face.
                    ear = self.calculate_ear(results.multi_face_landmarks[0].landmark)

                    # then Adds the EAR value to the list of EAR measurements for calibration.
                    ear_values.append(ear)

                except Exception as e:
                    print(f"EAR calculation failed: {e}")
                    continue
                
                # Show calibration countdown
                # Calculates how many seconds are left in the calibration period. The countdown we see on the screen.
                remaining = CALIBRATION_DURATION - (time.time() - start_time)

                # Draws a countdown text on the frame so the user sees how much time is left.
                cv2.putText(frame, f"Calibrating... {int(remaining)}s", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                # Shows the camera feed with the countdown text in a window named “Calibration”.
                cv2.imshow("Calibration", frame)

                # Waits 1 millisecond for a keypress, allowing the frame to display smoothly without freezing the program.
                cv2.waitKey(1)

        # After the loop, checks if any EAR data was collected during calibration.
        if ear_values:
            # If so, Sets the blink threshold to 85% of the average open-eye EAR — this means EAR values below this likely indicate a blink.

            # Set threshold to 85% of normal open-eye EAR
            self.blink_threshold = statistics.mean(ear_values) * THRESHOLD_SCALING # Currently set to 0.85
            print(f"Calibration complete. EAR threshold: {self.blink_threshold:.3f}")
            self.calibrated = True # Marks the calibration as complete with a boolean flag.

        # just means that if no EAR values were collected because the user isnt visible, or the user blinked too much, or the camera was not working properly, or anything in between/etc
        # it indicates a calibration failure.
        # In this case, it sets a default threshold value (0.2) for blink detection.
        else:

            # If theres no face detected or the camera is black or any color that is not a face 
            # then print the error message and set the default threshold
            if not ear_values:
                print("Calibration failed - no EAR values collected")
            else:
                print("Calibration failed - using default threshold")
                self.blink_threshold = 0.2  # Fallback
            
        # Then closes the calibration window after calibration is done.
        cv2.destroyAllWindows()


    # This method is used to detect blinks using the EAR (Eye Aspect Ratio) value.
    def detect_blink(self, ear):
        """State machine for accurate blink detection."""

        # If the Program is not calibrated yet, it will call the calibration method.
        # This ensures that the system has a reliable blink threshold before detection.
        if not self.calibrated:
            self.calibrate_blink_threshold()
            

        
        # Smooth EAR values

        # these 2 lines just means that
        # There may be times that the EAR values can jump around due to noise. 
        # So you keep a running list (ear_smoothed) and calculate the average for smoother, more stable detection.
        self.ear_smoothed.append(ear)
        smoothed_ear = statistics.mean(self.ear_smoothed) if self.ear_smoothed else ear
        # So calculate a lot then just find the average
        
        # STAR ⭐
        # State machine logic
        # If the smoothed EAR falls below the blink threshold, it probably means the eye is closing. 
        # So you increment a frame counter (blink_frame_counter) to track how long the eye has been closed.
        if smoothed_ear < self.blink_threshold:
            self.blink_frame_counter += 1
            
            # STAR ⭐
            # If the eye wasn’t already marked as closed, and it's been below the threshold for BLINK_CONSEC_FRAMES consecutive frames, 
            # then it's officially considered a blink.
            if not self.eyes_closed and self.blink_frame_counter >= BLINK_CONSEC_FRAMES:
                self.eyes_closed = True
                
        else:
            # If the smoothed EAR goes back above the threshold:
            if self.eyes_closed:
                self.blink_count += 1
                self.eyes_closed = False
                # Count it as a blink (increment blink_count) and reset the state to "open."
                
            # Reset the counter if the eyes are not closed anymore.
            self.blink_frame_counter = 0
            
        # Returns whether the eyes are currently closed (can be useful for other logic, like drowsiness detection or triggering alerts).
        return self.eyes_closed


    # This method is used to detect yawns using the MAR (Mouth Aspect Ratio) value.
    # it follows a similar logic to the blink detection.
    # It uses a threshold to determine if the mouth is open wide enough to be considered a yawn.
    # If the MAR exceeds the threshold for a certain number of frames, it counts as a yawn.
    def detect_yawn(self, mar):
        """Detect yawns using Mouth Aspect Ratio."""

        # STAR ⭐
        if mar > YAWN_ASPECT_RATIO_THRESHOLD:
            self.yawn_frame_counter += 1
            
            # STAR ⭐
            if not self.yawn_detected and self.yawn_frame_counter >= YAWN_MIN_FRAMES:
                self.yawn_detected = True
                self.yawn_count += 1
        else:
            self.yawn_frame_counter = 0
            self.yawn_detected = False
            
        return self.yawn_detected

    # PERCLOS (PERcentage of eye CLOSure)
    def get_perclos(self):
        """Calculate PERCLOS over the specified duration."""

        # Handle edge case
        # If we don't have enough EAR data (i.e., less than the number of frames needed to cover PERCLOS_DURATION seconds), 
        # return 0.0 since the calculation wouldn't be reliable.
        if len(self.ear_history) < PERCLOS_DURATION * self.frame_rate:
            return 0.0
            
        # this line retrieves the most recent EAR values needed to calculate PERCLOS over a defined time window. 
        # The use of a negative index allows for easy selection of the most recent data points 
        # without needing to know the exact length of the ear_history list.
        recent_ears = list(self.ear_history)[-int(PERCLOS_DURATION * self.frame_rate):]

        # this line efficiently counts how many frames in a recent sequence of frames show a blink,
        # based on the EAR value falling below a defined threshold.
        closed_frames = sum(1 for ear in recent_ears if ear < self.blink_threshold)
        return closed_frames / len(recent_ears)
    

    # And with all the logic and calculations done, we can now print the analysis of the session.
    # This method prints a summary of the session statistics, including total session duration, average PERCLOS, and total yawns.
    # It also generates visualizations (commented out since i dont really understand much of it for not its a work in progress feature) 
    # and saves the data to CSV files.
    def print_analysis(self, df):
        """Prints statistical analysis and generates visualizations."""
        print("\n=== Fatigue Analysis Report ===\n")

        # Basic Statistics

        # Prints total session length in minutes, assuming minutes_elapsed column counts minutes from zero, so adding 1 for actual total.
        print(f"[1] Total Session Duration: {df['minutes_elapsed'].max()+1} mins")

        # Prints the average PERCLOS value during the session, rounded to 2 decimals.
        print(f"[2] Average PERCLOS: {df['perclos'].mean():.2f}")

        # Prints total yawns recorded by reading the last value of yawn_count (assuming it accumulates over time).
        print(f"[3] Total Yawns: {df['yawn_count'].iloc[-1]}")

        # Correlation Analysis
        # Calculates the correlation matrix of all numerical columns in the DataFrame, useful for exploring relationships.
        correlation_matrix = df.corr()

        # WORK IN PROGRESS 🚧
        #  print the correlation matrix for deeper analysis.
        # print("\n[4] Correlation Matrix:")
        # print(correlation_matrix)
        # WORK IN PROGRESS 🚧

        # WORK IN PROGRESS 🚧
        # Visualization (will save plots automatically)
        # self.generate_visualizations(df)
        # WORK IN PROGRESS 🚧


    # WORK IN PROGRESS 🚧
    # def generate_visualizations(self, df):
    #     """Generates matplotlib visualizations."""
    #     import matplotlib.pyplot as plt

    #     plt.figure(figsize=(12, 6))

    #     # PERCLOS Trend
    #     plt.subplot(2, 2, 1)
    #     plt.plot(df['minutes_elapsed'], df['perclos'], 'b-')
    #     plt.axhline(y=PERCLOS_THRESHOLD, color='r', linestyle='--')
    #     plt.title('PERCLOS Over Time')
    #     plt.xlabel('Minutes')
    #     plt.ylabel('PERCLOS')

    #     # Yawn Frequency
    #     plt.subplot(2, 2, 2)
    #     df['yawns_per_min'] = df['yawn_count'].diff().fillna(0)
    #     plt.bar(df['minutes_elapsed'], df['yawns_per_min'])
    #     plt.title('Yawns per Minute')
    #     plt.xlabel('Minutes')

    #     # MAR Distribution
    #     plt.subplot(2, 2, 3)
    #     plt.hist(df['mar'], bins=20, edgecolor='black')
    #     plt.axvline(x=YAWN_ASPECT_RATIO_THRESHOLD, color='r', linestyle='--')
    #     plt.title('Mouth Aspect Ratio Distribution')

    #     # Correlation Heatmap
    #     plt.subplot(2, 2, 4)
    #     correlation_matrix = df.corr()
    #     plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    #     plt.colorbar()
    #     plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
    #     plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
    #     plt.title('Feature Correlations')

    #     plt.tight_layout()

    #     plot_filename = os.path.join(self.output_dir,
    #                             f"analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    #     plt.savefig(plot_filename)
    #     print(f"\nVisualizations saved to: {plot_filename}")
    #     plt.close()
    # WORK IN PROGRESS 🚧


    def print_summary_statistics(self, df):
        """Calculates and prints summary statistics from a DataFrame."""

        print("\n--- Session Summary Statistics (MediaPipe) ---")

        # Basic Statistics
        # Prints the total duration of the session by subtracting the earliest from latest timestamp, formatted to 2 decimals.
        print(f"Total Session Duration: {df['timestamp'].max() - df['timestamp'].min():.2f} seconds")

        # Prints total yawns by taking the last value of the yawn_count column (assumed cumulative).
        print(f"Total Yawns: {df['yawn_count'].iloc[-1]}")

        # Calculates and prints average frame rate (FPS) based on the recorded frame count and elapsed real time since self.start_time.
        #  Prints 0 if elapsed time is zero or negative (to avoid divide-by-zero).
        if (time.time() - self.start_time) > 0:
            print(f"Average Frame Rate: {self.frame_count / (time.time() - self.start_time):.2f} FPS")
        else:
            print("Average Frame Rate: 0.00 FPS")

        # PERCLOS Statistics (if you manage to approximate EAR)

        # If the DataFrame has a non-empty perclos column, it prints the average and maximum PERCLOS values with 3 decimals.
        if 'perclos' in df.columns and not df['perclos'].empty:
            perclos_values = df['perclos']
            print(f"Average PERCLOS: {perclos_values.mean():.3f}")
            print(f"Max PERCLOS: {perclos_values.max():.3f}")


            # Calculates and prints percentage of time where PERCLOS exceeds a threshold. If no data, prints 0%.
            if len(perclos_values) > 0:
                print(f"Time spent above PERCLOS threshold ({PERCLOS_THRESHOLD:.2f}): {len(perclos_values[perclos_values > PERCLOS_THRESHOLD]) / len(perclos_values) * 100:.2f}%")
            else:
                print(f"Time spent above PERCLOS threshold ({PERCLOS_THRESHOLD:.2f}): 0.00%")

        # MAR Statistics
        # If MAR data exists, prints average and max MAR values with 3 decimals.
        if 'mar' in df.columns and not df['mar'].empty:
            mar_values = df['mar']
            print(f"Average MAR: {mar_values.mean():.3f}")
            print(f"Max MAR: {mar_values.max():.3f}")


        # BOX 📦
        # Note: Blink and saccade detection with MediaPipe requires different approaches based on landmark movements.
        print("Blink and saccade detection might require different logic with MediaPipe.")


    # BOX 📦
    def detect_saccade(self, iris_landmarks):
        """
        Detects rapid eye movement (saccades) based on iris landmark displacement.
        
        ⚠️ NOTE:
        - Works best with ≥20 FPS.
        - Adjust SACCADE_THRESHOLD for your resolution/FPS.
        - Might need observation-based tweaks.
        
        ✅ Post-test checklist:
        [ ] Is smoothing window optimal?
        [ ] Is threshold tuned to your camera?
        [ ] Is detection triggering only on legit movements?


        """

        # Skip if frame rate is too low
        if self.frame_rate < 15:
            print("⚠️ Low FPS may reduce saccade detection accuracy.")

        # Get current iris center position (average of both irises for stability)
        left_iris = iris_landmarks[468]
        right_iris = iris_landmarks[473]
        iris_x = (left_iris.x + right_iris.x) / 2.0
        iris_y = (left_iris.y + right_iris.y) / 2.0
        current_pos = (iris_x, iris_y)

        # Initialize smoothing buffer if not yet
        if not hasattr(self, "iris_history"):
            self.iris_history = deque(maxlen=SACCADE_SMOOTHING_WINDOW)

        self.iris_history.append(current_pos)

        if len(self.iris_history) >= 2:
            dx = self.iris_history[-1][0] - self.iris_history[-2][0]
            dy = self.iris_history[-1][1] - self.iris_history[-2][1]
            distance = np.sqrt(dx**2 + dy**2)

            # If sudden spike in distance, count as saccade
            if distance > SACCADE_THRESHOLD:
                self.saccade_count += 1
                return True

        return False

    def process_frame(self, frame):
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mar = 0.0
        ear = 0.0
        saccade_detected = False 
        
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

            # Saccade detection
            saccade_detected = self.detect_saccade(landmarks)
            
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
            'yawn_detected': int(self.yawn_detected),
            'saccade_detected': int(saccade_detected)
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
        # raw_filename = os.path.join(self.output_dir, 
        #                           f"fatigue_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        # df.to_csv(raw_filename, index=False)
        # print(f"Raw data saved to: {raw_filename}")
        
        # Save aggregated data
        agg_df = df.groupby('minutes_elapsed').agg({
            'perclos': 'mean',
            'yawn_count': 'max',
            'mar': 'max',
            'blink_count': 'max',
            'eyes_closed': 'sum'
        }).reset_index()
        
        # agg_filename = os.path.join(self.output_dir,
        #                           f"aggregated_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        # agg_df.to_csv(agg_filename, index=False)
        # print(f"Aggregated data saved to: {agg_filename}")
        
        # # Generate and save visualizations
        # self.generate_visualizations(agg_df)
        
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

    @staticmethod
    def list_available_cameras(max_tested=5):
        """Prints available camera indices."""
        print("Scanning for available cameras...")
        available = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"Camera index {i}: AVAILABLE")
                available.append(i)
                cap.release()
            else:
                print(f"Camera index {i}: Not found")
        if not available:
            print("No cameras found.")
        print("\nSet CAMERA_INDEX = <number> at the top of the script to select your camera.\n")
        return available

    @staticmethod
    def select_camera():
        """Auto-select webcam if only one is available, prompt if both."""
        available_cams = MediaPipeFatigueTracker.list_available_cameras(max_tested=2)
        cam_names = {0: "Webcam", 1: "IRIUN Webcam"}
        named_cams = [f"{idx} ({cam_names.get(idx, 'Unknown')})" for idx in available_cams]
    
        if len(available_cams) == 0:
            print("No available cameras. Exiting.")
            exit()
        elif available_cams == [0]:
            print("Only default webcam found. Using camera index 0 (Webcam).")
            return 0
        elif available_cams == [1]:
            print("Only IRIUN webcam found. Using camera index 1 (IRIUN Webcam).")
            return 1
        elif set(available_cams) == {0, 1}:
            while True:
                try:
                    print("Available cameras:")
                    for idx in available_cams:
                        print(f"  {idx}: {cam_names.get(idx, 'Unknown')}")
                    user_input = input(f"Enter camera index to use [{', '.join(named_cams)}]: ")
                    idx = int(user_input)
                    if idx in available_cams:
                        return idx
                    else:
                        print(f"Invalid index. Choose from {named_cams}.")
                except ValueError:
                    print("Please enter a valid integer.")
        else:
            # Fallback: just use the first available
            print(f"Using camera index {available_cams[0]} ({cam_names.get(available_cams[0], 'Unknown')}).")
            return available_cams[0]
    
if __name__ == "__main__":
    CAMERA_INDEX = MediaPipeFatigueTracker.select_camera()
    tracker = MediaPipeFatigueTracker()
    tracker.run()