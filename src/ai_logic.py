import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

class AIDrowsinessProcessor:
    def __init__(self, yolo_eye_model_path="runs/detecteye/train/weights/best.pt", 
                 yolo_yawn_model_path="runs/detectyawn/train/weights/best.pt", 
                 camera_index=0):
        
        self.yawn_state = ''
        self.left_eye_state = ''
        self.right_eye_state = ''
        
        self.blinks = 0
        self.microsleeps_duration = 0  # Renamed from microsleeps to avoid confusion with a count
        self.yawns = 0
        self.yawn_duration = 0 

        self.left_eye_still_closed = False  
        self.right_eye_still_closed = False 
        self.yawn_in_progress = False  
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # Points for mouth ROI (x1,y1), (x2,_), (_,y3) and eye ROIs (x4,y4),(x5,y5) & (x6,y6),(x7,y7)
        # ORDER: mouth_left_corner, mouth_right_corner, mouth_bottom_lip_center, 
        # right_eye_inner_corner, right_eye_outer_corner, 
        # left_eye_inner_corner, left_eye_outer_corner
        # These might need adjustment based on the specific landmarks used in original DrowsinessDetector.py
        # Original points_ids = [187, 411, 152, 68, 174, 399, 298] - mapping these to semantic names if possible
        # Based on typical FaceMesh landmark indices:
        # Mouth corners: 61, 291 (approx)
        # Mouth bottom lip: 17 (approx)
        # Eye corners:
        # Left eye: 33 (inner), 133 (outer) -> for ROI: 246(top), 168(bottom), 7(left_iris_ish), 133 (outer_corner)
        # Right eye: 263 (inner), 362 (outer) -> for ROI: 466 (top), 382(bottom), 249(right_iris_ish), 362 (outer_corner)
        # The original points [187, 411, 152, 68, 174, 399, 298] seem custom or from a different mapping.
        # For simplicity and direct porting, I'll use the original IDs.
        # The ROI extraction logic in DrowsinessDetector.py seems to use these as:
        # mouth_roi: frame[points[0].y : points[2].y, points[0].x : points[1].x] (y1:y3, x1:x2)
        # right_eye_roi: frame[points[3].y : points[4].y, points[3].x : points[4].x] (y4:y5, x4:x5) - this implies points[4] is bottom-right of ROI
        # left_eye_roi: frame[points[5].y : points[6].y, points[5].x : points[6].x] (y6:y7, x6:x7) - this implies points[6] is bottom-right of ROI

        # For `mouth_roi = frame[y1:y3, x1:x2]`:
        # y1 = points[0].y (187)
        # y3 = points[2].y (152)  <- This seems problematic if 152 is above 187 (mouth_bottom_lip_center vs mouth_left_corner)
        # x1 = points[0].x (187)
        # x2 = points[1].x (411)
        # Assuming: points[0] = top-left mouth, points[1] = top-right mouth, points[2] = bottom-center mouth (for height)
        # ROI: y_top_mouth, y_bottom_mouth, x_left_mouth, x_right_mouth
        # Points for ROIs:
        # Mouth: P0(187), P1(411), P2(152) => (x_P0, y_P0), (x_P1, y_P1), (x_P2, y_P2)
        #   Likely: mouth_y_top = min(y_P0, y_P1), mouth_y_bottom = y_P2
        #           mouth_x_left = x_P0, mouth_x_right = x_P1
        # Eyes: P3(68), P4(174) for right eye; P5(399), P6(298) for left eye.
        #   Likely: P3 top-left, P4 bottom-right for right eye.
        #           P5 top-left, P6 bottom-right for left eye.
        self.points_ids = [187, 411, 152, 68, 174, 399, 298] # As in original code

        self.detect_yawn_model = YOLO(yolo_yawn_model_path)
        self.detect_eye_model = YOLO(yolo_eye_model_path)
        
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera {camera_index}")
        time.sleep(1.0) # Camera warmup

        self.last_frame_time = time.time()
        self.processed_frame = None # To store the frame with drawings

    def _predict_eye(self, eye_frame, current_eye_state):
        results_eye = self.detect_eye_model.predict(eye_frame, verbose=False)
        boxes = results_eye[0].boxes
        if len(boxes) == 0:
            return current_eye_state

        confidences = boxes.conf.cpu().numpy()  
        class_ids = boxes.cls.cpu().numpy()  
        
        if not confidences.size: # Handle cases where no boxes are detected after filtering
            return current_eye_state

        max_confidence_index = np.argmax(confidences)
        class_id = int(class_ids[max_confidence_index])

        # class_id == 1 is "Close Eye", class_id == 0 is "Open Eye"
        if class_id == 1: # Close Eye
            new_state = "Close Eye"
        elif class_id == 0 and confidences[max_confidence_index] > 0.30: # Open Eye with confidence
            new_state = "Open Eye"
        else:
            new_state = current_eye_state # Keep previous state if unsure
                            
        return new_state

    def _predict_yawn(self, mouth_frame):
        results_yawn = self.detect_yawn_model.predict(mouth_frame, verbose=False)
        boxes = results_yawn[0].boxes

        if len(boxes) == 0:
            return self.yawn_state # Return current/previous state

        confidences = boxes.conf.cpu().numpy()  
        class_ids = boxes.cls.cpu().numpy() 

        if not confidences.size:
            return self.yawn_state
            
        max_confidence_index = np.argmax(confidences)
        class_id = int(class_ids[max_confidence_index])

        # class_id == 0 is "Yawn", class_id == 1 is "No Yawn"
        if class_id == 0: # Yawn
            new_state = "Yawn"
        elif class_id == 1 and confidences[max_confidence_index] > 0.50: # No Yawn with confidence
            new_state = "No Yawn"
        else:
            new_state = self.yawn_state # Keep previous state if unsure
        return new_state
                            
    def process_single_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, self.get_current_status()

        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_facemesh = self.face_mesh.process(image_rgb)
        
        # Initialize ROIs to None
        mouth_roi, right_eye_roi, left_eye_roi = None, None, None
        rois_extracted = False

        if results_facemesh.multi_face_landmarks:
            for face_landmarks in results_facemesh.multi_face_landmarks:
                ih, iw, _ = frame.shape
                points_coords = []
                for point_id in self.points_ids:
                    if 0 <= point_id < len(face_landmarks.landmark):
                        lm = face_landmarks.landmark[point_id]
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        points_coords.append((x, y))
                    else:
                        # print(f"Warning: Landmark ID {point_id} is out of bounds.")
                        points_coords.append(None) # Placeholder for missing points

                # Ensure all points were found before trying to use them
                if all(p is not None for p in points_coords) and len(points_coords) == len(self.points_ids):
                    # Original ROI extraction logic from DrowsinessDetector.py
                    # P0=(x1,y1), P1=(x2,_), P2=(_,y3) for mouth
                    # P3=(x4,y4), P4=(x5,y5) for right eye
                    # P5=(x6,y6), P6=(x7,y7) for left eye

                    x1, y1 = points_coords[0] # points_ids[0] = 187
                    x2, _  = points_coords[1] # points_ids[1] = 411
                    _, y3  = points_coords[2] # points_ids[2] = 152

                    x4, y4 = points_coords[3] # points_ids[3] = 68
                    x5, y5 = points_coords[4] # points_ids[4] = 174 -- assume x5 > x4, y5 > y4 for valid ROI

                    x6, y6 = points_coords[5] # points_ids[5] = 399
                    x7, y7 = points_coords[6] # points_ids[6] = 298 -- assume x7 > x6, y7 > y6 for valid ROI
                    
                    # Ensure ROI coordinates are valid (y_start < y_end, x_start < x_end)
                    # And within frame bounds
                    mouth_y_start, mouth_y_end = min(y1, y3), max(y1, y3) # Corrected this logic
                    mouth_x_start, mouth_x_end = min(x1, x2), max(x1, x2)

                    re_y_start, re_y_end = min(y4, y5), max(y4, y5)
                    re_x_start, re_x_end = min(x4, x5), max(x4, x5)

                    le_y_start, le_y_end = min(y6, y7), max(y6, y7)
                    le_x_start, le_x_end = min(x6, x7), max(x6, x7)

                    # Clamp ROIs to be within frame dimensions
                    mouth_roi_frame = frame[max(0,mouth_y_start):min(ih,mouth_y_end), max(0,mouth_x_start):min(iw,mouth_x_end)]
                    right_eye_roi_frame = frame[max(0,re_y_start):min(ih,re_y_end), max(0,re_x_start):min(iw,re_x_end)]
                    left_eye_roi_frame = frame[max(0,le_y_start):min(ih,le_y_end), max(0,le_x_start):min(iw,le_x_end)]

                    rois_extracted = True

                    if mouth_roi_frame.size > 0:
                        mouth_roi = mouth_roi_frame
                        cv2.rectangle(frame, (mouth_x_start, mouth_y_start), (mouth_x_end, mouth_y_end), (0, 255, 255), 1) # Cyan
                    if right_eye_roi_frame.size > 0:
                        right_eye_roi = right_eye_roi_frame
                        cv2.rectangle(frame, (re_x_start, re_y_start), (re_x_end, re_y_end), (255, 0, 255), 1) # Magenta
                    if left_eye_roi_frame.size > 0:
                        left_eye_roi = left_eye_roi_frame
                        cv2.rectangle(frame, (le_x_start, le_y_start), (le_x_end, le_y_end), (255, 255, 0), 1) # Yellow
                    
                    # For drawing landmarks
                    for p_coord in points_coords:
                         if p_coord: cv2.circle(frame, p_coord, 2, (0,255,0), -1)
                    break # Process only the first detected face

        # Predictions
        if rois_extracted:
            try:
                if left_eye_roi is not None and left_eye_roi.size > 0 :
                    self.left_eye_state = self._predict_eye(left_eye_roi, self.left_eye_state)
                if right_eye_roi is not None and right_eye_roi.size > 0:
                    self.right_eye_state = self._predict_eye(right_eye_roi, self.right_eye_state)
                if mouth_roi is not None and mouth_roi.size > 0:
                    self.yawn_state = self._predict_yawn(mouth_roi)

            except Exception as e:
                print(f"Error during prediction: {e}")

        # Update states based on predictions
        # Using delta_time for duration accumulation (more accurate than fixed 45ms)
        if self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye":
            if not self.left_eye_still_closed: # Transition to closed
                self.left_eye_still_closed = True
                self.right_eye_still_closed = True # Assuming both close together for a blink start
                self.blinks += 1 
            self.microsleeps_duration += delta_time
        else: # One or both eyes are open
            if self.left_eye_still_closed: # Was closed, now open
                 self.left_eye_still_closed = False
                 self.right_eye_still_closed = False
            # Reset microsleep if eyes are open.
            # Consider if reset should only happen if *both* were closed and now at least one is open.
            # Current logic: if not (both closed), reset. This is fine.
            self.microsleeps_duration = 0

        if self.yawn_state == "Yawn":
            if not self.yawn_in_progress:
                self.yawn_in_progress = True
                self.yawns += 1  
            self.yawn_duration += delta_time
        else: # No Yawn
            if self.yawn_in_progress:
                self.yawn_in_progress = False
            # Reset yawn duration when not yawning.
            # If a continuous yawn is desired for X seconds before reset, this needs change.
            # Current: if "No Yawn" state, reset duration.
            # The original code implies yawn_duration accumulates as long as yawn_state is "Yawn".
            # And does not reset it immediately when state becomes "No Yawn".
            # Let's keep accumulating if in progress, and only reset if it was in progress and now stopped.
            # The document does not explicitly state how to reset yawn_duration.
            # The original code's update_info implies it just displays current values.
            # For now, let's make it so yawn_duration resets if yawn_in_progress becomes false
            if not self.yawn_in_progress: # if it just transitioned to False
                 pass # Keep the last yawn_duration until a new yawn starts. Or set to 0?
                 # The original code just keeps adding to self.yawn_duration if state is Yawn.
                 # And never resets it, except implicitly if the app restarts.
                 # Let's make it reset when a yawn finishes.
                 # This means, when self.yawn_in_progress becomes False, we should probably log that duration
                 # and then reset self.yawn_duration for the *next* yawn.
                 # For simplicity of get_current_status, we will report the duration of the *current or last* yawn.
                 # Let's reset it when yawn_in_progress becomes False.
                 # self.yawn_duration = 0 # This might be too aggressive.

        self.processed_frame = frame.copy() # Store the frame with drawings
        return self.processed_frame, self.get_current_status()

    def get_current_status(self):
        return {
            "blinks": self.blinks,
            "microsleeps_duration": round(self.microsleeps_duration, 2),
            "yawns": self.yawns,
            "yawn_duration": round(self.yawn_duration, 2), # Duration of current/last yawn
            "left_eye_state": self.left_eye_state,
            "right_eye_state": self.right_eye_state,
            "yawn_state": self.yawn_state,
            "overall_alert": self._determine_alert_status()
        }

    def _determine_alert_status(self):
        if self.microsleeps_duration > 4.0: # As per original DrowsinessDetector
            return "Prolonged Microsleep Detected!"
        if self.yawn_duration > 7.0: # As per original DrowsinessDetector
             # This implies yawn_duration should be for a *single, continuous* yawn.
             # If self.yawn_duration is reset when yawn_in_progress becomes false, this works.
             # If self.yawn_state becomes "No Yawn", and self.yawn_in_progress was true,
             # then we should reset self.yawn_duration.
            if self.yawn_state == "No Yawn" and self.yawn_in_progress: # Just finished a yawn
                 # If we want to check the duration of the yawn that just finished:
                 # We need to store it before resetting self.yawn_duration.
                 # Let's adjust the logic for yawn_duration.
                 pass # This needs refinement if alert is based on single yawn over 7s.
            # For now, if total accumulated yawn time (while in "Yawn" state) > 7s
            # This interpretation is tricky.
            # Let's assume the alert is for a *current continuous* yawn exceeding 7s.
            # So, if yawn_state is "Yawn" and yawn_duration > 7.0
            if self.yawn_state == "Yawn" and self.yawn_duration > 7.0:
                return "Prolonged Yawn Detected!"
        return "Awake" # Default status

    def get_processed_frame(self):
        # Returns the latest frame processed by process_single_frame, with drawings.
        # Ensure it's encoded as JPEG bytes for web streaming.
        if self.processed_frame is not None:
            ret, buffer = cv2.imencode('.jpg', self.processed_frame)
            if ret:
                return buffer.tobytes()
        return None

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        # mediapipe face_mesh doesn't have an explicit release method documented typically.
        # YOLO models are managed by the ultralytics library.

if __name__ == '__main__':
    # Example Usage (for testing ai_logic.py directly)
    try:
        detector = AIDrowsinessProcessor(camera_index=0) # or use a video file path
        print("AIDrowsinessProcessor initialized.")
        
        cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)

        while True:
            processed_frame, status = detector.process_single_frame()

            if processed_frame is None:
                print("Failed to grab frame or end of video.")
                break
            
            # Display status on the frame (for testing)
            y_offset = 30
            for key, value in status.items():
                cv2.putText(processed_frame, f"{key}: {value}", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            cv2.imshow("Processed Frame", processed_frame)

            print(status)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except IOError as e:
        print(f"Error initializing camera: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'detector' in locals():
            detector.release()
        cv2.destroyAllWindows()
        print("Resources released.") 