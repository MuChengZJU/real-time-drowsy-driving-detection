import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

class AIDrowsinessProcessor:
    def __init__(self, yolo_eye_model_path="runs/detecteye/train/weights/best.pt", 
                 yolo_yawn_model_path="runs/detectyawn/train/weights/best.pt"):
        
        self.yawn_state = ''
        self.left_eye_state = ''
        self.right_eye_state = ''
        
        self.blinks = 0
        self.microsleeps_duration = 0
        self.yawns = 0
        self.yawn_duration = 0 

        self.left_eye_still_closed = False  
        self.right_eye_still_closed = False 
        self.yawn_in_progress = False  
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.points_ids = [187, 411, 152, 68, 174, 399, 298]

        self.detect_yawn_model = YOLO(yolo_yawn_model_path)
        self.detect_eye_model = YOLO(yolo_eye_model_path)
        
        self.last_frame_time = time.time()
        self.last_processed_frame_with_drawings = None

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

    def process_frame(self, frame):
        if frame is None:
             return None, self.get_current_status()
             
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        drawable_frame = frame.copy()

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_facemesh = self.face_mesh.process(image_rgb)
        
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
                        points_coords.append(None)

                if all(p is not None for p in points_coords) and len(points_coords) == len(self.points_ids):
                    x1, y1 = points_coords[0]
                    x2, _  = points_coords[1]
                    _, y3  = points_coords[2]
                    x4, y4 = points_coords[3]
                    x5, y5 = points_coords[4]
                    x6, y6 = points_coords[5]
                    x7, y7 = points_coords[6]
                    
                    mouth_y_start, mouth_y_end = min(y1, y3), max(y1, y3)
                    mouth_x_start, mouth_x_end = min(x1, x2), max(x1, x2)
                    re_y_start, re_y_end = min(y4, y5), max(y4, y5)
                    re_x_start, re_x_end = min(x4, x5), max(x4, x5)
                    le_y_start, le_y_end = min(y6, y7), max(y6, y7)
                    le_x_start, le_x_end = min(x6, x7), max(x6, x7)

                    mouth_roi_frame = frame[max(0,mouth_y_start):min(ih,mouth_y_end), max(0,mouth_x_start):min(iw,mouth_x_end)]
                    right_eye_roi_frame = frame[max(0,re_y_start):min(ih,re_y_end), max(0,re_x_start):min(iw,re_x_end)]
                    left_eye_roi_frame = frame[max(0,le_y_start):min(ih,le_y_end), max(0,le_x_start):min(iw,le_x_end)]

                    rois_extracted = True

                    if mouth_roi_frame.size > 0:
                        mouth_roi = mouth_roi_frame
                        cv2.rectangle(drawable_frame, (mouth_x_start, mouth_y_start), (mouth_x_end, mouth_y_end), (0, 255, 255), 1)
                    if right_eye_roi_frame.size > 0:
                        right_eye_roi = right_eye_roi_frame
                        cv2.rectangle(drawable_frame, (re_x_start, re_y_start), (re_x_end, re_y_end), (255, 0, 255), 1)
                    if left_eye_roi_frame.size > 0:
                        left_eye_roi = left_eye_roi_frame
                        cv2.rectangle(drawable_frame, (le_x_start, le_y_start), (le_x_end, le_y_end), (255, 255, 0), 1)
                    
                    for p_coord in points_coords:
                         if p_coord: cv2.circle(drawable_frame, p_coord, 2, (0,255,0), -1)
                    break

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

        if self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye":
            if not self.left_eye_still_closed:
                self.left_eye_still_closed = True
                self.right_eye_still_closed = True
                self.blinks += 1 
            self.microsleeps_duration += delta_time
        else:
            if self.left_eye_still_closed:
                 self.left_eye_still_closed = False
                 self.right_eye_still_closed = False
            self.microsleeps_duration = 0

        if self.yawn_state == "Yawn":
            if not self.yawn_in_progress:
                self.yawn_in_progress = True
                self.yawns += 1  
            self.yawn_duration += delta_time
        else:
            if self.yawn_in_progress:
                self.yawn_in_progress = False
            if not self.yawn_in_progress: 
                 self.yawn_duration = 0

        self.last_processed_frame_with_drawings = drawable_frame 
        return self.last_processed_frame_with_drawings, self.get_current_status()

    def get_current_status(self):
        return {
            "blinks": self.blinks,
            "microsleeps_duration": round(self.microsleeps_duration, 2),
            "yawns": self.yawns,
            "yawn_duration": round(self.yawn_duration, 2),
            "left_eye_state": self.left_eye_state,
            "right_eye_state": self.right_eye_state,
            "yawn_state": self.yawn_state,
            "overall_alert": self._determine_alert_status()
        }

    def _determine_alert_status(self):
        if self.microsleeps_duration > 4.0:
            return "Prolonged Microsleep Detected!"
        if self.yawn_duration > 7.0:
            if self.yawn_state == "No Yawn" and self.yawn_in_progress:
                pass
            if self.yawn_state == "Yawn" and self.yawn_duration > 7.0:
                return "Prolonged Yawn Detected!"
        return "Awake"

    def get_processed_frame(self):
        if self.last_processed_frame_with_drawings is not None:
            ret, buffer = cv2.imencode('.jpg', self.last_processed_frame_with_drawings)
            if ret:
                return buffer.tobytes()
        return None

    def release(self):
        print("Releasing AI Processor resources (models loaded).")
        pass

if __name__ == '__main__':
    print("Testing AIDrowsinessProcessor (requires manual camera capture loop)")
    CAMERA_INDEX = 0
    test_cap = cv2.VideoCapture(CAMERA_INDEX)
    if not test_cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_INDEX}")
        exit()
    
    try:
        detector = AIDrowsinessProcessor()
        print("AIDrowsinessProcessor initialized for testing.")
        
        cv2.namedWindow("Processed Frame - Test", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = test_cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            processed_frame_drawable, status = detector.process_frame(frame)

            if processed_frame_drawable is None:
                print("Processing failed.")
                continue 
            
            y_offset = 30
            for key, value in status.items():
                cv2.putText(processed_frame_drawable, f"{key}: {value}", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            cv2.imshow("Processed Frame - Test", processed_frame_drawable)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("'q' pressed, exiting test loop.")
                break
    
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'detector' in locals():
            detector.release()
        if test_cap.isOpened():
            test_cap.release()
        cv2.destroyAllWindows()
        print("Test resources released.") 