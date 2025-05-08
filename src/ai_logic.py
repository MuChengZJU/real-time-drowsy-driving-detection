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
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5)
        self.points_ids = [187, 411, 152, 68, 174, 399, 298] # mouth_left, mouth_right, mouth_center_top, right_eye_outer, right_eye_inner, left_eye_outer, left_eye_inner

        self.detect_yawn_model = YOLO(yolo_yawn_model_path)
        self.detect_eye_model = YOLO(yolo_eye_model_path)
        
        self.last_frame_time = time.time()
        self.last_processed_frame_with_drawings = None

    def _parse_eye_prediction_result(self, result, current_eye_state):
        boxes = result.boxes
        if len(boxes) == 0:
            return current_eye_state

        confidences = boxes.conf.cpu().numpy()  
        class_ids = boxes.cls.cpu().numpy()  
        
        if not confidences.size: 
            return current_eye_state

        max_confidence_index = np.argmax(confidences)
        class_id = int(class_ids[max_confidence_index])

        if class_id == 1: 
            new_state = "Close Eye"
        elif class_id == 0 and confidences[max_confidence_index] > 0.30:
            new_state = "Open Eye"
        else:
            new_state = current_eye_state
                            
        return new_state

    def _predict_yawn(self, mouth_frame):
        results_yawn = self.detect_yawn_model.predict(mouth_frame, verbose=False)
        boxes = results_yawn[0].boxes
        if len(boxes) == 0: return self.yawn_state
        confidences = boxes.conf.cpu().numpy()  
        class_ids = boxes.cls.cpu().numpy() 
        if not confidences.size: return self.yawn_state
        max_confidence_index = np.argmax(confidences)
        class_id = int(class_ids[max_confidence_index])
        if class_id == 0: new_state = "Yawn"
        elif class_id == 1 and confidences[max_confidence_index] > 0.50: new_state = "No Yawn"
        else: new_state = self.yawn_state
        return new_state

    def process_frame(self, frame):
        if frame is None:
             # Return empty draw_elements if frame is None
             return {}, self.get_current_status()
             
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        # Prepare draw_elements dictionary
        draw_elements = {
            "face_landmarks_coords": [],
            "mouth_roi_bbox": None,
            "right_eye_roi_bbox": None,
            "left_eye_roi_bbox": None
        }

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # Optimization
        results_facemesh = self.face_mesh.process(image_rgb)
        
        mouth_roi_frame, right_eye_roi_frame, left_eye_roi_frame = None, None, None
        rois_extracted = False
        eye_rois_for_batch = []
        eye_roi_types = [] # To keep track of 'left' or 'right' for batched results

        if results_facemesh.multi_face_landmarks:
            for face_landmarks in results_facemesh.multi_face_landmarks:
                ih, iw, _ = frame.shape
                points_coords_for_draw = [] 
                
                # Extract specific landmark coordinates for ROIs
                # This assumes points_ids are correctly ordered for mouth, right eye, left eye
                # points_ids = [mouth_l, mouth_r, mouth_top_center, re_outer, re_inner, le_outer, le_inner]
                # indices:        0        1          2               3          4         5          6

                # Collect all landmark points for drawing
                for i, landmark in enumerate(face_landmarks.landmark):
                    x, y = int(landmark.x * iw), int(landmark.y * ih)
                    if i in self.points_ids: # Only store designated points for drawing if needed, or store all
                         points_coords_for_draw.append((x,y))
                draw_elements["face_landmarks_coords"] = points_coords_for_draw


                # Use landmark indices directly for ROI definition if they are stable
                # Or use a more robust method if landmark indices for specific features can vary
                
                # Example: if self.points_ids refers to specific points to define ROIs
                # Ensure points_ids has enough elements and they correspond to the features
                # For simplicity, we'll assume direct landmark access or a helper to get specific points
                
                # Mouth ROI (example using specific landmarks if known, e.g., 13, 14, 78, 308 for a wider mouth area)
                # For this example, let's use the provided points_ids if they are meant for ROI definition
                # Mouth: points_ids[0], points_ids[1], points_ids[2]
                # Right Eye: points_ids[3], points_ids[4]
                # Left Eye: points_ids[5], points_ids[6]

                # This section needs careful mapping of self.points_ids to actual ROI definitions
                # The original code used points_coords which was derived from self.points_ids
                # We will reconstruct a similar logic for ROI definition
                
                # Reconstruct points_coords from all landmarks for ROI definition
                # This is less efficient than directly using specific landmarks if their IDs are known and stable
                # For now, sticking to the logic similar to original:
                all_lm_coords = []
                for point_id in self.points_ids: # Iterate through the *indices* in points_ids
                    if 0 <= point_id < len(face_landmarks.landmark):
                        lm = face_landmarks.landmark[point_id]
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        all_lm_coords.append((x,y))
                    else:
                        all_lm_coords.append(None) # Should not happen if points_ids are valid indices

                if all(p is not None for p in all_lm_coords) and len(all_lm_coords) == len(self.points_ids):
                    # Assuming original points_ids mapping:
                    # [187(mouth_l), 411(mouth_r), 152(mouth_top_c), 68(re_out), 174(re_in), 399(le_out), 298(le_in)]
                    m_l, m_r, m_tc = all_lm_coords[0], all_lm_coords[1], all_lm_coords[2]
                    re_o, re_i = all_lm_coords[3], all_lm_coords[4]
                    le_o, le_i = all_lm_coords[5], all_lm_coords[6]

                    # Mouth ROI (more robust might be to use min/max of relevant points)
                    # Simplified: use mouth_l, mouth_r for width, mouth_tc for an anchor point for height
                    mouth_y_start, mouth_y_end = min(m_l[1], m_r[1], m_tc[1]), max(m_l[1], m_r[1], m_tc[1]) # Example, needs refinement
                    mouth_x_start, mouth_x_end = min(m_l[0], m_r[0]), max(m_l[0], m_r[0])
                    # Adjust mouth ROI to be more representative, e.g., extend below m_tc
                    mouth_y_start = m_tc[1] - (m_tc[1] - min(m_l[1], m_r[1])) // 2 # Approx
                    mouth_y_end = max(m_l[1], m_r[1]) + (max(m_l[1], m_r[1]) - m_tc[1]) //2 # Approx
                    
                    # Eye ROIs
                    re_y_start, re_y_end = min(re_o[1], re_i[1]), max(re_o[1], re_i[1])
                    re_x_start, re_x_end = min(re_o[0], re_i[0]), max(re_o[0], re_i[0])
                    le_y_start, le_y_end = min(le_o[1], le_i[1]), max(le_o[1], le_i[1])
                    le_x_start, le_x_end = min(le_o[0], le_i[0]), max(le_o[0], le_i[0])

                    # Add padding to ROIs (optional, can improve detection)
                    padding = 5 # pixels
                    mouth_x_start, mouth_y_start = max(0, mouth_x_start - padding), max(0, mouth_y_start - padding)
                    mouth_x_end, mouth_y_end = min(iw, mouth_x_end + padding), min(ih, mouth_y_end + padding)
                    re_x_start, re_y_start = max(0, re_x_start - padding), max(0, re_y_start - padding)
                    re_x_end, re_y_end = min(iw, re_x_end + padding), min(ih, re_y_end + padding)
                    le_x_start, le_y_start = max(0, le_x_start - padding), max(0, le_y_start - padding)
                    le_x_end, le_y_end = min(iw, le_x_end + padding), min(ih, le_y_end + padding)

                    mouth_roi_frame = frame[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
                    right_eye_roi_frame = frame[re_y_start:re_y_end, re_x_start:re_x_end]
                    left_eye_roi_frame = frame[le_y_start:le_y_end, le_x_start:le_x_end]
                    
                    rois_extracted = True

                    if mouth_roi_frame.size > 0:
                        draw_elements["mouth_roi_bbox"] = (mouth_x_start, mouth_y_start, mouth_x_end, mouth_y_end)
                    
                    if right_eye_roi_frame.size > 0:
                        eye_rois_for_batch.append(right_eye_roi_frame)
                        eye_roi_types.append("right")
                        draw_elements["right_eye_roi_bbox"] = (re_x_start, re_y_start, re_x_end, re_y_end)
                    
                    if left_eye_roi_frame.size > 0:
                        eye_rois_for_batch.append(left_eye_roi_frame)
                        eye_roi_types.append("left")
                        draw_elements["left_eye_roi_bbox"] = (le_x_start, le_y_start, le_x_end, le_y_end)
                    
                    # Keypoints for drawing are already in draw_elements["face_landmarks_coords"]
                    # based on ALL landmarks, not just self.points_ids. If only points_ids are needed:
                    # draw_elements["face_landmarks_coords"] = all_lm_coords # if all_lm_coords are the specific points
                    
                    # If we want to draw ALL face mesh points:
                    all_face_mesh_points_for_draw = []
                    for landmark in face_landmarks.landmark:
                        x, y = int(landmark.x * iw), int(landmark.y * ih)
                        all_face_mesh_points_for_draw.append((x,y))
                    draw_elements["face_landmarks_coords"] = all_face_mesh_points_for_draw


                    break # Process only the first detected face

        if rois_extracted:
            try:
                if eye_rois_for_batch:
                    eye_results = self.detect_eye_model.predict(eye_rois_for_batch, verbose=False)
                    for i, result in enumerate(eye_results):
                        eye_type = eye_roi_types[i]
                        if eye_type == "right":
                            self.right_eye_state = self._parse_eye_prediction_result(result, self.right_eye_state)
                        elif eye_type == "left":
                             self.left_eye_state = self._parse_eye_prediction_result(result, self.left_eye_state)

                if mouth_roi_frame is not None and mouth_roi_frame.size > 0: # Use mouth_roi_frame here
                    self.yawn_state = self._predict_yawn(mouth_roi_frame)

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
            # Reset yawn_duration only if it was in progress and now it's not
            # Or, if it's "No Yawn" and not in progress, ensure duration is 0
            # The original logic resets yawn_duration if not (yawn_in_progress).
            # This means if it was never in progress, it stays 0. If it *was* and stops, it becomes 0.
            if not self.yawn_in_progress: # Corrected condition
                 self.yawn_duration = 0

        return draw_elements, self.get_current_status()

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
        # Check yawn state correctly with yawn_in_progress for alert
        if self.yawn_in_progress and self.yawn_duration > 7.0 : # Alert only if currently yawning and duration is long
             return "Prolonged Yawn Detected!"
        # if self.yawn_duration > 7.0: # Original logic
        #     if self.yawn_state == "No Yawn" and self.yawn_in_progress: # This case seems unlikely with current state updates
        #         pass
        #     if self.yawn_state == "Yawn" and self.yawn_duration > 7.0:
        #         return "Prolonged Yawn Detected!"
        return "Awake"

    def draw_overlays(self, frame_to_draw_on, status_info, draw_elements):
        overlay_frame = frame_to_draw_on.copy()

        if draw_elements:
            if draw_elements.get("mouth_roi_bbox"):
                x1, y1, x2, y2 = draw_elements["mouth_roi_bbox"]
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            
            if draw_elements.get("right_eye_roi_bbox"):
                x1, y1, x2, y2 = draw_elements["right_eye_roi_bbox"]
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

            if draw_elements.get("left_eye_roi_bbox"):
                x1, y1, x2, y2 = draw_elements["left_eye_roi_bbox"]
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
            
            # Draw face landmarks
            if draw_elements.get("face_landmarks_coords"):
                for p_coord in draw_elements["face_landmarks_coords"]:
                    if p_coord: # Ensure coordinate is not None
                        cv2.circle(overlay_frame, p_coord, 1, (0,255,0), -1) # Smaller radius for all landmarks

        # Draw status text from status_info
        if status_info:
            y_offset = 30
            for key, value in status_info.items():
                cv2.putText(overlay_frame, f"{key}: {value}", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
        
        return overlay_frame

    def release(self):
        print("Releasing AI Processor resources (models loaded).")
        # No specific resources like camera to release here in this class anymore.
        # Models are managed by YOLO and FaceMesh instances, will be GC'd.
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

            # Process frame to get AI data and status
            draw_data, status = detector.process_frame(frame)

            # Create a drawable frame and add overlays
            # The draw_overlays function expects the original frame to draw upon
            frame_with_overlays = detector.draw_overlays(frame, status, draw_data)
            
            cv2.imshow("Processed Frame - Test", frame_with_overlays)

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