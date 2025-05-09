import cv2
import numpy as np
# from ultralytics import YOLO # No longer directly needed for inference
import mediapipe as mp
import time
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter # Fallback for full TensorFlow

class AIDrowsinessProcessor:
    def __init__(self, 
                 tflite_eye_model_path="model/tflite/eye_detect_320_int8.tflite", 
                 tflite_yawn_model_path="model/tflite/yawn_detect_320_int8.tflite",
                 camera_index=0 # Added camera_index, though not used in this class directly
                 ):
        
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
        # These points_ids seem to be indices into the full list of 468/478 landmarks.
        # For eye ROI: mediapipe provides specific contours e.g., LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # For mouth ROI: MOUTH_OUTLINE = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        # Using min/max of these contour points is more robust for ROI.
        # The current points_ids = [187, 411, 152, 68, 174, 399, 298] might be for specific single points.
        # We will use direct landmark indices for ROIs later for better clarity.
        self.points_ids = [187, 411, 152, 68, 174, 399, 298] 


        # Load TFLite models
        self.eye_interpreter = Interpreter(model_path=tflite_eye_model_path)
        self.eye_interpreter.allocate_tensors()
        self.eye_input_details = self.eye_interpreter.get_input_details()
        self.eye_output_details = self.eye_interpreter.get_output_details()
        self.eye_input_height = self.eye_input_details[0]['shape'][1]
        self.eye_input_width = self.eye_input_details[0]['shape'][2]
        print(f"Eye TFLite model: {tflite_eye_model_path}, Input shape: (1, {self.eye_input_height}, {self.eye_input_width}, 3)")
        print(f"Eye TFLite Input Details: dtype={self.eye_input_details[0]['dtype']}, quantization={self.eye_input_details[0]['quantization']}")
        print(f"Eye TFLite Output Details: {self.eye_output_details}")


        self.yawn_interpreter = Interpreter(model_path=tflite_yawn_model_path)
        self.yawn_interpreter.allocate_tensors()
        self.yawn_input_details = self.yawn_interpreter.get_input_details()
        self.yawn_output_details = self.yawn_interpreter.get_output_details()
        self.yawn_input_height = self.yawn_input_details[0]['shape'][1]
        self.yawn_input_width = self.yawn_input_details[0]['shape'][2]
        print(f"Yawn TFLite model: {tflite_yawn_model_path}, Input shape: (1, {self.yawn_input_height}, {self.yawn_input_width}, 3)")
        print(f"Yawn TFLite Input Details: dtype={self.yawn_input_details[0]['dtype']}, quantization={self.yawn_input_details[0]['quantization']}")
        print(f"Yawn TFLite Output Details: {self.yawn_output_details}")
        
        self.last_frame_time = time.time()
        self.last_processed_frame_with_drawings = None

    def _preprocess_image_for_tflite(self, roi_frame, target_height, target_width, input_details):
        if roi_frame is None or roi_frame.size == 0:
            return None
        img_resized = cv2.resize(roi_frame, (target_width, target_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # Model usually expects RGB
        
        input_data = np.expand_dims(img_rgb, axis=0) # Add batch dimension

        if input_details[0]['dtype'] == np.float32:
            # Normalize to [0, 1] if input is float32
            input_data = input_data.astype(np.float32) / 255.0
        elif input_details[0]['dtype'] == np.uint8:
            # For UINT8 quantized models, input is typically uint8 in range [0, 255]
            # No normalization needed if the model expects this range directly.
            # Ultralytics exported INT8 TFLite models might still expect float input that they dequantize/quantize internally,
            # or they might expect uint8. This needs to be verified by checking input_details['quantization']
            # For now, we assume if dtype is uint8, it's already scaled.
            input_data = input_data.astype(np.uint8)
        else:
            # Handle other types if necessary, though float32 and uint8 are most common
            raise ValueError(f"Unsupported TFLite input data type: {input_details[0]['dtype']}")
            
        return input_data

    def _parse_tflite_output(self, output_data, original_roi_shape, conf_threshold=0.30):
        """
        Parses the output from a TFLite YOLO model.
        output_data is typically [1, num_boxes, 5 + num_classes]
        (x_center, y_center, width, height, confidence, class_probs...)
        Coordinates are normalized to the TFLite model's input dimensions.
        """
        # print(f"[AI_LOGIC DEBUG] _parse_tflite_output: Raw TFLite output_data shape: {output_data.shape}") # Reduced verbosity

        best_class_id = -1
        max_confidence = -1.0

        if output_data.shape[1] < output_data.shape[2] and (output_data.shape[1] == (4 + 2) or output_data.shape[1] == (4+1+2)):
             # print(f"[AI_LOGIC DEBUG] Transposing output from {output_data.shape} to (0, 2, 1)") # Reduced verbosity
             output_data = np.transpose(output_data, (0, 2, 1))
        
        # print(f"[AI_LOGIC DEBUG] Final TFLite output_data shape for parsing: {output_data.shape}") # Reduced verbosity

        detections = output_data[0] 
        output_dim_size = detections.shape[1]
        # print(f"[AI_LOGIC DEBUG] Number of detections: {detections.shape[0]}, Output_dim_size (per detection): {output_dim_size}") # Reduced verbosity

        for i in range(detections.shape[0]):
            detection = detections[i] 
            current_confidence = 0
            current_class_id = -1

            if output_dim_size == 6: 
                class_scores = detection[4:] 
                current_confidence = np.max(class_scores)
                current_class_id = np.argmax(class_scores)
            elif output_dim_size == 7: 
                object_confidence = detection[4]
                class_scores = detection[5:]
                current_confidence = object_confidence * np.max(class_scores) if np.max(class_scores) > 0 else object_confidence 
                current_class_id = np.argmax(class_scores)
            else:
                # print(f"[AI_LOGIC DEBUG] Branch 3 (dim={output_dim_size}): Unexpected output dimension. Skipping parsing for this detection.") # Reduced verbosity
                continue 

            if current_confidence > conf_threshold and current_confidence > max_confidence:
                max_confidence = current_confidence
                best_class_id = current_class_id
        
        # print(f"[AI_LOGIC DEBUG] _parse_tflite_output result: BestClassID: {best_class_id}, MaxConfidence: {max_confidence}") # Reduced verbosity
        return best_class_id, max_confidence


    def _predict_eye_state_tflite(self, eye_roi_frame):
        # t_pred_eye_enter = time.time() # Optional: for even finer detail
        if eye_roi_frame is None or eye_roi_frame.size == 0:
            return "Unknown"

        input_data = self._preprocess_image_for_tflite(eye_roi_frame, self.eye_input_height, self.eye_input_width, self.eye_input_details)
        if input_data is None:
            return "Unknown"
        
        t_before_invoke = time.time()
        self.eye_interpreter.set_tensor(self.eye_input_details[0]['index'], input_data)
        self.eye_interpreter.invoke()
        output_data = self.eye_interpreter.get_tensor(self.eye_output_details[0]['index'])
        t_after_invoke = time.time()
        print(f"[PERF_PROFILE_DETAIL] _predict_eye_state_tflite: invoke took: {(t_after_invoke - t_before_invoke)*1000:.2f} ms")
        
        class_id, confidence = self._parse_tflite_output(output_data, eye_roi_frame.shape, conf_threshold=0.3)

        if class_id == 1: return "Close Eye"
        elif class_id == 0: return "Open Eye"
        return "Unknown"

    def _predict_yawn_tflite(self, mouth_roi_frame):
        # t_pred_yawn_enter = time.time() # Optional: for even finer detail
        if mouth_roi_frame is None or mouth_roi_frame.size == 0:
            return self.yawn_state 

        input_data = self._preprocess_image_for_tflite(mouth_roi_frame, self.yawn_input_height, self.yawn_input_width, self.yawn_input_details)
        if input_data is None:
            return self.yawn_state

        t_before_invoke = time.time()
        self.yawn_interpreter.set_tensor(self.yawn_input_details[0]['index'], input_data)
        self.yawn_interpreter.invoke()
        output_data = self.yawn_interpreter.get_tensor(self.yawn_output_details[0]['index'])
        t_after_invoke = time.time()
        print(f"[PERF_PROFILE_DETAIL] _predict_yawn_tflite: invoke took: {(t_after_invoke - t_before_invoke)*1000:.2f} ms")

        class_id, confidence = self._parse_tflite_output(output_data, mouth_roi_frame.shape, conf_threshold=0.5)
        
        if class_id == 0: return "Yawn"
        elif class_id == 1: return "No Yawn" 
        return self.yawn_state


    def process_frame(self, frame):
        t_pf_enter = time.time()
        # print(f"[PERF_PROFILE] process_frame: ENTER at {t_pf_enter}") # Reduced verbosity

        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        draw_elements = {'face_landmarks': None, 'left_eye_roi': None, 'right_eye_roi': None, 'mouth_roi': None}
        
        t_mp_start = time.time()
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t_mp_end = time.time()
        print(f"[PERF_PROFILE] process_frame: MediaPipe mesh processing took: {(t_mp_end - t_mp_start)*1000:.2f} ms")

        left_eye_roi_frame, right_eye_roi_frame, mouth_roi_frame = None, None, None
        
        t_roi_ext_start = time.time()
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0] 
            draw_elements['face_landmarks'] = face_landmarks 

            LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 373, 374, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            MOUTH_OUTLINE_CONTOUR = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

            img_h, img_w = frame.shape[:2]

            def get_roi_bbox_from_landmarks(landmark_list, indices, img_width, img_height, padding=5):
                if not landmark_list or not indices: return None
                # Ensure landmark indices are within bounds before accessing
                valid_indices = [i for i in indices if i < len(landmark_list.landmark)]
                if not valid_indices: return None

                points_data = [(landmark_list.landmark[i].x * img_width, landmark_list.landmark[i].y * img_height) for i in valid_indices]
                if not points_data: return None 
                
                points = np.array(points_data, dtype=np.int32)
                
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(img_width, x_max + padding)
                y_max = min(img_height, y_max + padding)

                if x_max > x_min and y_max > y_min:
                    return (x_min, y_min, x_max - x_min, y_max - y_min) 
                return None

            left_eye_bbox = get_roi_bbox_from_landmarks(face_landmarks, LEFT_EYE_CONTOUR, img_w, img_h, padding=10)
            right_eye_bbox = get_roi_bbox_from_landmarks(face_landmarks, RIGHT_EYE_CONTOUR, img_w, img_h, padding=10)
            mouth_bbox = get_roi_bbox_from_landmarks(face_landmarks, MOUTH_OUTLINE_CONTOUR, img_w, img_h, padding=15)
            
            if left_eye_bbox:
                x, y, w, h = left_eye_bbox
                left_eye_roi_frame = frame[y:y+h, x:x+w]
                draw_elements['left_eye_roi'] = (x,y,w,h)
            if right_eye_bbox:
                x, y, w, h = right_eye_bbox
                right_eye_roi_frame = frame[y:y+h, x:x+w]
                draw_elements['right_eye_roi'] = (x,y,w,h)
            if mouth_bbox:
                x, y, w, h = mouth_bbox
                mouth_roi_frame = frame[y:y+h, x:x+w]
                draw_elements['mouth_roi'] = (x,y,w,h)
        t_roi_ext_end = time.time()
        print(f"[PERF_PROFILE] process_frame: ROI extraction (conditional) took: {(t_roi_ext_end - t_roi_ext_start)*1000:.2f} ms")

        t_eye_pred_start = time.time()
        if left_eye_roi_frame is not None:
            self.left_eye_state = self._predict_eye_state_tflite(left_eye_roi_frame)
        else: self.left_eye_state = "Unknown"
        
        if right_eye_roi_frame is not None:
            self.right_eye_state = self._predict_eye_state_tflite(right_eye_roi_frame)
        else: self.right_eye_state = "Unknown"
        t_eye_pred_end = time.time()
        print(f"[PERF_PROFILE] process_frame: Both Eye state predictions took: {(t_eye_pred_end - t_eye_pred_start)*1000:.2f} ms")

        t_yawn_pred_start = time.time()
        if mouth_roi_frame is not None:
            current_yawn_detection = self._predict_yawn_tflite(mouth_roi_frame)
        else: current_yawn_detection = "No Yawn" 
        t_yawn_pred_end = time.time()
        print(f"[PERF_PROFILE] process_frame: Yawn state prediction took: {(t_yawn_pred_end - t_yawn_pred_start)*1000:.2f} ms")
        
        t_state_logic_start = time.time()
        if self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye":
            if not self.left_eye_still_closed and not self.right_eye_still_closed: # Both just closed
                self.blinks += 1
                self.blink_start_time = current_time # Record time when blink starts
            self.left_eye_still_closed = True
            self.right_eye_still_closed = True
            # Microsleep check: if eyes remain closed for a certain duration
            if hasattr(self, 'blink_start_time'): # Check if blink_start_time is initialized
                 if current_time - self.blink_start_time > 0.5: # Threshold for microsleep (e.g., 0.5 seconds)
                    self.microsleeps_duration += delta_time # Accumulate microsleep duration
        else: # One or both eyes are open
            self.left_eye_still_closed = False
            self.right_eye_still_closed = False
            # If they were in a microsleep, it ends. Duration is already accumulated.
            # Reset blink_start_time when eyes open to correctly time the next closure
            if hasattr(self, 'blink_start_time'):
                del self.blink_start_time

        # Yawns
        if current_yawn_detection == "Yawn":
            if not self.yawn_in_progress:
                self.yawns += 1
                self.yawn_start_time = current_time
                self.yawn_in_progress = True
            # Accumulate duration as long as yawn is detected in the current frame
            # This means self.yawn_duration will be the duration of the ongoing or last completed yawn *within the detection period*
            if hasattr(self, 'yawn_start_time'): # Ensure yawn_start_time exists
                self.yawn_duration = current_time - self.yawn_start_time             
        else: # No Yawn detected in the current frame
            if self.yawn_in_progress: # Yawn just ended
                 self.yawn_in_progress = False
                 # self.yawn_duration would hold the duration of the just-ended yawn.
                 # If you want to reset it for the *next* yawn, or accumulate total time spent yawning, adjust here.
                 # For displaying duration of the last yawn, this is fine. If it becomes "No Yawn", this value persists.
            # If not currently yawning and wasn't in progress, yawn_duration might need to be 0 or reflect last known.
            # To show 0 when not yawning, and >0 when a yawn just finished/is in progress:
            if not self.yawn_in_progress and self.yawn_state == "No Yawn": # Check previous state as well
                 self.yawn_duration = 0 # Explicitly reset if no yawn is active and previous was also no yawn

        self.yawn_state = current_yawn_detection # Update overall yawn state for the next frame's logic
        t_state_logic_end = time.time()
        print(f"[PERF_PROFILE] process_frame: State update logic took: {(t_state_logic_end - t_state_logic_start)*1000:.2f} ms")

        current_status_dict = self.get_current_status()
        
        t_pf_exit = time.time()
        print(f"[PERF_PROFILE] process_frame: EXIT, Total time: {(t_pf_exit - t_pf_enter)*1000:.2f} ms")
        return draw_elements, current_status_dict

    def get_current_status(self):
        alert_status = self._determine_alert_status()
        return {
            "blinks": self.blinks,
            "microsleeps_duration": round(self.microsleeps_duration, 2),
            "yawns": self.yawns,
            "yawn_duration": round(self.yawn_duration, 2),
            "left_eye_state": self.left_eye_state,
            "right_eye_state": self.right_eye_state,
            "yawn_state": self.yawn_state,
            "alert_status": alert_status
        }

    def _determine_alert_status(self):
        # Customizable thresholds
        MICROSLEEP_THRESHOLD_SECONDS = 2.0 
        PROLONGED_YAWN_THRESHOLD_SECONDS = 3.0
        # YAWN_FREQUENCY_THRESHOLD_COUNT = 3 # Example: 3 yawns in last N seconds (needs time window logic)

        if self.microsleeps_duration >= MICROSLEEP_THRESHOLD_SECONDS:
            return "Prolonged Microsleep Detected!"
        if self.yawn_duration >= PROLONGED_YAWN_THRESHOLD_SECONDS: # Based on current continuous yawn
            return "Prolonged Yawn Detected!"
        # Add more complex alert logic if needed (e.g., frequent yawns)
        
        # Default states
        if self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye":
            return "Eyes Closed"
        if self.yawn_state == "Yawn":
            return "Yawning"
        
        return "Awake"


    def draw_overlays(self, frame_to_draw_on, status_info, draw_elements):
        # Draw face landmarks
        if draw_elements.get("face_landmarks_coords"):
            for (x, y) in draw_elements["face_landmarks_coords"]:
                cv2.circle(frame_to_draw_on, (x, y), 1, (0, 255, 0), -1) # Green dots for landmarks

        # Draw ROI bounding boxes
        roi_color = (255, 165, 0) # Orange for ROIs
        for roi_key in ["mouth_roi_bbox", "right_eye_roi_bbox", "left_eye_roi_bbox"]:
            bbox = draw_elements.get(roi_key)
            if bbox:
                cv2.rectangle(frame_to_draw_on, (bbox[0], bbox[1]), (bbox[2], bbox[3]), roi_color, 2)
        
        # Display status text
        y_offset = 30
        cv2.putText(frame_to_draw_on, f"Alert: {status_info['alert_status']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
        cv2.putText(frame_to_draw_on, f"L-Eye: {status_info['left_eye_state']}, R-Eye: {status_info['right_eye_state']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame_to_draw_on, f"Blinks: {status_info['blinks']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame_to_draw_on, f"Microsleep (s): {status_info['microsleeps_duration']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame_to_draw_on, f"Yawn: {status_info['yawn_state']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame_to_draw_on, f"Yawns: {status_info['yawns']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame_to_draw_on, f"Yawn Duration (s): {status_info['yawn_duration']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame_to_draw_on

    def release(self):
        if self.face_mesh:
            self.face_mesh.close()
        # No specific release for TFLite interpreter needed like model.close() unless using delegates that need cleanup.
        print("AIDrowsinessProcessor resources released.")

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