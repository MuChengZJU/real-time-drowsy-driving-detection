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
        print(f"[AI_LOGIC DEBUG] _parse_tflite_output: Raw TFLite output_data shape: {output_data.shape}")
        # print(f"[AI_LOGIC DEBUG] TFLite output data sample [0, :5, :]: {output_data[0,:5,:]}") # Might be too verbose

        best_class_id = -1
        max_confidence = -1.0

        # If shape is (1, num_coords_plus_classes, num_proposals), then transpose
        if output_data.shape[1] < output_data.shape[2] and (output_data.shape[1] == (4 + 2) or output_data.shape[1] == (4+1+2)): # Assuming 2 classes for eye/yawn
             print(f"[AI_LOGIC DEBUG] Transposing output from {output_data.shape} to (0, 2, 1)")
             output_data = np.transpose(output_data, (0, 2, 1))
        
        print(f"[AI_LOGIC DEBUG] Final TFLite output_data shape for parsing: {output_data.shape}")

        detections = output_data[0] 
        output_dim_size = detections.shape[1] # Dimension of a single detection, e.g., 6 for (x,y,w,h,c0,c1)
        print(f"[AI_LOGIC DEBUG] Number of detections: {detections.shape[0]}, Output_dim_size (per detection): {output_dim_size}")

        # Filter out detections with very low scores early if possible (though NMS in model should handle this)
        # For example, if obj_conf is present and very low.

        for i in range(detections.shape[0]):
            detection = detections[i] 
            current_confidence = 0
            current_class_id = -1

            # print(f"[AI_LOGIC DEBUG] Parsing detection {i}: {detection}") # Can be very verbose

            if output_dim_size == 6: # Assuming 4 coords + 2 class scores (e.g., from Ultralytics export with NMS)
                # detection = [x,y,w,h, class0_score, class1_score]
                class_scores = detection[4:] 
                current_confidence = np.max(class_scores)
                current_class_id = np.argmax(class_scores)
                # print(f"[AI_LOGIC DEBUG] Branch 1 (dim=6): Raw scores: {class_scores}, Conf: {current_confidence}, Class: {current_class_id}")
            elif output_dim_size == 7: # Assuming 4 coords + 1 obj_conf + 2 class scores
                # detection = [x,y,w,h, obj_conf, class0_score, class1_score]
                object_confidence = detection[4]
                class_scores = detection[5:]
                # current_confidence = object_confidence # Often, obj_conf is the primary confidence.
                                                    # Or, sometimes obj_conf * max(class_scores)
                current_confidence = object_confidence * np.max(class_scores) if np.max(class_scores) > 0 else object_confidence 
                current_class_id = np.argmax(class_scores)
                # print(f"[AI_LOGIC DEBUG] Branch 2 (dim=7): ObjConf: {object_confidence}, Raw scores: {class_scores}, FinalConf: {current_confidence}, Class: {current_class_id}")
            else:
                print(f"[AI_LOGIC DEBUG] Branch 3 (dim={output_dim_size}): Unexpected output dimension. Skipping parsing for this detection.")
                # Fallback or needs more specific parsing based on your model output.
                # This part is CRITICAL and depends heavily on the exact TFLite model's output signature.
                # You might need to inspect the model output with a tool like Netron or print shapes/values.
                # For now, let's assume it's [x,y,w,h, conf_class0, conf_class1]
                # if output_dim_size < 2: continue # Not enough info
                # class_scores = detection[-(output_dim_size - 4):] # Heuristic: last elements are class scores
                # current_confidence = np.max(class_scores)
                # current_class_id = np.argmax(class_scores)
                continue # Skip this detection if we don't know how to parse it

            if current_confidence > conf_threshold and current_confidence > max_confidence:
                max_confidence = current_confidence
                best_class_id = current_class_id
        
        print(f"[AI_LOGIC DEBUG] _parse_tflite_output result: BestClassID: {best_class_id}, MaxConfidence: {max_confidence}")
        return best_class_id, max_confidence


    def _predict_eye_state_tflite(self, eye_roi_frame):
        if eye_roi_frame is None or eye_roi_frame.size == 0:
            return "Unknown"

        input_data = self._preprocess_image_for_tflite(eye_roi_frame, self.eye_input_height, self.eye_input_width, self.eye_input_details)
        if input_data is None:
            return "Unknown"

        self.eye_interpreter.set_tensor(self.eye_input_details[0]['index'], input_data)
        self.eye_interpreter.invoke()
        output_data = self.eye_interpreter.get_tensor(self.eye_output_details[0]['index'])
        
        class_id, confidence = self._parse_tflite_output(output_data, eye_roi_frame.shape, conf_threshold=0.3)

        # print(f"Eye detection - Class ID: {class_id}, Confidence: {confidence}")

        if class_id == 1: # Assuming class 1 is "Close Eye"
            return "Close Eye"
        elif class_id == 0: # Assuming class 0 is "Open Eye"
            return "Open Eye"
        return "Unknown" # Or previous state if confidence is low

    def _predict_yawn_tflite(self, mouth_roi_frame):
        if mouth_roi_frame is None or mouth_roi_frame.size == 0:
            return self.yawn_state # Keep previous state

        input_data = self._preprocess_image_for_tflite(mouth_roi_frame, self.yawn_input_height, self.yawn_input_width, self.yawn_input_details)
        if input_data is None:
            return self.yawn_state

        self.yawn_interpreter.set_tensor(self.yawn_input_details[0]['index'], input_data)
        self.yawn_interpreter.invoke()
        output_data = self.yawn_interpreter.get_tensor(self.yawn_output_details[0]['index'])

        class_id, confidence = self._parse_tflite_output(output_data, mouth_roi_frame.shape, conf_threshold=0.5)
        
        # print(f"Yawn detection - Class ID: {class_id}, Confidence: {confidence}")

        if class_id == 0:  # Assuming class 0 is "Yawn"
            return "Yawn"
        elif class_id == 1:  # Assuming class 1 is "No Yawn"
            return "No Yawn"
        return self.yawn_state # Keep previous state if low confidence or unknown

    def process_frame(self, frame):
        if frame is None:
             return {}, self.get_current_status()
             
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        draw_elements = {
            "face_landmarks_coords": [],
            "mouth_roi_bbox": None,
            "right_eye_roi_bbox": None,
            "left_eye_roi_bbox": None
        }

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results_facemesh = self.face_mesh.process(image_rgb)
        
        mouth_roi_frame, right_eye_roi_frame, left_eye_roi_frame = None, None, None
        # rois_extracted = False # Not strictly needed with direct ROI processing

        if results_facemesh.multi_face_landmarks:
            for face_landmarks in results_facemesh.multi_face_landmarks:
                ih, iw, _ = frame.shape
                
                # Simplified ROI extraction using specific landmark indices (more robust than self.points_ids approach)
                # These are standard MediaPipe Face Mesh landmark indices.
                LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144] # Example subset for bounding box
                RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380] # Example subset
                MOUTH_INDICES = [61, 291, 0, 17] # Example: left, right, top, bottom points of mouth outline
                
                def get_roi_bbox_from_landmarks(landmark_list, indices, img_width, img_height, padding=5):
                    xs = [landmark_list[i].x * img_width for i in indices]
                    ys = [landmark_list[i].y * img_height for i in indices]
                    if not xs or not ys: return None, None
                    
                    x_min, x_max = int(min(xs)), int(max(xs))
                    y_min, y_max = int(min(ys)), int(max(ys))

                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(img_width, x_max + padding)
                    y_max = min(img_height, y_max + padding)
                    
                    if x_max > x_min and y_max > y_min:
                        return (x_min, y_min, x_max, y_max), frame[y_min:y_max, x_min:x_max]
                    return None, None

                landmarks_list = face_landmarks.landmark

                left_eye_bbox, left_eye_roi_frame = get_roi_bbox_from_landmarks(landmarks_list, LEFT_EYE_INDICES, iw, ih)
                right_eye_bbox, right_eye_roi_frame = get_roi_bbox_from_landmarks(landmarks_list, RIGHT_EYE_INDICES, iw, ih)
                mouth_bbox, mouth_roi_frame = get_roi_bbox_from_landmarks(landmarks_list, MOUTH_INDICES, iw, ih, padding=10) # More padding for mouth

                if left_eye_bbox: draw_elements["left_eye_roi_bbox"] = left_eye_bbox
                if right_eye_bbox: draw_elements["right_eye_roi_bbox"] = right_eye_bbox
                if mouth_bbox: draw_elements["mouth_roi_bbox"] = mouth_bbox
                
                all_face_mesh_points_for_draw = []
                for landmark in landmarks_list:
                    x, y = int(landmark.x * iw), int(landmark.y * ih)
                    all_face_mesh_points_for_draw.append((x,y))
                draw_elements["face_landmarks_coords"] = all_face_mesh_points_for_draw
                
                # Prediction
                try:
                    if left_eye_roi_frame is not None and left_eye_roi_frame.size > 0:
                        self.left_eye_state = self._predict_eye_state_tflite(left_eye_roi_frame)
                    else:
                        self.left_eye_state = "Unknown" # Or keep previous

                    if right_eye_roi_frame is not None and right_eye_roi_frame.size > 0:
                        self.right_eye_state = self._predict_eye_state_tflite(right_eye_roi_frame)
                    else:
                        self.right_eye_state = "Unknown" # Or keep previous
                    
                    if mouth_roi_frame is not None and mouth_roi_frame.size > 0:
                        self.yawn_state = self._predict_yawn_tflite(mouth_roi_frame)
                    # else: keep previous yawn_state

                except Exception as e:
                    print(f"Error during TFLite prediction: {e}")
                    # Optionally reset states or log more verbosely
                    self.left_eye_state = "Error"
                    self.right_eye_state = "Error"
                    self.yawn_state = "Error"

                break # Process only the first detected face

        # Update drowsiness logic (remains largely the same)
        if self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye":
            if not self.left_eye_still_closed: # Checks if this is the start of a continuous closure
                self.left_eye_still_closed = True
                self.right_eye_still_closed = True # Assuming both start closing together for a blink
                self.blinks += 1 
            self.microsleeps_duration += delta_time
        else:
            if self.left_eye_still_closed or self.right_eye_still_closed: # If either was closed and now isn't
                 self.left_eye_still_closed = False
                 self.right_eye_still_closed = False
            self.microsleeps_duration = 0 # Reset microsleep if eyes are not consistently closed

        if self.yawn_state == "Yawn":
            if not self.yawn_in_progress:
                self.yawn_in_progress = True
                self.yawns += 1  
            self.yawn_duration += delta_time
        else: # "No Yawn" or other states
            if self.yawn_in_progress: # If it was yawning and now stopped
                self.yawn_in_progress = False
                # self.yawn_duration = 0 # Reset duration when yawn stops
            # If yawn_state is "No Yawn" and was not in progress, duration should remain 0 or be reset.
            # The key is to reset duration when a yawn event concludes.
            if not self.yawn_in_progress : # Simplified: if not yawning, duration does not accumulate / is reset
                 self.yawn_duration = 0


        current_status_dict = self.get_current_status()
        # Potentially update self.last_processed_frame_with_drawings if drawing is done here
        # For now, drawing is handled by app.py after getting status and draw_elements

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