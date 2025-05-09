import sys
import os
import datetime
import cv2
import time
import threading
import queue
import psutil
import argparse
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from src.ai_logic import AIDrowsinessProcessor

# --- Tee class for duplicating stream to file and console ---
class Tee(object):
    def __init__(self, filepath, original_stream, mode='a', encoding='utf-8'):
        self.file = None
        self.original_stream = original_stream
        self.filepath = filepath
        try:
            self.file = open(filepath, mode, encoding=encoding)
        except Exception as e:
            # Use sys.__stderr__ for bootstrap errors if original_stream is not yet reliable or file fails
            sys.__stderr__.write(f"Error opening log file {self.filepath}: {e}\\n")

    def __del__(self):
        if self.file and not self.file.closed:
            try:
                self.file.close()
            except Exception as e:
                sys.__stderr__.write(f"Error closing log file in Tee.__del__: {e}\\n")
        self.file = None

    def write(self, data):
        try:
            if self.file and not self.file.closed:
                self.file.write(data)
            else: # Attempt to reopen if closed unexpectedly
                try:
                    self.file = open(self.filepath, 'a', encoding='utf-8')
                    if self.file:
                         self.file.write(data)
                except Exception as reopen_e:
                    sys.__stderr__.write(f"Error reopening log file in Tee.write: {reopen_e}\\n")

            if self.original_stream:
                self.original_stream.write(data)
            self.flush() # Call flush here to ensure immediate write
        except Exception as e:
            sys.__stderr__.write(f"Error in Tee.write: {e}\\nData preview: {str(data)[:100]}\\n")


    def flush(self):
        try:
            if self.file and not self.file.closed:
                self.file.flush()
            if self.original_stream:
                self.original_stream.flush()
        except Exception as e:
             sys.__stderr__.write(f"Error in Tee.flush: {e}\\n")


# --- Logging Setup ---
def setup_file_logging():
    logs_dir = "logs"
    try:
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(logs_dir, f"app_{timestamp}.log")

        # This print will go to the original stdout before redirection by Tee
        sys.__stdout__.write(f"Attempting to log stdout and stderr to: {log_file_path}\\n")

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Create Tee objects. Both will write to the same log file.
        # sys.stdout will write to log_file_path and original_stdout.
        # sys.stderr will write to log_file_path and original_stderr.
        sys.stdout = Tee(log_file_path, original_stdout)
        sys.stderr = Tee(log_file_path, original_stderr)

        # This print will be the first to be processed by the Tee objects for both file and console.
        print("File logging setup complete. Subsequent output will be logged and also printed to console.")
    except Exception as e:
        # Use original stderr for critical errors during logging setup
        sys.__stderr__.write(f"CRITICAL ERROR setting up file logging: {e}\\n")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')

# --- Model Configurations ---
MODEL_CONFIGURATIONS = {
    "int8_320": {
        "eye_model": "model/tflite/eye_detect_320_int8.tflite",
        "yawn_model": "model/tflite/yawn_detect_320_int8.tflite",
        "description": "INT8 quantized models with 320x320 input. Fastest, good for RPi."
    },
    "fp16_320": {
        "eye_model": "model/tflite/eye_detect_320_fp16.tflite",
        "yawn_model": "model/tflite/yawn_detect_320_fp16.tflite",
        "description": "FP16 models with 320x320 input. Balanced speed and precision."
    },
    "int8_640": {
        "eye_model": "model/tflite/eye_detect_640_int8.tflite",
        "yawn_model": "model/tflite/yawn_detect_640_int8.tflite",
        "description": "INT8 quantized models with 640x640 input. Slower, potentially more accurate."
    },
    "fp16_640": {
        "eye_model": "model/tflite/eye_detect_640_fp16.tflite",
        "yawn_model": "model/tflite/yawn_detect_640_fp16.tflite",
        "description": "FP16 models with 640x640 input. Slowest, potentially highest precision."
    },
    # Add more configurations as needed, e.g., for different model versions or custom paths
}
# --- >>> SELECT YOUR DEFAULT MODEL CONFIGURATION BY CHANGING THE KEY BELOW <<< ---
DEFAULT_MODEL_CONFIG_KEY = "fp16_320" # Change this key to select a default config

# --- Global Variables and Locks ---
camera_object = None
ai_processor = None
detection_active = False
stop_event = threading.Event()

latest_raw_frame = None
latest_ai_status_dict = None
latest_ai_draw_elements = None
frame_queue_for_ai = queue.Queue(maxsize=2)

raw_frame_lock = threading.Lock()
ai_status_lock = threading.Lock()
ai_draw_elements_lock = threading.Lock()
resource_lock = threading.Lock()

camera_thread = None
ai_processing_thread = None

performance_stats = {
    "camera_fps": 0,
    "ai_frame_process_time_ms": 0,
    "ram_usage_mb": 0,
    "cpu_usage_percent_core": [],
}
stats_lock = threading.Lock()

# --- Application Arguments (global to be accessible by initialize_resources) ---
APP_ARGS = None

# --- Initialization and Teardown ---
def initialize_resources(cam_idx, eye_model_path, yawn_model_path):
    global camera_object, ai_processor, detection_active, stop_event
    global latest_raw_frame, latest_ai_status_dict, latest_ai_draw_elements

    with resource_lock:
        if camera_object is None:
            print(f"Opening camera {cam_idx}...")
            camera_object = cv2.VideoCapture(cam_idx)
            if not camera_object.isOpened():
                print(f"Error: Cannot open camera {cam_idx}")
                camera_object = None
                socketio.emit('error_event', {'message': f'Failed to open camera {cam_idx}. Please ensure it is connected and not in use.'})
                return False
            print("Camera opened successfully.")
            latest_raw_frame = None

        if ai_processor is None:
            print("Initializing AIDrowsinessProcessor...")
            try:
                ai_processor = AIDrowsinessProcessor(
                    tflite_eye_model_path=eye_model_path,
                    tflite_yawn_model_path=yawn_model_path,
                    camera_index=cam_idx
                )
                print(f"AIDrowsinessProcessor initialized with Eye: {eye_model_path}, Yawn: {yawn_model_path}")
                latest_ai_status_dict = ai_processor.get_current_status()
                latest_ai_draw_elements = {}
            except Exception as e:
                print(f"Error initializing AIDrowsinessProcessor: {e}")
                socketio.emit('error_event', {'message': f'Error initializing AI models: {str(e)}. Check model paths and integrity.'})
                ai_processor = None
                return False
        
        stop_event.clear()
        detection_active = True
    return True

def release_resources():
    global camera_object, ai_processor, detection_active
    global camera_thread, ai_processing_thread

    print("Releasing resources...")
    detection_active = False
    stop_event.set()

    if camera_thread is not None and camera_thread.is_alive():
        print("Joining camera thread...")
        camera_thread.join(timeout=2)
        if camera_thread.is_alive():
            print("Camera thread did not join in time.")

    if ai_processing_thread is not None and ai_processing_thread.is_alive():
        print("Joining AI processing thread...")
        ai_processing_thread.join(timeout=2)
        if ai_processing_thread.is_alive():
            print("AI processing thread did not join in time.")
    
    while not frame_queue_for_ai.empty():
        try:
            frame_queue_for_ai.get_nowait()
        except queue.Empty:
            break

    with resource_lock:
        if camera_object is not None:
            print("Releasing camera...")
            camera_object.release()
            camera_object = None
            print("Camera released.")
        
        if ai_processor is not None:
            print("Releasing AI Processor...")
            if hasattr(ai_processor, 'release') and callable(getattr(ai_processor, 'release')):
                 ai_processor.release()
            ai_processor = None
            print("AI Processor released.")
    print("Resources released.")

# --- Video Streaming and AI Processing Threads ---
def camera_thread_function():
    global latest_raw_frame, frame_queue_for_ai, raw_frame_lock, stats_lock, performance_stats
    global detection_active, camera_object, APP_ARGS, stop_event
    
    print(f"[CAMERA_THREAD] Starting. Initial detection_active: {detection_active}, stop_event: {stop_event.is_set()}")
    
    frame_count = 0
    failed_grab_count = 0
    last_fps_calc_time = time.time()
    loop_count = 0

    if not detection_active or stop_event.is_set():
        print("[CAMERA_THREAD] Exiting immediately due to initial state (detection_active or stop_event).")
        print("[CAMERA_THREAD] Camera thread finishing early.")
        return

    while detection_active and not stop_event.is_set():
        loop_count += 1
        try:
            if camera_object is None:
                print("[CAMERA_THREAD] CRITICAL: camera_object is None. Stopping.")
                stop_event.set()
                detection_active = False
                break
            if not camera_object.isOpened():
                print("[CAMERA_THREAD] CRITICAL: camera_object is not opened. Stopping.")
                stop_event.set()
                detection_active = False 
                break
            
            ret, frame = camera_object.read()
            
            if not ret:
                failed_grab_count += 1
                print(f"[CAMERA_THREAD] WARNING: Failed to grab frame (ret=False). Attempt: {failed_grab_count}")
                if failed_grab_count > APP_ARGS.camera_max_grab_attempts: 
                    print(f"[CAMERA_THREAD] CRITICAL: Failed to grab frame after {failed_grab_count} attempts. Stopping detection.")
                    socketio.emit('error_event', {'message': 'Failed to grab frame after multiple attempts.'})
                    stop_event.set()
                    detection_active = False 
                    break
                time.sleep(0.1) 
                continue 
            
            if frame is None:
                failed_grab_count +=1
                print(f"[CAMERA_THREAD] WARNING: Grabbed frame is None. Attempt: {failed_grab_count}")
                if failed_grab_count > APP_ARGS.camera_max_grab_attempts:
                    print(f"[CAMERA_THREAD] CRITICAL: Grabbed frame is None after {failed_grab_count} attempts. Stopping detection.")
                    socketio.emit('error_event', {'message': 'Grabbed None frame after multiple attempts.'})
                    stop_event.set()
                    detection_active = False 
                    break
                time.sleep(0.1)
                continue

            failed_grab_count = 0 

            with raw_frame_lock:
                latest_raw_frame = frame.copy()

            try:
                frame_queue_for_ai.put(frame.copy(), block=True, timeout=0.1)
            except queue.Full:
                pass 

            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_calc_time >= 5.0: 
                fps = frame_count / (current_time - last_fps_calc_time) if (current_time - last_fps_calc_time) > 0 else 0
                with stats_lock:
                    performance_stats["camera_fps"] = round(fps, 2)
                print(f"[CAMERA_THREAD] Captured {frame_count} frames in last ~{(current_time - last_fps_calc_time):.2f}s. Approx FPS: {fps:.2f}")
                frame_count = 0
                last_fps_calc_time = current_time
            
            if APP_ARGS.camera_fps > 0:
                time.sleep(1.0 / APP_ARGS.camera_fps)
            else:
                time.sleep(0.01) 

        except Exception as e:
            print(f"[CAMERA_THREAD] CRITICAL ERROR in loop: {e}")
            import traceback
            print(traceback.format_exc())
            stop_event.set()
            detection_active = False
            socketio.emit('error_event', {'message': f'Critical error in camera thread: {str(e)}'})
            break
        
    print(f"[CAMERA_THREAD] Camera thread finishing. Final detection_active: {detection_active}, stop_event: {stop_event.is_set()}, loop_count: {loop_count}")

def ai_processing_thread_function():
    global latest_ai_status_dict, latest_ai_draw_elements, ai_status_lock, ai_draw_elements_lock, frame_queue_for_ai
    global performance_stats, stats_lock, detection_active, ai_processor, stop_event
    print("[AI_THREAD] AI Processing and status emission started.")

    while detection_active and not stop_event.is_set():
        try:
            raw_frame_for_ai = frame_queue_for_ai.get(block=True, timeout=1.0)
        except queue.Empty:
            if not detection_active or stop_event.is_set(): break
            continue

        if raw_frame_for_ai is None or ai_processor is None:
            if not detection_active or stop_event.is_set(): break
            continue

        start_time = time.time()
        draw_elements, status_dict = ai_processor.process_frame(raw_frame_for_ai)
        end_time = time.time()
        process_time_ms = (end_time - start_time) * 1000

        with ai_status_lock:
            latest_ai_status_dict = status_dict
        with ai_draw_elements_lock:
            latest_ai_draw_elements = draw_elements
        
        ram_usage = psutil.virtual_memory().used / (1024 * 1024)
        cpu_cores_usage = psutil.cpu_percent(percpu=True)

        with stats_lock:
            performance_stats["ai_frame_process_time_ms"] = round(process_time_ms, 1)
            performance_stats["ram_usage_mb"] = round(ram_usage, 1)
            performance_stats["cpu_usage_percent_core"] = [f"{usage}%" for usage in cpu_cores_usage]
            full_status_update = {**status_dict, "performance": performance_stats}
        
        socketio.emit('status_update', full_status_update)

    print("AI Processing and status emission finishing.")

def video_stream_generator():
    global latest_raw_frame, raw_frame_lock, latest_ai_status_dict, ai_status_lock, latest_ai_draw_elements, ai_draw_elements_lock
    global detection_active, ai_processor, stop_event, APP_ARGS
    print(f"[VIDEO_STREAM] Function CALLED. detection_active: {detection_active}, stop_event: {stop_event.is_set()}, APP_ARGS.display_fps: {getattr(APP_ARGS, 'display_fps', 'N/A')}", flush=True)
    
    TARGET_VIDEO_FPS = APP_ARGS.display_fps if hasattr(APP_ARGS, 'display_fps') and APP_ARGS.display_fps is not None else 20 # Robust access
    delay_per_frame = 1.0 / TARGET_VIDEO_FPS if TARGET_VIDEO_FPS > 0 else 0.05 
    frame_counter = 0
    print(f"[VIDEO_STREAM] Initialized with TARGET_VIDEO_FPS: {TARGET_VIDEO_FPS}, delay_per_frame: {delay_per_frame}", flush=True)

    while True:
        print(f"[VIDEO_STREAM] Top of while True loop. Frame: {frame_counter}", flush=True) 
        try:
            print(f"[VIDEO_STREAM] Inside try block. Frame: {frame_counter}", flush=True) 
            if stop_event.is_set() and not detection_active:
                print("[VIDEO_STREAM] stop_event is set and detection not active. Exiting loop.", flush=True)
                break

            start_frame_gen_time = time.time()
            
            current_frame_to_display = None
            frame_source = "None"
            with raw_frame_lock:
                if latest_raw_frame is not None:
                    current_frame_to_display = latest_raw_frame.copy()
                    frame_source = "latest_raw_frame"
            
            if current_frame_to_display is None:
                # ALWAYS yield a frame, even if it's a placeholder, to keep the stream alive.
                placeholder_text = "Detection Stopped or Starting..."
                if detection_active:
                    placeholder_text = "Waiting for AI frame..."
                
                placeholder = np.zeros((APP_ARGS.display_height if hasattr(APP_ARGS, 'display_height') else 480, 
                                        APP_ARGS.display_width if hasattr(APP_ARGS, 'display_width') else 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, placeholder_text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                try:
                    ret_enc, encoded_image = cv2.imencode('.jpg', placeholder)
                    if not ret_enc:
                        print("[VIDEO_STREAM] Error encoding placeholder image.", flush=True)
                        time.sleep(delay_per_frame) # Avoid fast loop on error
                        continue
                    print(f"[VIDEO_STREAM] PRE-YIELD (placeholder) frame. Text: {placeholder_text}", flush=True)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + encoded_image.tobytes() + b'\r\n')
                    print(f"[VIDEO_STREAM] POST-YIELD (placeholder) frame.", flush=True)
                except Exception as e:
                    print(f"[VIDEO_STREAM] Error yielding placeholder: {e}", flush=True)
                time.sleep(delay_per_frame) # Wait after sending placeholder
                continue # Go to next iteration to try to get a real frame
            
            # If we have a real frame, proceed to process and yield it.
            frame_counter += 1

            current_status_for_draw = None
            current_draw_elements = None
            if detection_active:
                with ai_status_lock:
                    if latest_ai_status_dict is not None:
                        current_status_for_draw = latest_ai_status_dict.copy()
                with ai_draw_elements_lock:
                    if latest_ai_draw_elements is not None:
                        current_draw_elements = latest_ai_draw_elements.copy()

            final_frame_to_encode = current_frame_to_display # Default to current frame

            if detection_active and ai_processor is not None and current_status_for_draw and current_draw_elements:
                try:
                    frame_with_overlays = ai_processor.draw_overlays(
                        current_frame_to_display, # Pass the original copy
                        current_status_for_draw, 
                        current_draw_elements
                    )
                    final_frame_to_encode = frame_with_overlays # Update frame to be encoded
                except Exception as e:
                    print(f"[VIDEO_STREAM] Error during drawing overlays: {e}", flush=True)
                    # Fallback to sending the frame without overlays in case of drawing error
            elif not detection_active:
                 cv2.putText(final_frame_to_encode, "Detection Stopped", (10, final_frame_to_encode.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
            
            try:
                ret_enc, encoded_image = cv2.imencode('.jpg', final_frame_to_encode)
                if not ret_enc:
                    print(f"[VIDEO_STREAM] Error encoding frame {frame_counter} for streaming.", flush=True)
                    # Don't yield if encoding failed
                else:
                    print(f"[VIDEO_STREAM] PRE-YIELD frame {frame_counter}. Size: {len(encoded_image.tobytes())}", flush=True) # Log Before Yield
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + encoded_image.tobytes() + b'\r\n')
                    print(f"[VIDEO_STREAM] POST-YIELD frame {frame_counter}.", flush=True) # Log After Yield
            except Exception as e:
                print(f"[VIDEO_STREAM] Error during final imencode or yield of frame {frame_counter}: {e}", flush=True)

            elapsed_time = time.time() - start_frame_gen_time
            sleep_time = delay_per_frame - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        except Exception as e:
            print(f"[VIDEO_STREAM] CRITICAL ERROR in main loop: {e}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
            # Attempt to yield a visual error frame to the client
            try:
                error_display_frame = np.zeros((APP_ARGS.display_height if hasattr(APP_ARGS, 'display_height') else 480, 
                                                APP_ARGS.display_width if hasattr(APP_ARGS, 'display_width') else 640, 3), dtype=np.uint8)
                cv2.putText(error_display_frame, "Stream Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(error_display_frame, str(e)[:60], (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
                (flag, encoded_error_Image) = cv2.imencode(".jpg", error_display_frame)
                if flag:
                    print("[VIDEO_STREAM] Yielding visual error frame.", flush=True)
                    yield (b'--frame\\r\\n' b'Content-Type: image/jpeg\\r\\n\\r\\n' +
                           bytearray(encoded_error_Image) + b'\\r\\n')
            except Exception as e2:
                print(f"[VIDEO_STREAM] Failed to yield visual error frame: {e2}", flush=True)
            time.sleep(1) # Avoid rapid looping on persistent error

    print("[VIDEO_STREAM] Video stream generator finishing.", flush=True)

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with ai_status_lock:
        if latest_ai_status_dict:
            return jsonify(latest_ai_status_dict)
    return jsonify({"status": "AI data not available yet"})

@app.route('/start_detection', methods=['POST'])
def start_detection_route():
    global camera_thread, ai_processing_thread, detection_active, APP_ARGS
    print("Attempting to start detection...")

    if detection_active and camera_thread is not None and camera_thread.is_alive():
        print("Detection is already active.")
        return jsonify({"status": "Detection already active"}), 200

    if (camera_thread is not None and not camera_thread.is_alive()) or \
       (ai_processing_thread is not None and not ai_processing_thread.is_alive()):
        print("Threads were dead but detection_active might have been true. Releasing first.")
        release_resources()

    if not initialize_resources(APP_ARGS.camera_index, APP_ARGS.eye_model, APP_ARGS.yawn_model):
        print("Failed to initialize resources for start_detection.")
        return jsonify({"status": "Failed to initialize resources. Check camera and model paths."}), 500

    print("Starting threads for detection...")
    camera_thread = threading.Thread(target=camera_thread_function, daemon=True)
    ai_processing_thread = threading.Thread(target=ai_processing_thread_function, daemon=True)
    
    camera_thread.start()
    ai_processing_thread.start()
    detection_active = True
    socketio.emit('system_message', {'message': 'Detection started successfully.'})    
    print("Detection started successfully.")
    return jsonify({"status": "Detection started"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection_route():
    global detection_active
    print("Attempting to stop detection...")
    if not detection_active and (camera_thread is None or not camera_thread.is_alive()):
        print("Detection is not active or threads are already stopped.")
        if camera_object is not None or ai_processor is not None:
            print("Performing cleanup release...")
            release_resources()
        return jsonify({"status": "Detection already stopped or was not running"}), 200

    release_resources()
    socketio.emit('system_message', {'message': 'Detection stopped successfully.'})    
    print("Detection stopped successfully.")
    return jsonify({"status": "Detection stopped"})

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    initial_status = {}
    with ai_status_lock:
        if latest_ai_status_dict:
            initial_status.update(latest_ai_status_dict)
    with stats_lock:
        initial_status["performance"] = performance_stats
    
    emit('status_update', initial_status)
    emit('system_message', {'message': 'Connected to Drowsiness Detection server.'})    

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# --- Main Execution ---
if __name__ == '__main__':
    setup_file_logging() # Added: Call logging setup first
    
    # Define default paths from the selected configuration for argparse
    selected_config = MODEL_CONFIGURATIONS.get(DEFAULT_MODEL_CONFIG_KEY)
    if not selected_config:
        print(f"Error: DEFAULT_MODEL_CONFIG_KEY '{DEFAULT_MODEL_CONFIG_KEY}' not found in MODEL_CONFIGURATIONS. Using fallback defaults.")
        default_eye_model_for_argparse = 'model/tflite/eye_detect_320_int8.tflite'
        default_yawn_model_for_argparse = 'model/tflite/yawn_detect_320_int8.tflite'
    else:
        default_eye_model_for_argparse = selected_config["eye_model"]
        default_yawn_model_for_argparse = selected_config["yawn_model"]

    parser = argparse.ArgumentParser(description="Drowsiness Detection System with Flask and SocketIO")
    parser.add_argument('--eye_model', type=str, 
                        default=default_eye_model_for_argparse, 
                        help='Path to the TFLite eye detection model.')
    parser.add_argument('--yawn_model', type=str, 
                        default=default_yawn_model_for_argparse, 
                        help='Path to the TFLite yawn detection model.')
    parser.add_argument('--camera_index', type=int, default=0, 
                        help='Index of the camera to use.')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the Flask app on.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the Flask app on.')
    parser.add_argument('--config_name', type=str, default=None,
                        choices=list(MODEL_CONFIGURATIONS.keys()),
                        help='Name of the model configuration to use from MODEL_CONFIGURATIONS.')
    parser.add_argument('--camera_fps', type=int, default=30, help='Target FPS for camera capture.')
    parser.add_argument('--display_fps', type=int, default=20, help='Target FPS for video stream display.')
    parser.add_argument('--camera_max_grab_attempts', type=int, default=50, help='Max attempts to grab frame before stopping.')
    
    APP_ARGS = parser.parse_args()

    # If --config_name is provided via command line, it overrides the code-defined DEFAULT_MODEL_CONFIG_KEY
    # and subsequently the model paths, unless --eye_model or --yawn_model are also explicitly set.
    
    final_config_key_to_use = DEFAULT_MODEL_CONFIG_KEY
    if APP_ARGS.config_name:
        final_config_key_to_use = APP_ARGS.config_name
        print(f"Using command-line specified model configuration: '{final_config_key_to_use}'")
    
    config_to_apply = MODEL_CONFIGURATIONS.get(final_config_key_to_use)
    if config_to_apply:
        if APP_ARGS.eye_model == default_eye_model_for_argparse:
            APP_ARGS.eye_model = config_to_apply['eye_model']
        if APP_ARGS.yawn_model == default_yawn_model_for_argparse:
            APP_ARGS.yawn_model = config_to_apply['yawn_model']
    else:
        print(f"Warning: Configuration key '{final_config_key_to_use}' not found. Using direct argparse model paths or fallbacks.")

    print(f"Starting Drowsiness Detection System with final settings:")
    print(f"  Selected Configuration Key: {final_config_key_to_use} (Description: {MODEL_CONFIGURATIONS.get(final_config_key_to_use, {}).get('description', 'N/A')})")
    print(f"  Eye Model: {APP_ARGS.eye_model}")
    print(f"  Yawn Model: {APP_ARGS.yawn_model}")
    print(f"  Camera Index: {APP_ARGS.camera_index}")
    print(f"  Host: {APP_ARGS.host}")
    print(f"  Port: {APP_ARGS.port}")

    try:
        socketio.run(app, host=APP_ARGS.host, port=APP_ARGS.port, debug=False, use_reloader=False)
    finally:
        print("Application shutting down. Releasing resources...")
        release_resources()
        print("Cleanup complete. Exiting.") 