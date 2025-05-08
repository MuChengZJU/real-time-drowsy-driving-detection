import cv2
import time
import threading
import queue
import psutil
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from src.ai_logic import AIDrowsinessProcessor # Updated import if class name changed

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')

# --- Global Variables and Locks ---
camera_object = None
ai_processor = None
detection_active = False
stop_event = threading.Event() # Used to signal threads to stop

# Frame and AI data sharing
latest_raw_frame = None
latest_ai_status_dict = None
latest_ai_draw_elements = None # New: for drawing data from AI
frame_queue_for_ai = queue.Queue(maxsize=2) # Keep queue size small to process recent frames

# Locks
raw_frame_lock = threading.Lock()
ai_status_lock = threading.Lock()
ai_draw_elements_lock = threading.Lock() # New lock for drawing data
resource_lock = threading.Lock() # For camera and AI processor initialization

# Threads
camera_thread = None
ai_processing_thread = None

# Performance Metrics
performance_stats = {
    "camera_fps": 0,
    "ai_frame_process_time_ms": 0,
    "ram_usage_mb": 0,
    "cpu_usage_percent_core": [],
}
stats_lock = threading.Lock()

# --- Initialization and Teardown ---
def initialize_resources():
    global camera_object, ai_processor, detection_active, stop_event
    global latest_raw_frame, latest_ai_status_dict, latest_ai_draw_elements

    with resource_lock:
        if camera_object is None:
            print("Opening camera 0...")
            camera_object = cv2.VideoCapture(0)
            if not camera_object.isOpened():
                print("Error: Cannot open camera 0")
                camera_object = None # Ensure it's None if failed
                return False
            print("Camera opened successfully.")
            latest_raw_frame = None # Reset on init

        if ai_processor is None:
            print("Initializing AIDrowsinessProcessor...")
            try:
                # Ensure model paths in AIDrowsinessProcessor are correct
                ai_processor = AIDrowsinessProcessor(
                    yolo_eye_model_path="runs/detecteye/train/weights/best.pt",
                    yolo_yawn_model_path="runs/detectyawn/train/weights/best.pt"
                )
                print("AIDrowsinessProcessor initialized successfully.")
                latest_ai_status_dict = ai_processor.get_current_status() # Get initial status
                latest_ai_draw_elements = {} # Initialize as empty dict
            except Exception as e:
                print(f"Error initializing AIDrowsinessProcessor: {e}")
                ai_processor = None # Ensure it's None if failed
                return False
        
        stop_event.clear() # Clear event for new start
        detection_active = True
    return True

def release_resources():
    global camera_object, ai_processor, detection_active
    global camera_thread, ai_processing_thread

    print("Releasing resources...")
    detection_active = False # Signal threads to stop
    stop_event.set() # Trigger stop for threads

    if camera_thread is not None and camera_thread.is_alive():
        print("Joining camera thread...")
        camera_thread.join(timeout=2)
        if camera_thread.is_alive():
            print("Camera thread did not join in time.")
    print("Camera thread finished.")

    if ai_processing_thread is not None and ai_processing_thread.is_alive():
        print("Joining AI processing thread...")
        ai_processing_thread.join(timeout=2)
        if ai_processing_thread.is_alive():
            print("AI processing thread did not join in time.")
    print("AI processing thread finished.")
    
    # Clear the queue in case AI thread exited while items were present
    while not frame_queue_for_ai.empty():
        try:
            frame_queue_for_ai.get_nowait()
        except queue.Empty:
            break
    print("Frame queue cleared.")

    with resource_lock:
        if camera_object is not None:
            print("Releasing camera...")
            camera_object.release()
            camera_object = None
            print("Camera released.")
        
        if ai_processor is not None:
            print("Releasing AI Processor...")
            ai_processor.release() # Call if defined in AIDrowsinessProcessor
            ai_processor = None
            print("AI Processor released.")
    print("Resources released.")

# --- Video Streaming and AI Processing Threads ---
def camera_thread_function():
    global latest_raw_frame, performance_stats, detection_active
    print("Camera thread started.")
    
    frame_count = 0
    last_fps_calc_time = time.time()

    while detection_active and not stop_event.is_set():
        if camera_object is None or not camera_object.isOpened():
            print("Camera not available in camera_thread. Stopping.")
            stop_event.set()
            break
        
        ret, frame = camera_object.read()
        if not ret:
            print("Failed to grab frame from camera. Stopping.")
            stop_event.set()
            break

        with raw_frame_lock:
            latest_raw_frame = frame.copy() # Store a copy

        try:
            # Non-blocking put, or with a small timeout, to avoid stalling if AI queue is full
            frame_queue_for_ai.put(frame, block=True, timeout=0.1) 
        except queue.Full:
            # Optional: log if queue is often full, indicating AI is too slow
            # print("AI processing queue is full. Frame dropped from camera thread.")
            pass # Frame is dropped if AI can't keep up

        frame_count += 1
        current_time = time.time()
        if current_time - last_fps_calc_time >= 1.0: # Calculate FPS every second
            fps = frame_count / (current_time - last_fps_calc_time)
            with stats_lock:
                performance_stats["camera_fps"] = round(fps, 2)
            # print(f"Camera FPS: {performance_stats['camera_fps']}") # Console log for FPS
            frame_count = 0
            last_fps_calc_time = current_time
        
        # Control camera capture rate if necessary, e.g. to match a target FPS
        # time.sleep(1/30) # Example: cap at 30 FPS, but usually VideoCapture handles this

    print("Camera thread finishing.")

def ai_processing_thread_function():
    global latest_ai_status_dict, latest_ai_draw_elements, performance_stats, detection_active
    print("AI Processing and status emission started.")

    while detection_active and not stop_event.is_set():
        try:
            raw_frame_for_ai = frame_queue_for_ai.get(block=True, timeout=1.0) # Wait for a frame
        except queue.Empty:
            continue # No frame, loop again

        if raw_frame_for_ai is None or ai_processor is None:
            continue

        start_time = time.time()
        
        # process_frame now returns draw_elements and status_dict
        draw_elements, status_dict = ai_processor.process_frame(raw_frame_for_ai)
        
        end_time = time.time()
        process_time_ms = (end_time - start_time) * 1000

        with ai_status_lock:
            latest_ai_status_dict = status_dict
        with ai_draw_elements_lock: # New: Update draw elements
            latest_ai_draw_elements = draw_elements
        
        ram_usage = psutil.virtual_memory().used / (1024 * 1024) # MB
        cpu_cores_usage = psutil.cpu_percent(percpu=True)

        with stats_lock:
            performance_stats["ai_frame_process_time_ms"] = round(process_time_ms, 1)
            performance_stats["ram_usage_mb"] = round(ram_usage, 1)
            performance_stats["cpu_usage_percent_core"] = [f"{usage}%" for usage in cpu_cores_usage]
            
            # Combine status_dict with performance_stats for WebSocket emission
            full_status_update = {**status_dict, "performance": performance_stats}
        
        socketio.emit('status_update', full_status_update)
        # print(f"AI Frame Process Time: {performance_stats['ai_frame_process_time_ms']} ms | RAM: {performance_stats['ram_usage_mb']} MB | CPU per Core: {performance_stats['cpu_usage_percent_core']}")

    print("AI Processing and status emission finishing.")

def video_stream_generator():
    global latest_raw_frame, latest_ai_status_dict, latest_ai_draw_elements, ai_processor
    print("Video stream generator started.")
    
    TARGET_VIDEO_FPS = 20 # Target FPS for the video stream
    delay_per_frame = 1.0 / TARGET_VIDEO_FPS

    while not stop_event.is_set(): # Stream as long as not explicitly stopped
        start_time = time.time()
        
        current_frame_to_display = None
        with raw_frame_lock:
            if latest_raw_frame is not None:
                current_frame_to_display = latest_raw_frame.copy()
                # print(f"VG: Copied latest_raw_frame. Shape: {current_frame_to_display.shape if current_frame_to_display is not None else 'None'}") # DEBUG
            # else:
                # print("VG: latest_raw_frame is None") # DEBUG

        if current_frame_to_display is None:
            # print("VG: current_frame_to_display is None before AI. Sleeping.") # DEBUG
            time.sleep(delay_per_frame / 2) # Wait a bit for a frame to appear
            continue

        # Get the latest AI data (status and draw elements)
        # current_status_for_draw = None
        # current_draw_elements = None
        # with ai_status_lock:
        #     if latest_ai_status_dict is not None:
        #          current_status_for_draw = latest_ai_status_dict.copy()
        # with ai_draw_elements_lock:
        #     if latest_ai_draw_elements is not None:
        #         current_draw_elements = latest_ai_draw_elements.copy()

        print("VG: Skipping AI overlay for debugging video stream.") # DEBUG: Temporarily skip AI overlay
        # if ai_processor is not None and current_status_for_draw and current_draw_elements:
        #     # Call draw_overlays with the current raw frame and latest AI data
        #     # ai_processor.draw_overlays expects: frame_to_draw_on, status_info, draw_elements
        #     try:
        #         frame_with_overlays = ai_processor.draw_overlays(
        #             current_frame_to_display, # The copy of latest_raw_frame
        #             current_status_for_draw, 
        #             current_draw_elements
        #         )
        #         current_frame_to_display = frame_with_overlays
        #     except Exception as e:
        #         print(f"Error in draw_overlays: {e}")
        #         # Fallback to current_frame_to_display without overlays if drawing fails

        # Encode and yield the frame
        # print(f"VG: Before imencode. Frame shape: {current_frame_to_display.shape if current_frame_to_display is not None else 'None'}") # DEBUG
        ret, buffer = cv2.imencode('.jpg', current_frame_to_display, [cv2.IMWRITE_JPEG_QUALITY, 70]) # Quality 70
        # print(f"VG: imencode result: {ret}") # DEBUG
        if not ret:
            # print("VG: imencode failed.") # DEBUG
            continue
        
        frame_bytes = buffer.tobytes()
        print(f"VG: Length of frame_bytes before yield: {len(frame_bytes)}") # DEBUG: Check size of encoded frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Ensure the loop runs at roughly TARGET_VIDEO_FPS
        elapsed_time = time.time() - start_time
        sleep_duration = delay_per_frame - elapsed_time
        if sleep_duration > 0:
            time.sleep(sleep_duration)

    print("Video stream generator finished.")

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not detection_active: # Or if camera not ready
        # Optionally, return a static image or a message
        print("Video feed requested but detection not active or camera not ready.")
        # For now, let's allow the generator to handle it, it might show nothing or last frame
        # A better approach might be to return a 204 No Content or a placeholder image response
        pass # Let it proceed to the generator which will likely show nothing if not active
    return Response(video_stream_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with ai_status_lock:
        s = latest_ai_status_dict if latest_ai_status_dict else {}
    with stats_lock:
        p = performance_stats if performance_stats else {}
    return jsonify({**s, "performance": p})


@app.route('/start_detection', methods=['POST'])
def start_detection_route():
    global detection_active, camera_thread, ai_processing_thread
    print("Attempting to start detection...")
    if not detection_active:
        if initialize_resources():
            detection_active = True # Set True by initialize_resources if successful
            stop_event.clear() # Ensure stop_event is clear before starting threads

            print("Starting processing threads...")
            if camera_thread is None or not camera_thread.is_alive():
                camera_thread = threading.Thread(target=camera_thread_function, daemon=True)
                camera_thread.start()
            
            if ai_processing_thread is None or not ai_processing_thread.is_alive():
                ai_processing_thread = threading.Thread(target=ai_processing_thread_function, daemon=True)
                ai_processing_thread.start()
            
            print("Processing threads started command issued.")
            return jsonify({"status": "Detection started successfully"})
        else:
            print("Failed to initialize resources for start_detection.")
            return jsonify({"status": "Failed to initialize resources"}), 500
    else:
        print("Processing threads already active or starting.")
        return jsonify({"status": "Detection is already active or starting"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection_route():
    global detection_active
    print("Attempting to stop detection...")
    if detection_active: # If it was active or trying to start
        detection_active = False # Primary signal for threads to stop
        stop_event.set()      # Event to break blocking calls like queue.get
        
        # Joining threads is now handled in release_resources, which should be called
        # when the app exits or explicitly if needed. For a simple stop, just setting flags is enough.
        # release_resources() # For immediate cleanup. Or defer to app shutdown.
        # For now, let's not release here, to allow quick restart. Resources are released on app exit.

        print("Detection stopping command issued.")
        return jsonify({"status": "Detection stopping"})
    else:
        print("Detection is not active.")
        return jsonify({"status": "Detection not active"})

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    print(f"Client connected. Total clients: {request.sid}") # request.sid is specific to Flask-SocketIO
    # Send initial full status if available
    current_full_status = {}
    with ai_status_lock:
        if latest_ai_status_dict:
            current_full_status.update(latest_ai_status_dict)
    with stats_lock:
         current_full_status["performance"] = performance_stats.copy() # Send a copy
    
    if current_full_status:
        emit('status_update', current_full_status)


@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    try:
        # Initialize resources once at the start if you want camera/AI ready on launch
        # For control via buttons, initialize_resources() is called by /start_detection
        # If you want the video feed to show something immediately (e.g., raw camera),
        # you might start camera_thread without full AI processing initially.
        # For now, let's rely on /start_detection to initialize everything.

        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Shutting down...")
    finally:
        print("Server shutting down. Ensuring processing threads are stopped...")
        release_resources()
        print("Flask-SocketIO server stopped.") 