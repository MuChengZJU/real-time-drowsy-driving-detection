import cv2
import time
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO, emit
import threading
import queue # Import queue
import psutil # Import psutil

# Assuming ai_logic.py is in the src directory and this app.py is in the root of real-time-drowsy-driving-detection
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from ai_logic import AIDrowsinessProcessor

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'secret_drowsiness_key!'
socketio = SocketIO(app, async_mode='threading')

# --- Global Variables for Shared Resources and State ---
ai_processor = None
cap = None
frame_queue = queue.Queue(maxsize=1) # Queue to hold the latest frame
stop_event = threading.Event()       # Event to signal threads to stop
processing_thread = None
camera_thread = None
processing_active = False
processing_lock = threading.Lock()   # Lock for safely managing state transitions
client_count = 0
current_process = psutil.Process(os.getpid()) # Get current process for monitoring
psutil.cpu_percent(percpu=True) # Initialize per-core usage calculation

# --- Camera Capture Thread ---
def capture_frames(camera_device, frame_q, stop_evt):
    global cap
    print("Camera thread started.")
    if not cap or not cap.isOpened():
        print(f"Error: Camera {camera_device} is not opened in capture thread.")
        return

    frame_read_count = 0
    fps_start_time = time.time()

    while not stop_evt.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera.")
            # Optionally add a small delay or break if errors persist
            time.sleep(0.1)
            continue
        
        frame_read_count += 1
        # Put the latest frame into the queue, overwriting if full
        try:
            frame_q.put_nowait(frame) 
        except queue.Full:
            # Queue is full (size 1), try to get and discard the old one, then put the new one
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass # Should not happen if it was full, but handle anyway
            try:
                frame_q.put_nowait(frame)
            except queue.Full:
                print("Warning: Could not place frame in queue even after clearing.")
                pass # Skip frame if queue is still somehow full
        
        # Calculate and print Camera FPS periodically
        current_time = time.time()
        elapsed_time = current_time - fps_start_time
        if elapsed_time >= 2.0: # Print every 2 seconds
            camera_fps = frame_read_count / elapsed_time
            print(f"Camera FPS: {camera_fps:.2f}")
            fps_start_time = current_time
            frame_read_count = 0

        # Small sleep to prevent this thread from consuming 100% CPU if camera is fast
        time.sleep(0.005) # Adjust as needed, e.g., 5ms

    print("Camera thread finished.")

# --- AI Processing Thread ---
def process_and_emit_status(frame_q, stop_evt):
    global ai_processor, processing_active
    print("AI Processing and status emission started.")
    if not ai_processor:
        print("Cannot start status emission: AI Processor not initialized.")
        return

    target_process_interval = 0.25  # Keep the interval for *attempting* processing
    last_process_time = time.time()
    latest_frame = None
    
    # Reset AI FPS calculation, focus on processing time
    # ai_frame_processed_count = 0
    # ai_fps_start_time = time.time()
    last_monitor_time = time.time()
    cpu_per_core_usage = [0.0] * psutil.cpu_count() # Initialize per-core list
    ram_usage_mb = 0.0
    processing_time_ms = 0.0

    while not stop_evt.is_set():
        try:
            latest_frame = frame_q.get_nowait()
        except queue.Empty:
            pass 

        if latest_frame is None: 
            socketio.sleep(0.01)
            continue
            
        current_time = time.time()
        
        # Process frame if interval has passed
        if current_time - last_process_time >= target_process_interval:
            try:
                frame_process_start_time = time.time()
                _, status_data = ai_processor.process_frame(latest_frame)
                frame_process_end_time = time.time()
                processing_time_ms = (frame_process_end_time - frame_process_start_time) * 1000 # Calculate in ms
                last_process_time = frame_process_end_time # Update based on actual end time
                
                # Calculate Monitor Stats periodically (less frequently is fine)
                if current_time - last_monitor_time >= 2.0:
                    cpu_per_core_usage = psutil.cpu_percent(percpu=True) # Get list of per-core usage
                    ram_usage_mb = current_process.memory_info().rss / (1024 * 1024)
                    last_monitor_time = current_time
                    # Print more detailed info to console
                    print(f"AI Frame Process Time: {processing_time_ms:.1f} ms | RAM: {ram_usage_mb:.1f} MB | CPU per Core: {[f'{c:.1f}%' for c in cpu_per_core_usage]}")

                if status_data:
                    # Add detailed monitor stats to status data
                    status_data['processing_time_ms'] = round(processing_time_ms, 1)
                    status_data['cpu_per_core_usage'] = [round(c, 1) for c in cpu_per_core_usage]
                    status_data['ram_usage_mb'] = round(ram_usage_mb, 1)
                    socketio.emit('status_update', status_data)
                else:
                    print("No status data from AI processor processing.")
                    pass 
                
            except Exception as e:
                print(f"Error in processing/emitting status: {e}")
                # Consider adding traceback for detailed debugging
                # import traceback
                # traceback.print_exc()
                socketio.sleep(1) 
        else:
            socketio.sleep(0.01) 
    
    print("AI Processing and status emission stopped.")

# --- Resource Initialization and Management ---
def initialize_resources(camera_index=0):
    global ai_processor, cap
    print("Initializing resources...")
    resources_initialized = False
    try:
        if cap is None or not cap.isOpened():
            print(f"Opening camera {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                 raise IOError(f"Cannot open camera {camera_index}")
            print("Camera opened successfully.")
            time.sleep(1.0) # Camera warmup

        if ai_processor is None:
            print("Initializing AIDrowsinessProcessor...")
            eye_model_path = os.path.join(os.path.dirname(__file__), 'runs/detecteye/train/weights/best.pt')
            yawn_model_path = os.path.join(os.path.dirname(__file__), 'runs/detectyawn/train/weights/best.pt')
            # Initialize without camera index now
            ai_processor = AIDrowsinessProcessor(
                yolo_eye_model_path=eye_model_path, 
                yolo_yawn_model_path=yawn_model_path
            )
            print("AIDrowsinessProcessor initialized successfully.")
        resources_initialized = True
    except Exception as e:
        print(f"Error initializing resources: {e}")
        # Clean up partially initialized resources
        if cap and cap.isOpened():
            cap.release()
            cap = None
        ai_processor = None 
    return resources_initialized

def start_processing_threads():
    global processing_active, camera_thread, processing_thread, stop_event, frame_queue
    if not processing_active:
        if initialize_resources():
            print("Starting processing threads...")
            stop_event.clear()
            # Ensure queue is empty before starting
            while not frame_queue.empty():
                try: frame_queue.get_nowait() 
                except queue.Empty: break
                
            camera_thread = threading.Thread(target=capture_frames, args=(0, frame_queue, stop_event))
            processing_thread = threading.Thread(target=process_and_emit_status, args=(frame_queue, stop_event))
            
            processing_active = True # Set active before starting threads
            camera_thread.start()
            processing_thread.start()
            print("Processing threads started.")
            return True
        else:
            print("Failed to initialize resources. Cannot start processing.")
            return False
    else:
        print("Processing threads already active.")
        return True # Already running

def stop_processing_threads():
    global processing_active, camera_thread, processing_thread, stop_event, cap, ai_processor
    if processing_active:
        print("Stopping processing threads...")
        stop_event.set()
        
        if camera_thread and camera_thread.is_alive():
            camera_thread.join()
            print("Camera thread joined.")
        if processing_thread and processing_thread.is_alive():
            processing_thread.join()
            print("Processing thread joined.")

        # Release resources after threads have stopped
        if cap and cap.isOpened():
            cap.release()
            cap = None
            print("Camera released.")
        if ai_processor:
            ai_processor.release() # Call the processor's release method
            ai_processor = None
            print("AI Processor released.")
            
        processing_active = False
        camera_thread = None
        processing_thread = None
        print("Processing threads stopped and resources released.")
        return True
    else:
        print("Processing threads not active.")
        return False

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

def video_stream_generator():
    global ai_processor, processing_active
    # Video stream now relies on the latest processed frame from ai_processor
    # It doesn't need direct access to the camera thread or queue
    print("Video stream generator started.")
    while processing_active:
        if ai_processor:
            frame_bytes = ai_processor.get_processed_frame() # Get last frame processed by AI thread
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # AI processor might be initializing or not have processed a frame yet
                pass 
        else:
             # AI processor not available (e.g., during stop/start)
             pass
             
        # Limit the rate at which the generator yields frames
        time.sleep(0.03) # Aim for ~33 FPS max yield rate
    print("Video stream generator finished.")

@app.route('/video_feed')
def video_feed():
    # The generator itself checks processing_active
    return Response(video_stream_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status_http():
    global ai_processor
    # Return default/inactive status as processing time/per-core usage might not be ready
    default_status = {
        "blinks": 0, "microsleeps_duration": 0.0, "yawns": 0,
        "yawn_duration": 0.0, "left_eye_state": "-", "right_eye_state": "-",
        "yawn_state": "-", "overall_alert": "Inactive",
        "processing_time_ms": 0.0,
        "cpu_per_core_usage": [0.0] * psutil.cpu_count(),
        "ram_usage_mb": 0.0
    }
    if ai_processor and processing_active:
        # Get latest status, but monitoring stats might be slightly delayed
        status = ai_processor.get_current_status()
        # Fill in defaults if monitoring stats aren't in the base status
        status['processing_time_ms'] = status.get('processing_time_ms', 0.0)
        status['cpu_per_core_usage'] = status.get('cpu_per_core_usage', [0.0] * psutil.cpu_count())
        status['ram_usage_mb'] = status.get('ram_usage_mb', 0.0)
        return jsonify(status)
    return jsonify(default_status), 200

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    global client_count
    client_count += 1
    print(f'Client connected. Total clients: {client_count}')
    with processing_lock:
        # Automatically start processing if not already active when first client connects
        if not processing_active and client_count == 1:
             if not start_processing_threads():
                  # Failed to start, notify client
                  emit('status_update', {'error': 'AI Processor failed to initialize or start'})

    # Always send ack
    emit('connection_ack', {'message': 'Connected to server'})
    # Send initial status, including defaults for monitor stats
    if ai_processor and processing_active:
         initial_status = ai_processor.get_current_status()
         initial_status['processing_time_ms'] = 0.0
         initial_status['cpu_per_core_usage'] = [0.0] * psutil.cpu_count()
         initial_status['ram_usage_mb'] = 0.0
         emit('status_update', initial_status)

@socketio.on('disconnect')
def handle_disconnect():
    global client_count
    client_count -= 1
    print(f'Client disconnected. Total clients: {client_count}')
    if client_count <= 0: # Use <= 0 for safety
        client_count = 0 # Reset to 0 if it goes negative somehow
        with processing_lock:
            # Stop processing only if no clients are connected
            stop_processing_threads()

# --- Manual Control Routes ---
@app.route('/start_detection', methods=['POST'])
def start_detection_route():
    with processing_lock:
        if start_processing_threads():
            return jsonify({"message": "AI detection started or already active."}), 200
        else:
            return jsonify({"error": "Failed to start AI detection."}), 500

@app.route('/stop_detection', methods=['POST'])
def stop_detection_route():
    with processing_lock:
        if stop_processing_threads():
            return jsonify({"message": "AI detection stopped."}), 200
        else:
             return jsonify({"message": "AI detection not active."}), 200

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    # Resources (camera, AI model) are now initialized lazily when processing starts
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    finally:
        # Ensure resources are released on server shutdown if still active
        print("Server shutting down. Ensuring processing threads are stopped...")
        stop_processing_threads() 