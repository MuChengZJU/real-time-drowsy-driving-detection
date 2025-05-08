import cv2
import time
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO, emit
import threading

# Assuming ai_logic.py is in the src directory and this app.py is in the root of real-time-drowsy-driving-detection
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from ai_logic import AIDrowsinessProcessor

app = Flask(__name__, template_folder='templates') # Define template_folder
app.config['SECRET_KEY'] = 'secret_drowsiness_key!'
socketio = SocketIO(app, async_mode='threading') # Use threading for async mode

# Global variable for the AI processor instance
ai_processor = None
processing_active = False
processing_lock = threading.Lock()
client_count = 0

def initialize_processor():
    global ai_processor
    if ai_processor is None:
        print("Initializing AIDrowsinessProcessor...")
        try:
            # Adjust model paths if they are relative to the script's location or a common 'models' folder
            # Assuming 'runs' directory is at the same level as app.py
            eye_model_path = os.path.join(os.path.dirname(__file__), 'runs/detecteye/train/weights/best.pt')
            yawn_model_path = os.path.join(os.path.dirname(__file__), 'runs/detectyawn/train/weights/best.pt')
            ai_processor = AIDrowsinessProcessor(
                yolo_eye_model_path=eye_model_path, 
                yolo_yawn_model_path=yawn_model_path,
                camera_index=0 # Or a video file path for testing
            )
            print("AIDrowsinessProcessor initialized successfully.")
        except Exception as e:
            print(f"Error initializing AIDrowsinessProcessor: {e}")
            ai_processor = None # Ensure it's None if initialization fails

@app.route('/')
def index():
    return render_template('index.html')

def video_stream_generator():
    global ai_processor, processing_active
    if ai_processor is None:
        print("AI Processor not initialized for video stream.")
        # Optionally, you can return a placeholder image or an error message stream here
        # For now, just stop if not initialized.
        return

    while processing_active:
        frame_bytes = ai_processor.get_processed_frame() # This should give JPEG bytes
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If get_processed_frame returns None (e.g., end of video file), stop streaming
            # Or handle camera read errors if applicable
            # print("No frame available from processor or processor stopped.") # Reduce log noise
            pass # Just continue loop, sleep below handles rate
            # Add a small delay to prevent tight loop on error
            # time.sleep(0.1) 
            # If camera is the source, you might want to continue trying or stop based on processing_active
            # For now, if processing_active is true, we assume it tries to get a frame again.
            # if not processing_active: # Break if processing was explicitly stopped
            #      break
        # Add a small sleep here to limit the yield rate and prevent busy-looping
        time.sleep(0.03) # Aim for ~33 FPS max yield rate, actual content updates less often

@app.route('/video_feed')
def video_feed():
    if not processing_active:
        # print("Video feed request but processing is not active.")
        # Perhaps return a static image indicating a stopped feed or a 503 error.
        # For now, let's make it clear that if processing_active is false, the generator won't yield.
        pass 
    return Response(video_stream_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status_http():
    global ai_processor
    if ai_processor and processing_active:
        status = ai_processor.get_current_status()
        return jsonify(status)
    return jsonify({"error": "AI processor not active or not initialized"}), 503

# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    global client_count, processing_active, ai_processor
    client_count += 1
    print(f'Client connected. Total clients: {client_count}')
    with processing_lock:
        if not processing_active and client_count > 0:
            initialize_processor() # Initialize on first connect if not already
            if ai_processor: # Check if initialization was successful
                print("Starting AI processing thread due to client connection.")
                processing_active = True
                socketio.start_background_task(target=process_and_emit_status)
            else:
                print("AI Processor failed to initialize. Cannot start processing.")
                emit('status_update', {'error': 'AI Processor failed to initialize'})
    emit('connection_ack', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    global client_count, processing_active, ai_processor
    client_count -= 1
    print(f'Client disconnected. Total clients: {client_count}')
    if client_count == 0:
        with processing_lock:
            print("Last client disconnected. Stopping AI processing.")
            processing_active = False
            # Optionally release AI processor resources if no clients are connected for a while
            # if ai_processor:
            #     ai_processor.release()
            #     ai_processor = None 
            #     print("AI Processor resources released.")

def process_and_emit_status():
    global ai_processor, processing_active
    print("AI Processing and status emission started.")
    if not ai_processor:
        print("Cannot start status emission: AI Processor not initialized.")
        return

    target_process_interval = 0.25  # Increased target interval to 250ms (~4 FPS)
    last_process_time = time.time()

    while processing_active:
        current_time = time.time()
        
        # Check if enough time has passed since the last processing
        if current_time - last_process_time >= target_process_interval:
            try:
                # Process a frame to update internal states and get the latest status
                _, status_data = ai_processor.process_single_frame()
                last_process_time = time.time() # Update last process time *after* processing

                if status_data:
                    socketio.emit('status_update', status_data)
                else:
                    # This might happen if camera fails or video ends
                    print("No status data from AI processor.")
                    # Maybe stop processing if this persists?
                    # processing_active = False
                    # break
                    pass 
                
            except Exception as e:
                print(f"Error in processing/emitting status: {e}")
                # Consider stopping processing or attempting to reinitialize processor on certain errors
                socketio.sleep(1) # Wait a bit before retrying if an error occurs
        else:
            # Not enough time passed, sleep briefly to prevent busy-waiting
            # This sleep value can be tuned. Smaller value = more responsive check, higher CPU usage.
            socketio.sleep(0.01) # Sleep for 10ms
    
    print("AI Processing and status emission stopped.")
    # When loop finishes (processing_active is False), release resources
    if ai_processor:
        ai_processor.release()
        ai_processor = None # Reset for potential re-initialization
        print("AI Processor resources released after processing stopped.")

@app.route('/start_detection', methods=['POST'])
def start_detection_route():
    global processing_active, ai_processor
    with processing_lock:
        if not processing_active:
            initialize_processor()
            if ai_processor:
                print("Starting AI processing via POST request.")
                processing_active = True
                # Ensure the background task is started if not already (e.g. if no WS clients yet)
                # This might lead to multiple tasks if not careful; socketio handles one per target.
                # Check if a task for this target is already running might be complex.
                # Simpler: rely on client_count for starting via WebSocket connect.
                # For direct POST, we might need a separate flag or check.
                # For now, if a POST starts it, the process_and_emit_status should run.
                # This needs to be robust if multiple POSTs or WS connects happen.
                # Let's assume the socketio.start_background_task is idempotent for the same target or handled.
                socketio.start_background_task(target=process_and_emit_status)
                return jsonify({"message": "AI detection started."}), 200
            else:
                return jsonify({"error": "AI processor failed to initialize."}), 500
        return jsonify({"message": "AI detection already active."}), 200

@app.route('/stop_detection', methods=['POST'])
def stop_detection_route():
    global processing_active
    with processing_lock:
        if processing_active:
            print("Stopping AI processing via POST request.")
            processing_active = False
            # The background task will see processing_active as False and exit, releasing resources.
            return jsonify({"message": "AI detection stopped."}), 200
        return jsonify({"message": "AI detection not active."}), 200

if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    # initialize_processor() # Initialize AI processor at startup (optional, can be lazy)
    # The current logic initializes when the first client connects or /start_detection is called.
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    # Note: use_reloader=False is important for background threads and resource management like camera.
    # If debug=True and use_reloader=True, Flask might start two instances of the app. 