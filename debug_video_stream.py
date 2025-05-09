import cv2
import time
import threading
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

latest_raw_frame = None
raw_frame_lock = threading.Lock()
stop_event = threading.Event()
camera_object = None
camera_thread = None
CAMERA_INDEX = 0
TARGET_CAMERA_FPS = 15  # Lower FPS for debugging
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

def camera_thread_function():
    global latest_raw_frame, camera_object
    print("[DEBUG_CAMERA] Camera thread starting...")
    
    if camera_object is None or not camera_object.isOpened():
        print(f"[DEBUG_CAMERA] Camera not open. Attempting to open camera {CAMERA_INDEX}")
        temp_cam = cv2.VideoCapture(CAMERA_INDEX)
        if not temp_cam.isOpened():
            print(f"[DEBUG_CAMERA] CRITICAL: Failed to open camera {CAMERA_INDEX}. Thread exiting.")
            return
        with raw_frame_lock: # Protect camera_object assignment if needed, though primarily for latest_raw_frame
            camera_object = temp_cam
        print(f"[DEBUG_CAMERA] Camera {CAMERA_INDEX} opened successfully.")

    frame_count = 0
    last_fps_calc_time = time.time()

    while not stop_event.is_set():
        if camera_object is None or not camera_object.isOpened():
            print("[DEBUG_CAMERA] Camera object is None or not opened in loop. Exiting.")
            break
            
        ret, frame = camera_object.read()
        if not ret or frame is None:
            print("[DEBUG_CAMERA] Failed to grab frame or frame is None.")
            time.sleep(0.1)
            continue

        # Resize for consistency if needed, or just use raw
        frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        with raw_frame_lock:
            latest_raw_frame = frame_resized.copy()

        frame_count += 1
        current_time = time.time()
        if current_time - last_fps_calc_time >= 5.0:
            fps = frame_count / (current_time - last_fps_calc_time)
            print(f"[DEBUG_CAMERA] Approx FPS: {fps:.2f}")
            frame_count = 0
            last_fps_calc_time = current_time
        
        if TARGET_CAMERA_FPS > 0:
            time.sleep(1.0 / TARGET_CAMERA_FPS)
        else:
            time.sleep(0.01) # Minimal sleep if FPS is 0 or negative

    print("[DEBUG_CAMERA] Camera thread finishing.")
    if camera_object is not None:
        camera_object.release()
        camera_object = None
        print("[DEBUG_CAMERA] Camera released.")


def video_stream_generator_debug():
    global latest_raw_frame
    print("[DEBUG_STREAM] Video stream generator CALLED.", flush=True)
    frame_counter = 0
    while not stop_event.is_set():
        current_frame_to_display = None
        with raw_frame_lock:
            if latest_raw_frame is not None:
                current_frame_to_display = latest_raw_frame.copy()

        if current_frame_to_display is None:
            # Create a placeholder if no frame is available yet
            current_frame_to_display = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            cv2.putText(current_frame_to_display, "Waiting for camera...", 
                        (50, int(DISPLAY_HEIGHT / 2)), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2)
            print("[DEBUG_STREAM] No raw frame, using placeholder.", flush=True)
        
        frame_counter += 1
        # Draw a simple frame count on the image for visual feedback
        cv2.putText(current_frame_to_display, f"Frame: {frame_counter}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        try:
            ret_enc, encoded_image = cv2.imencode('.jpg', current_frame_to_display)
            if not ret_enc:
                print(f"[DEBUG_STREAM] Error encoding frame {frame_counter} for streaming.", flush=True)
                time.sleep(0.1) # Avoid fast loop on encoding error
                continue
            
            print(f"[DEBUG_STREAM] Frame {frame_counter} encoded. Size: {len(encoded_image.tobytes())}", flush=True) # Log after successful encoding
            print(f"[DEBUG_STREAM] PRE-YIELD frame {frame_counter}.", flush=True)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_image.tobytes() + b'\r\n')
            print(f"[DEBUG_STREAM] POST-YIELD frame {frame_counter}.", flush=True)

        except Exception as e:
            print(f"[DEBUG_STREAM] Error during imencode or yield of frame {frame_counter}: {e}", flush=True)
            time.sleep(0.1) # Avoid fast loop on other errors

        # Control frame rate for the stream
        if TARGET_CAMERA_FPS > 0 :
             time.sleep(1.0 / TARGET_CAMERA_FPS) 
        else: # if TARGET_CAMERA_FPS is 0 or less, stream as fast as possible (with minimal delay)
             time.sleep(0.01)


    print("[DEBUG_STREAM] Video stream generator finishing.", flush=True)

@app.route('/')
def index_debug():
    return """
    <html>
        <head><title>Debug Video Stream</title></head>
        <body>
            <h1>Debug Video Stream</h1>
            <img src="/video_feed_debug" width="640" height="480">
        </body>
    </html>
    """

@app.route('/video_feed_debug')
def video_feed_debug_route():
    return Response(video_stream_generator_debug(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_debug_components():
    global camera_thread, stop_event
    stop_event.clear()
    
    # Ensure camera is opened before starting thread or handle it inside
    # For this debug, camera_thread_function will attempt to open it.
    
    print("[DEBUG_MAIN] Starting debug camera thread...")
    camera_thread = threading.Thread(target=camera_thread_function, daemon=True)
    camera_thread.start()
    print("[DEBUG_MAIN] Debug camera thread started.")

def stop_debug_components():
    global camera_thread, stop_event, camera_object
    print("[DEBUG_MAIN] Stopping debug components...")
    stop_event.set()
    if camera_thread is not None and camera_thread.is_alive():
        print("[DEBUG_MAIN] Joining camera thread...")
        camera_thread.join(timeout=2)
        if camera_thread.is_alive():
            print("[DEBUG_MAIN] Camera thread did not join in time.")
    
    # Explicit release here as well, though thread should handle it
    if camera_object is not None:
        print("[DEBUG_MAIN] Releasing camera object from main context.")
        camera_object.release()
        camera_object = None
    print("[DEBUG_MAIN] Debug components stopped.")

if __name__ == '__main__':
    try:
        print("[DEBUG_MAIN] Initializing and starting debug components...")
        start_debug_components() # Start camera capture
        print("[DEBUG_MAIN] Starting Flask app for debug streaming...")
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False) # Use a different port
    except KeyboardInterrupt:
        print("[DEBUG_MAIN] KeyboardInterrupt received.")
    finally:
        print("[DEBUG_MAIN] Application shutting down...")
        stop_debug_components()
        print("[DEBUG_MAIN] Cleanup complete. Exiting.") 