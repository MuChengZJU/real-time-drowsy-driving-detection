import os
import numpy as np

# Try to import the TFLite runtime
try:
    # Prefer tflite_runtime for general use
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback to full TensorFlow if tflite_runtime is not available
    try:
        import tensorflow.lite as tflite
        print("Using TensorFlow Lite interpreter from full TensorFlow package.")
    except ImportError:
        print("ERROR: Neither tflite_runtime nor TensorFlow Lite is installed.")
        print("Please install either by running: pip install tflite-runtime OR pip install tensorflow")
        exit()

MODEL_DIR = "./model/tflite/" # Relative to where the script is run

def check_model_input_details(model_path):
    """Loads a TFLite model and prints its input tensor details."""
    print(f"--- Checking Model: {model_path} ---")
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors() # Must be called before get_input_details
        
        input_details = interpreter.get_input_details()
        
        if not input_details:
            print("  ERROR: Could not retrieve input details.")
            return

        print(f"  Number of inputs: {len(input_details)}")
        for i, detail in enumerate(input_details):
            print(f"  Input #{i}:")
            print(f"    Name: {detail.get('name', 'N/A')}") # 'name' key might not always exist
            print(f"    Index: {detail['index']}")
            print(f"    Shape: {detail['shape']}")
            # shape_signature might provide dynamic shape info if available
            print(f"    Shape Signature: {detail.get('shape_signature', 'N/A')}") 
            print(f"    dtype: {detail['dtype']}")
            print(f"    Quantization (scale, zero_point): {detail['quantization']}")
            # quantization_parameters gives more verbose info
            # print(f"    Quantization Parameters: {detail.get('quantization_parameters', 'N/A')}") 
            print("-" * 20)
            
    except Exception as e:
        print(f"  ERROR loading or inspecting model {model_path}: {e}")
    print("\n")

if __name__ == "__main__":
    abs_model_dir = os.path.abspath(MODEL_DIR)
    if not os.path.isdir(abs_model_dir):
        print(f"ERROR: Model directory not found: {abs_model_dir}")
        print(f"Please ensure the script is run from '01_driver/real-time-drowsy-driving-detection/' or adjust MODEL_DIR.")
        exit()

    print(f"Scanning for .tflite models in: {abs_model_dir}\n")
    found_models = False
    for filename in os.listdir(abs_model_dir):
        if filename.endswith(".tflite"):
            found_models = True
            model_file_path = os.path.join(abs_model_dir, filename)
            check_model_input_details(model_file_path)
    
    if not found_models:
        print(f"No .tflite models found in {abs_model_dir}")

    print("Script finished.") 