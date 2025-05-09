from ultralytics import YOLO
import os
import shutil # 用于文件移动

# --- 配置 ---
# 模型路径
EYE_MODEL_PT_PATH = 'runs/detecteye/train/weights/best.pt'
YAWN_MODEL_PT_PATH = 'runs/detectyawn/train/weights/best.pt'

# 校准数据 YAML 文件路径 (使用新创建的YAML文件)
EYE_CALIBRATION_DATA_YAML = 'eye_calibration_config.yaml'
YAWN_CALIBRATION_DATA_YAML = 'yawn_calibration_config.yaml'

# 要测试的图像尺寸列表
IMAGE_SIZES_TO_EXPORT = [320, 640]

# 目标输出目录
TARGET_OUTPUT_DIR = './model/tflite/'

# --- 确保目标输出目录存在 ---
os.makedirs(TARGET_OUTPUT_DIR, exist_ok=True)
print(f"所有转换后的模型将保存在: {os.path.abspath(TARGET_OUTPUT_DIR)}")

# --- 辅助函数，用于生成输出路径 ---
def get_output_path(original_pt_path, image_size, precision_str):
    """
    生成 TFLite 模型的输出路径。
    Ultralytics 8.0.x export 会将模型保存在 <original_pt_dir>/<original_pt_name>_saved_model/ 目录中。
    文件名将是 <original_pt_name>_<precision_suffix>.tflite。
    例如: runs/detecteye/train/weights/best.pt -> runs/detecteye/train/weights/best_saved_model/best_int8.tflite
    我们将基于此结构来指明预期的输出。
    """
    model_name = os.path.splitext(os.path.basename(original_pt_path))[0] # 例如 'best'
    model_dir = os.path.dirname(original_pt_path)
    # Ultralytics 会自动处理实际的文件名，这里我们构建一个预期的提示路径
    # 实际文件名中的 imgsz 通常不会显式体现，而是通过 _saved_model 文件夹区分（如果导出器这样做）
    # 但 Ultralytics 的 TFLite 导出文件名主要是基于精度后缀
    return os.path.join(model_dir, f"{model_name}_saved_model", f"{model_name}_{image_size}_{precision_str}.tflite")

# --- 主转换逻辑 ---
models_to_convert = [
    {"name": "眼睛状态检测", "pt_path": EYE_MODEL_PT_PATH, "calibration_data": EYE_CALIBRATION_DATA_YAML, "output_prefix": "eye_detect"},
    {"name": "打哈欠检测", "pt_path": YAWN_MODEL_PT_PATH, "calibration_data": YAWN_CALIBRATION_DATA_YAML, "output_prefix": "yawn_detect"},
]

for model_info in models_to_convert:
    print(f"\n{'='*20} 开始处理模型: {model_info['name']} ({model_info['pt_path']}) {'='*20}")
    model = YOLO(model_info['pt_path'])
    
    original_model_dir = os.path.dirname(model_info['pt_path'])
    original_base_name = os.path.splitext(os.path.basename(model_info['pt_path']))[0] # e.g., 'best'
    # Ultralytics 导出的模型会放在 <original_model_dir>/<original_base_name>_saved_model/
    ultralytics_output_dir = os.path.join(original_model_dir, f"{original_base_name}_saved_model")

    for img_size in IMAGE_SIZES_TO_EXPORT:
        print(f"\n--- 图像尺寸: {img_size} ---")

        # 1. 转换为 INT8 TFLite
        try:
            print(f"正在转换为 INT8 TFLite (imgsz: {img_size}) 使用配置: {model_info['calibration_data']}...")
            model.export(
                format='tflite',
                imgsz=img_size,
                int8=True,
                data=model_info['calibration_data'], # 使用YAML配置文件
            )
            
            source_file_name_int8 = f"{original_base_name}_int8.tflite"
            source_file_path_int8 = os.path.join(ultralytics_output_dir, source_file_name_int8)

            target_file_name_int8 = f"{model_info['output_prefix']}_{img_size}_int8.tflite"
            target_file_path_int8 = os.path.join(TARGET_OUTPUT_DIR, target_file_name_int8)

            if os.path.exists(source_file_path_int8):
                shutil.move(source_file_path_int8, target_file_path_int8)
                print(f"INT8 TFLite 模型 ({model_info['name']}, imgsz: {img_size}) 转换完成并移动到: {target_file_path_int8}")
            else:
                print(f"错误：未找到预期的 INT8 TFLite 输出文件: {source_file_path_int8}")

        except Exception as e:
            print(f"转换为 INT8 TFLite ({model_info['name']}, imgsz: {img_size}) 时出错: {e}")

        # 2. 转换为 FP16 TFLite (半精度)
        try:
            print(f"正在转换为 FP16 TFLite (imgsz: {img_size})...")
            model.export(
                format='tflite',
                imgsz=img_size,
                half=True,
            )

            # Ultralytics 生成的文件名是 _float16.tflite
            source_file_name_fp16 = f"{original_base_name}_float16.tflite" 
            source_file_path_fp16 = os.path.join(ultralytics_output_dir, source_file_name_fp16)
            
            target_file_name_fp16 = f"{model_info['output_prefix']}_{img_size}_fp16.tflite" #我们期望的目标文件名
            target_file_path_fp16 = os.path.join(TARGET_OUTPUT_DIR, target_file_name_fp16)

            if os.path.exists(source_file_path_fp16):
                shutil.move(source_file_path_fp16, target_file_path_fp16)
                print(f"FP16 TFLite 模型 ({model_info['name']}, imgsz: {img_size}) 转换完成并移动到: {target_file_path_fp16}")
            else:
                # Fallback in case the naming convention changes or for older versions
                fallback_source_name_fp16 = f"{original_base_name}_fp16.tflite"
                fallback_source_path_fp16 = os.path.join(ultralytics_output_dir, fallback_source_name_fp16)
                if os.path.exists(fallback_source_path_fp16):
                    shutil.move(fallback_source_path_fp16, target_file_path_fp16)
                    print(f"FP16 TFLite 模型 ({model_info['name']}, imgsz: {img_size}) (using fallback name {fallback_source_name_fp16}) 转换完成并移动到: {target_file_path_fp16}")
                else:
                    print(f"错误：未找到预期的 FP16 TFLite 输出文件 ({source_file_name_fp16} 或 {fallback_source_name_fp16}) 在: {ultralytics_output_dir}")
                
        except Exception as e:
            print(f"转换为 FP16 TFLite ({model_info['name']}, imgsz: {img_size}) 时出错: {e}")
            
    # 可选：清理空的 _saved_model 目录
    # try:
    #     if os.path.exists(ultralytics_output_dir) and not os.listdir(ultralytics_output_dir):
    #         print(f"清理空的目录: {ultralytics_output_dir}")
    #         shutil.rmtree(ultralytics_output_dir)
    #     elif os.path.exists(ultralytics_output_dir):
    #         # 如果目录不为空，可能是因为生成了其他类型的文件（如 .onnx, .pb）
    #         print(f"注意: 目录 {ultralytics_output_dir} 中可能还有其他文件，未被删除。")
    # except Exception as e:
    #     print(f"清理目录 {ultralytics_output_dir} 时出错: {e}")


print("\n" + "="*50 + "\n")
print("所有转换过程完成。")
print(f"请检查输出目录: {os.path.abspath(TARGET_OUTPUT_DIR)}")
print("文件名将包含模型前缀 (eye_detect/yawn_detect), 图像尺寸 (320/640), 和精度 (int8/fp16)。")

print("\n" + "="*50 + "\n")
print("所有转换过程完成。")
print("请检查各个模型 'runs/detect***/train/weights/<model_name>_saved_model/' 目录下的输出文件。")
print("文件名将包含 '_int8' 或 '_fp16' 后缀。")
print("注意：Ultralytics 在导出 TFLite 时，imgsz 参数主要影响模型内部的输入定义，")
print("导出的 TFLite 文件名本身通常不直接包含图像尺寸（例如 320 或 640），")
print("而是通过 _int8.tflite 或 _fp16.tflite 这样的后缀来区分精度。")
print("您需要根据导出时使用的 imgsz 参数来对应哪个模型是哪个尺寸的。") 