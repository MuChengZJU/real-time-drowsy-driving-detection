import os
import random
import shutil

# --- 配置参数 ---

# 源数据根目录 (包含 eye1, eye2, yawn 子目录)
SOURCE_DATA_ROOT = "data/kaggle_dataset/"

# 目标校准数据根目录 (将在项目根目录下创建 data/calibration_images_eye 和 data/calibration_images_yawn)
TARGET_CALIBRATION_ROOT = "data/"

# 为眼睛状态模型挑选的图像总数量
NUM_IMAGES_EYE_TOTAL = 100
# 为打哈欠模型挑选的图像总数量
NUM_IMAGES_YAWN_TOTAL = 100

# 允许的图像文件扩展名
ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

# --- 辅助函数 ---

def collect_image_files_from_dirs(source_dirs, allowed_extensions):
    """
    从指定的多个源目录中收集所有允许扩展名的图像文件路径。
    """
    image_files = []
    for directory in source_dirs:
        if not os.path.isdir(directory):
            print(f"警告：目录 '{directory}' 不存在或不是一个目录，已跳过。")
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(allowed_extensions):
                    image_files.append(os.path.join(root, file))
    return image_files

def copy_selected_images_to_target(selected_image_files, target_calibration_dir, model_name=""):
    """
    将【已经选择好】的图像文件列表复制到目标校准目录。
    确保目标目录存在，如果不存在则创建。
    """
    if not selected_image_files:
        print(f"提示：{model_name} 没有最终选择的图像文件进行复制。")
        return 0 # 返回复制的数量

    if not os.path.exists(target_calibration_dir):
        try:
            os.makedirs(target_calibration_dir)
            print(f"已创建目标目录: '{target_calibration_dir}'")
        except OSError as e:
            print(f"错误：无法创建目标目录 '{target_calibration_dir}': {e}")
            return 0
    else:
        print(f"目标目录 '{target_calibration_dir}' 已存在。")

    copied_count = 0
    print(f"开始为 {model_name} 复制 {len(selected_image_files)} 张选定的图像到 '{target_calibration_dir}'...")
    for img_path in selected_image_files:
        try:
            base_name = os.path.basename(img_path)
            target_path = os.path.join(target_calibration_dir, base_name)
            
            # 为了避免覆盖来自不同源文件夹的同名文件，如果目标已存在，则给新文件一个稍微不同的名字。
            counter = 0
            original_target_path = target_path
            while os.path.exists(target_path):
                counter += 1
                name, ext = os.path.splitext(base_name)
                target_path = os.path.join(target_calibration_dir, f"{name}_{counter}{ext}")
            
            if counter > 0:
                print(f"提示：文件 '{original_target_path}' 已存在。将 '{img_path}' 复制为 '{target_path}'")
            
            shutil.copy(img_path, target_path)
            copied_count += 1
        except Exception as e:
            print(f"复制文件 '{img_path}' 到 '{target_calibration_dir}' 时出错: {e}")

    print(f"成功为 {model_name} 复制 {copied_count} 张图像到 '{target_calibration_dir}'。")
    if len(selected_image_files) > copied_count:
        print(f"有 {len(selected_image_files) - copied_count} 张图像未能成功复制。")
    return copied_count

# --- 主逻辑 ---
def main():
    """
    主函数，执行校准数据准备过程。
    """
    print("开始准备校准数据集...")
    print(f"源数据根目录: {os.path.abspath(SOURCE_DATA_ROOT)}")
    print(f"目标校准数据根目录: {os.path.abspath(TARGET_CALIBRATION_ROOT)}")

    # --- 1. 准备眼睛状态检测模型的校准数据 ---
    print("\n--- 处理眼睛状态检测模型数据 ---")
    eye_target_calib_dir = os.path.join(TARGET_CALIBRATION_ROOT, "calibration_images_eye")
    
    # 定义眼睛状态的类别及其对应的源文件夹
    eye_classes_sources = {
        "open_eyes": [
            os.path.join(SOURCE_DATA_ROOT, "eye1", "test", "open eyes"),
            os.path.join(SOURCE_DATA_ROOT, "eye1", "train", "open eyes"),
            os.path.join(SOURCE_DATA_ROOT, "eye2", "train", "open"),
            os.path.join(SOURCE_DATA_ROOT, "eye2", "val", "open"),
        ],
        "closed_eyes": [
            os.path.join(SOURCE_DATA_ROOT, "eye1", "test", "close eyes"),
            os.path.join(SOURCE_DATA_ROOT, "eye1", "train", "close eyes"),
            os.path.join(SOURCE_DATA_ROOT, "eye2", "train", "closed"),
            os.path.join(SOURCE_DATA_ROOT, "eye2", "val", "closed"),
        ]
    }
    
    final_selected_eye_images = []
    num_eye_classes = len(eye_classes_sources)

    if num_eye_classes > 0:
        # 计算每个类别大致应该挑选的图像数量
        base_num_per_class = NUM_IMAGES_EYE_TOTAL // num_eye_classes
        remainder = NUM_IMAGES_EYE_TOTAL % num_eye_classes
        print(f"计划为眼睛模型从 {num_eye_classes} 个类别中挑选总共约 {NUM_IMAGES_EYE_TOTAL} 张图像。")

        current_class_index = 0
        for class_name, source_dirs in eye_classes_sources.items():
            print(f"  处理眼睛类别: '{class_name}'")
            class_image_files = collect_image_files_from_dirs(source_dirs, ALLOWED_EXTENSIONS)
            print(f"    找到 {len(class_image_files)} 张图像。")
            
            num_to_pick_for_this_class = base_num_per_class
            if current_class_index < remainder: # 将余数分配给前面的类别
                num_to_pick_for_this_class += 1
            current_class_index += 1

            if not class_image_files:
                print(f"    警告：类别 '{class_name}' 没有图像可选。目标数量 {num_to_pick_for_this_class} 未满足。")
                continue

            if len(class_image_files) < num_to_pick_for_this_class:
                print(f"    警告：类别 '{class_name}' 图像数量 ({len(class_image_files)}) 少于目标数量 ({num_to_pick_for_this_class})。将选取所有可用图像 ({len(class_image_files)})。")
                selected_for_class = class_image_files # 取全部
            else:
                selected_for_class = random.sample(class_image_files, num_to_pick_for_this_class)
            
            final_selected_eye_images.extend(selected_for_class)
            print(f"    为类别 '{class_name}' 选取了 {len(selected_for_class)} 张图像。")
    else:
        print("错误：没有为眼睛模型定义类别源数据。")

    # 打乱最终选取的图像列表，确保混合
    random.shuffle(final_selected_eye_images)
    print(f"总共为眼睛模型选取了 {len(final_selected_eye_images)} 张图像进行校准。")
    copy_selected_images_to_target(final_selected_eye_images, eye_target_calib_dir, model_name="眼睛状态检测模型")

    # --- 2. 准备打哈欠检测模型的校准数据 ---
    print("\n--- 处理打哈欠检测模型数据 ---")
    yawn_target_calib_dir = os.path.join(TARGET_CALIBRATION_ROOT, "calibration_images_yawn")
    
    yawn_classes_sources = {
        "yawn": [os.path.join(SOURCE_DATA_ROOT, "yawn", "yawn")],
        "no_yawn": [os.path.join(SOURCE_DATA_ROOT, "yawn", "no yawn")]
    }

    final_selected_yawn_images = []
    num_yawn_classes = len(yawn_classes_sources)

    if num_yawn_classes > 0:
        base_num_per_class = NUM_IMAGES_YAWN_TOTAL // num_yawn_classes
        remainder = NUM_IMAGES_YAWN_TOTAL % num_yawn_classes
        print(f"计划为打哈欠模型从 {num_yawn_classes} 个类别中挑选总共约 {NUM_IMAGES_YAWN_TOTAL} 张图像。")
        
        current_class_index = 0
        for class_name, source_dirs in yawn_classes_sources.items():
            print(f"  处理打哈欠类别: '{class_name}'")
            class_image_files = collect_image_files_from_dirs(source_dirs, ALLOWED_EXTENSIONS)
            print(f"    找到 {len(class_image_files)} 张图像。")

            num_to_pick_for_this_class = base_num_per_class
            if current_class_index < remainder:
                num_to_pick_for_this_class += 1
            current_class_index +=1
            
            if not class_image_files:
                print(f"    警告：类别 '{class_name}' 没有图像可选。目标数量 {num_to_pick_for_this_class} 未满足。")
                continue

            if len(class_image_files) < num_to_pick_for_this_class:
                print(f"    警告：类别 '{class_name}' 图像数量 ({len(class_image_files)}) 少于目标数量 ({num_to_pick_for_this_class})。将选取所有可用图像 ({len(class_image_files)})。")
                selected_for_class = class_image_files
            else:
                selected_for_class = random.sample(class_image_files, num_to_pick_for_this_class)
            
            final_selected_yawn_images.extend(selected_for_class)
            print(f"    为类别 '{class_name}' 选取了 {len(selected_for_class)} 张图像。")
    else:
        print("错误：没有为打哈欠模型定义类别源数据。")
            
    random.shuffle(final_selected_yawn_images)
    print(f"总共为打哈欠模型选取了 {len(final_selected_yawn_images)} 张图像进行校准。")
    copy_selected_images_to_target(final_selected_yawn_images, yawn_target_calib_dir, model_name="打哈欠检测模型")

    print("\n校准数据准备脚本执行完毕。")
    print(f"请检查以下目录中生成的校准数据：")
    print(f"- 眼睛模型: {os.path.abspath(eye_target_calib_dir)}")
    print(f"- 打哈欠模型: {os.path.abspath(yawn_target_calib_dir)}")

if __name__ == "__main__":
    # 假设此脚本位于 real-time-drowsy-driving-detection/scripts/ 目录下
    # 并且期望从项目根目录 (real-time-drowsy-driving-detection) 执行
    # 例如: python scripts/kaggle_dataset_Images.py
    # 路径 SOURCE_DATA_ROOT 和 TARGET_CALIBRATION_ROOT 是相对于项目根目录的。
    main()