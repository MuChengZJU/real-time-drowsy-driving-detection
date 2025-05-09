# YOLOv8 模型到 TFLite 转换与 INT8 量化指南

## 1. 引言与目标

本文档旨在指导您如何将项目中训练好的 YOLOv8 模型（用于眼睛状态检测和打哈欠检测）转换为 TensorFlow Lite (TFLite) 格式，并应用 INT8 量化。目标是生成轻量级、推理速度更快的模型，以便在资源受限的设备（如树莓派）上高效运行，从而解决当前单帧计算时间过长的问题。

根据项目结构，我们有两个主要的模型需要转换：
-   眼睛状态检测模型: `runs/detecteye/train/weights/best.pt`
-   打哈欠检测模型: `runs/detectyawn/train/weights/best.pt`

## 2. 环境准备

确保您的 Python 环境中已安装必要的库。主要依赖 `ultralytics` 包，它会自动处理 TensorFlow Lite 的转换。如果尚未安装或版本较旧，请更新：

```bash
pip install ultralytics tensorflow
```
或者，如果已经安装了 `requirements.txt`，通常已经包含了这些。

## 3. 转换与 INT8 量化步骤

以下步骤应在具有完整Python环境（例如您的Ubuntu开发服务器）上执行，而不是直接在树莓派上执行。

### 3.1. 准备校准数据集 (Calibration Data) - 【非常重要】

INT8 量化通过将模型的浮点权重转换为8位整数来减小模型大小并加速推理。为了在量化过程中最大限度地减少精度损失，需要一个"校准数据集"。这个数据集应该包含一小部分（例如 50-100 张）来自您的训练集或验证集的代表性图像。

**操作：**
1.  **收集图像**：从您的眼睛状态检测和打哈欠检测数据集中挑选出一些有代表性的图像。这些图像应该能覆盖模型在实际应用中可能遇到的各种场景。
2.  **创建数据YAML文件**：YOLOv8 的导出工具需要一个 YAML 文件来指定校准数据。创建一个名为 `calibration_data.yaml` (或类似名称) 的文件，内容如下：

    ```yaml
    # calibration_data.yaml
    # 确保路径对于执行转换脚本的机器是正确的

    # 指向包含校准图像的目录
    # path: /path/to/your/calibration_images_directory # 方法1: 指定一个目录

    # 或者，分别列出图像路径 (如果图像分散)
    # train: # 实际上用作校准，可以不用 val 和 test
    #   - /path/to/image1.jpg
    #   - /path/to/image2.png
    #   - ...

    # 对于 Ultralytics YOLOv8 TFLite INT8 量化，通常只需要提供一个包含图像的目录路径，
    # 或者一个指向训练数据配置文件（如训练时使用的 data.yaml）的路径，
    # 只要该配置文件能让 Ultralytics 找到图像即可。

    # 最简单的方式是创建一个包含校准图像的文件夹，例如 'data/calibration_images/'
    # 然后在导出时，将 'data' 参数设置为这个文件夹的父目录，或者直接指向训练时用的 data.yaml
    # 例如，如果您的训练数据 YAML 像这样:
    #   train: ../datasets/coco/images/train2017
    #   val: ../datasets/coco/images/val2017
    # 您可以在导出时使用这个训练的 data.yaml
    # model.export(format='tflite', data='path/to/your_training_data.yaml', int8=True)

    # 为了简单起见，建议创建一个专门的校准图像文件夹，例如：
    # project_root/
    #  ├── data/
    #  │   └── calibration_images/  <-- 将你的50-100张校准图片放在这里
    #  │       ├── img1.jpg
    #  │       └── img2.jpg
    #  └── (其他文件和文件夹)
    #
    # 然后在导出脚本的 `data` 参数中，可以使用指向这个 calibration_images 文件夹的相对或绝对路径。
    # 或者，如果您的训练数据已经有了一个 `data.yaml` (例如 `coco128.yaml` 的格式)，
    # 并且您的校准图像是其中的一部分，可以直接使用那个 `data.yaml`。

    # 假设您创建了一个 'data/calibration_images/' 目录并填充了图片，
    # 在Python脚本中导出时，`data` 参数可以设置为 'data/calibration_images/'。
    # 或者，您可以创建一个简单的yaml文件，例如 `custom_calibration.yaml`:
    # train: path/to/your/calibration_images_directory
    # val: path/to/your/calibration_images_directory # 或者省略
    # nc: <number_of_classes_for_the_model> # 确保这个与模型匹配
    # names: ['class1', 'class2', ...] # 确保这个与模型匹配

    # 对于本项目，因为眼睛和打哈欠模型可能使用不同的数据集，
    # 您可能需要为每个模型准备单独的校准图像集或YAML。
    # 例如，创建一个 `eye_calibration_data.yaml` 和 `yawn_calibration_data.yaml`。

    # eye_calibration_data.yaml:
    # path: path/to/eye_calibration_images/
    # train: images/ # 假设图像在 path 下的 images 子目录
    # val: images/
    # nc: 2 # 例如: open_eye, close_eye
    # names: ['open_eye', 'close_eye']

    # yawn_calibration_data.yaml:
    # path: path/to/yawn_calibration_images/
    # train: images/
    # val: images/
    # nc: 2 # 例如: yawn, no_yawn
    # names: ['yawn', 'no_yawn']
    ```
    **重要提示**：请根据您的实际数据集结构调整此 `calibration_data.yaml`。最简单的方法是创建一个包含校准图像的文件夹，然后在导出脚本中直接使用该文件夹的路径或指向包含该文件夹路径的简单YAML文件。

### 3.2. Python 脚本进行转换和量化

创建一个 Python 脚本（例如 `export_tflite.py`）来执行转换。

```python
from ultralytics import YOLO

# --- 配置 ---
# 模型路径
EYE_MODEL_PT_PATH = 'runs/detecteye/train/weights/best.pt'
YAWN_MODEL_PT_PATH = 'runs/detectyawn/train/weights/best.pt'

# 校准数据配置文件 (确保您已创建这些文件或相应的目录)
# 请根据您的实际情况修改路径
EYE_CALIBRATION_YAML = 'path/to/your/eye_calibration_data.yaml' # 或者直接是校准图片目录
YAWN_CALIBRATION_YAML = 'path/to/your/yawn_calibration_data.yaml' # 或者直接是校准图片目录

# 输出 TFLite 模型的文件名
EYE_MODEL_TFLITE_INT8_PATH = 'runs/detecteye/train/weights/best_int8.tflite'
YAWN_MODEL_TFLITE_INT8_PATH = 'runs/detectyawn/train/weights/best_int8.tflite'

# 图像尺寸 (imgsz) - 应与训练时使用的尺寸或树莓派推理时期望的输入尺寸一致
# 常见的尺寸有 320, 416, 640。较小的尺寸推理更快，但精度可能略低。
# 对于树莓派，可以从 320 或 416 开始尝试。
IMAGE_SIZE = 320 # 或者 416, 640

# --- 转换眼睛状态检测模型 ---
try:
    print(f"正在转换眼睛状态检测模型: {EYE_MODEL_PT_PATH}")
    eye_model = YOLO(EYE_MODEL_PT_PATH)
    eye_model.export(
        format='tflite',
        imgsz=IMAGE_SIZE,
        int8=True,
        data=EYE_CALIBRATION_YAML, # 关键：提供校准数据
        # optimize=True, # tflite 导出时默认会进行优化
        # half=False, # INT8量化不需要FP16
    )
    print(f"眼睛状态检测模型成功转换为 INT8 TFLite: {EYE_MODEL_TFLITE_INT8_PATH}")
    print("注意：实际输出文件名可能由 Ultralytics 自动命名为 <model_name>_saved_model/<model_name>_int8.tflite 或类似结构，请检查导出日志。上述路径是期望路径。")
    print("通常，导出的文件会位于原 .pt 文件相同的目录下，并带有 _int8.tflite 后缀，例如 best_int8.tflite")

except Exception as e:
    print(f"转换眼睛状态检测模型时出错: {e}")

print("\n" + "="*50 + "\n")

# --- 转换打哈欠检测模型 ---
try:
    print(f"正在转换打哈欠检测模型: {YAWN_MODEL_PT_PATH}")
    yawn_model = YOLO(YAWN_MODEL_PT_PATH)
    yawn_model.export(
        format='tflite',
        imgsz=IMAGE_SIZE,
        int8=True,
        data=YAWN_CALIBRATION_YAML, # 关键：提供校准数据
    )
    print(f"打哈欠检测模型成功转换为 INT8 TFLite: {YAWN_MODEL_TFLITE_INT8_PATH}")
    print("注意：实际输出文件名可能由 Ultralytics 自动命名，请检查导出日志。")

except Exception as e:
    print(f"转换打哈欠检测模型时出错: {e}")

print("\n转换过程完成。请检查 'runs/detecteye/train/weights/' 和 'runs/detectyawn/train/weights/' 目录下生成的 TFLite 文件。")
print(f"预期的输出文件是 '{EYE_MODEL_TFLITE_INT8_PATH}' 和 '{YAWN_MODEL_TFLITE_INT8_PATH}' (或类似名称，如 best_int8.tflite)。")
```

**运行脚本：**
```bash
python export_tflite.py
```

### 3.3. 关于 `imgsz` 参数
`imgsz` 参数指定了模型输入图像的尺寸。
-   它应该是一个整数（例如 `imgsz=320`）或一个包含宽度和高度的元组（例如 `imgsz=(320, 256)`）。
-   为了在树莓派上获得最佳性能（速度），建议使用较小的图像尺寸，例如 `320` 或 `416`。您可能需要试验不同的尺寸，以在速度和精度之间找到最佳平衡。
-   这个尺寸应与您在树莓派上进行推理时提供给模型的图像预处理尺寸一致。

## 4. 输出文件

执行上述脚本后，您应该会在原始 `.pt` 文件所在的目录（即 `runs/detecteye/train/weights/` 和 `runs/detectyawn/train/weights/`）下找到相应的 `.tflite` 文件。Ultralytics 通常会将INT8量化后的模型命名为 `<original_name>_int8.tflite`，例如 `best_int8.tflite`。

这些 `_int8.tflite` 文件就是您部署到树莓派上的模型。

## 5. 后续步骤：在树莓派上部署

1.  **复制模型**：将生成的 `_int8.tflite` 模型文件复制到您的树莓派项目中。
2.  **安装 TFLite Runtime**：在树莓派上，您需要安装 TensorFlow Lite runtime 来加载和运行这些模型。
    ```bash
    # 根据您的树莓派 OS 版本和 Python 版本选择合适的 TFLite Runtime wheel
    # 访问: https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
    # 例如:
    pip install tflite-runtime
    # 或者特定版本的 wheel:
    # pip install https://dl.google.com/coral/python/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_armv7l.whl
    ```
3.  **修改推理代码**：
    *   在您的树莓派应用（特别是 `src/ai_logic.py`）中，修改加载模型的逻辑，使用 `tflite_runtime.interpreter.Interpreter` 来加载和运行 `.tflite` 模型。
    *   确保输入图像的预处理（调整大小、归一化等）与 TFLite 模型期望的输入格式一致（特别是 `imgsz`）。

这个转换和量化过程应该能显著减小模型大小并提高在树莓派上的推理速度。测试转换后的模型，看看单帧计算时间是否得到改善。
