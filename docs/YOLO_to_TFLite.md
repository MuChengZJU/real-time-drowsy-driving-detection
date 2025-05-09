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

INT8 量化通过将模型的浮点权重转换为8位整数来减小模型大小并加速推理。为了在量化过程中最大限度地减少精度损失，需要一个"校准数据集"。这个数据集应该包含一小部分（例如 50-100 张）来自您的**原始训练集或验证集**的代表性图像。这些图像应能覆盖模型在实际应用中可能遇到的各种场景。

**具体操作步骤：**

**A. 针对眼睛状态检测模型 (`runs/detecteye/train/weights/best.pt`)**

1.  **数据来源定位**：
    *   根据项目中的 `notebooks/LoadData.ipynb` 脚本，用于训练眼睛状态检测模型的图像数据（例如 `MichalMlodawski/closed-open-eyes` 数据集）在经过预处理后，被保存到了项目根目录下的 `datasets/images/train/` 和 `datasets/images/val/` 子目录中。
    *   这些目录包含了YOLO格式的图像，是您进行校准数据的直接来源。

2.  **挑选校准图像**：
    *   从 `datasets/images/train/` 或 `datasets/images/val/` 目录中，随机挑选 **50 到 100 张** 具有代表性的图像。
    *   确保这些图像能覆盖不同的光照条件、头部姿态、以及清晰的睁眼和闭眼状态。

3.  **组织校准图像并配置路径**：
    *   在您的项目根目录下创建一个新的文件夹，例如 `data/calibration_images_eye/`。
    *   将步骤2中挑选的图像复制到这个新创建的 `data/calibration_images_eye/` 文件夹中。
    *   在后续的 `export_tflite.py` 脚本（见 3.2 节）中，将 `EYE_CALIBRATION_YAML`变量的值设置为此目录的路径，例如：
        ```python
        EYE_CALIBRATION_YAML = 'data/calibration_images_eye/'
        ```
        或者，您也可以创建一个简单的 `eye_config.yaml` 文件（内容如下），然后将 `EYE_CALIBRATION_YAML` 设置为该YAML文件的路径。直接使用目录通常更简单。
        ```yaml
        # eye_config.yaml (示例)
        # path: ../data/calibration_images_eye  # 假设此yaml在项目根目录，图片在其子目录
        train: ./data/calibration_images_eye/  # Ultralytics 会查找此路径下的图像
        val: ./data/calibration_images_eye/
        nc: 2  # 类别数量，根据您的模型确定 (例如: 睁眼, 闭眼)
        names: ['open_eye', 'close_eye'] # 类别名称，根据您的模型确定
        ```

**B. 针对打哈欠检测模型 (`runs/detectyawn/train/weights/best.pt`)**

1.  **数据来源定位与准备**：
    *   根据项目 `README.md`，打哈欠检测模型使用的是Kaggle上的 "[Yawning Dataset](https://www.kaggle.com/datasets/deepankarvarma/yawning-dataset-classification?select=yawn)"。
    *   **您需要首先下载此数据集。**
    *   **关键步骤 - 数据预处理**：下载的原始数据集可能不是直接的YOLO训练格式（即独立的图像文件和对应的标签文件）。您需要按照**训练该打哈欠模型时所采用的相同预处理流程**来处理这些原始数据。这可能包括：
        *   从视频中提取帧（如果数据集包含视频）。
        *   将图像转换为合适的格式（如 `.jpg` 或 `.png`）。
        *   为每张图像创建YOLO格式的标签文件（`.txt`），其中包含类别和边界框信息。
        *   将处理后的图像和标签分别存放到特定的目录结构中，例如 `yawn_dataset_prepared/images/train/` 和 `yawn_dataset_prepared/labels/train/` （以及对应的 `val` 目录）。
        *   **请参考您项目中的 `notebooks/train.ipynb` 或用于处理打哈欠数据的相关脚本，以了解详细的数据准备步骤和最终的存储路径。** 校准数据必须与模型训练时看到的数据格式和内容特征一致。

2.  **挑选校准图像**：
    *   从您准备好的打哈欠模型训练图像目录（例如，前一步中创建的 `yawn_dataset_prepared/images/train/` 或 `yawn_dataset_prepared/images/val/`）中，随机挑选 **50 到 100 张** 具有代表性的图像。
    *   确保图像覆盖不同的打哈欠状态（张大嘴、未张嘴）、面部朝向和光照条件。

3.  **组织校准图像并配置路径**：
    *   在您的项目根目录下创建一个新的文件夹，例如 `data/calibration_images_yawn/`。
    *   将步骤2中挑选的打哈欠图像复制到这个新创建的 `data/calibration_images_yawn/` 文件夹中。
    *   在后续的 `export_tflite.py` 脚本（见 3.2 节）中，将 `YAWN_CALIBRATION_YAML` 变量的值设置为此目录的路径，例如：
        ```python
        YAWN_CALIBRATION_YAML = 'data/calibration_images_yawn/'
        ```
        同样，您也可以选择创建一个对应的 `yawn_config.yaml` 文件。

**通用提示**：
*   校准数据集不需要很大，但其包含的图像必须能够充分代表模型在实际推理中会遇到的数据分布。
*   确保校准图像与模型训练时使用的图像具有相同的预处理方式（例如，如果训练时对图像进行了特定的归一化或尺寸调整，理论上校准数据也应相似，但Ultralytics的导出工具通常能处理原始图像进行校准）。
*   `ultralytics` 的 `export` 函数在进行INT8量化时，`data` 参数可以直接指向包含校准图像的目录，或者指向一个描述数据集的YAML文件（如训练时使用的 `data.yaml`，或者您为校准专门创建的简单YAML）。直接使用图像目录通常是最直接的方法。

完成以上步骤后，您就拥有了进行INT8量化所需的校准数据。接下来的步骤是运行 `export_tflite.py` 脚本。

**创建数据YAML文件（旧版说明，可作为参考或替代方案，但直接使用图像目录更推荐）：**

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
