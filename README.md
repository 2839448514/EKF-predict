# EKF-predict

扩展卡尔曼滤波加YOLO预测物体移动轨迹

## 项目介绍

本项目使用扩展卡尔曼滤波器（EKF）和YOLO模型进行物体检测和轨迹预测。项目包含一个图形用户界面（GUI），由`app.py`文件中的`ObjectDetectionApp`类实现。

## 安装步骤

1. 克隆此仓库到本地：
    ```bash
    git clone https://github.com/2839448514/EKF-predict.git
    cd EKF-predict
    ```

2. 创建并激活虚拟环境（可选）：
    ```bash
    python -m venv venv
    source venv/bin/activate  # 对于Windows用户，使用 `venv\Scripts\activate`
    ```

3. 安装依赖项：
    ```bash
    pip install -r requirements.txt
    ```

## 使用说明

1. 运行项目：
    ```bash
    python app.py
    ```

2. 启动后，您将看到一个图形用户界面（GUI），可以选择模型、输入源（摄像头或视频文件）以及检测设置。

3. 选择模型并加载，点击“使用摄像头”或“打开视频文件”按钮开始检测。

4. 在检测过程中，您可以调整预测步数、历史轨迹长度等设置。

5. 如果启用了保存视频功能，可以选择保存位置，检测结果将保存为视频文件。

## 示例代码

以下是一个简单的示例代码，展示如何使用项目中的主要功能：

```python
import sys
from PyQt6.QtWidgets import QApplication
from app import ObjectDetectionApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec())
```

## 文件说明

- `app.py`：包含图形用户界面（GUI）和主要功能实现。
- `kalman_filter.py`：扩展卡尔曼滤波器实现，用于跟踪和预测目标运动。
- `main.py`：检查CUDA是否可用。
- `model_utils.py`：处理模型路径和加载的工具函数。
- `object_tracker.py`：物体跟踪器实现，使用卡尔曼滤波器进行跟踪和预测。
- `requirements.txt`：列出项目所需的依赖项。

## 贡献

欢迎提交问题（Issues）和拉取请求（Pull Requests）来改进本项目。

## 许可证

本项目使用MIT许可证，详情请参阅`LICENSE`文件。
