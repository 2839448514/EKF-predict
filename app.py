import sys
import os
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QComboBox, QSpinBox, QCheckBox, QFrame, QStatusBar,
                            QGroupBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon, QFont
from ultralytics import YOLO
from object_tracker import ObjectTracker

# 从model_utils导入函数，如果文件不存在，则添加内联函数
try:
    from model_utils import get_model_path, get_available_models, get_model_name
except ImportError:
    # 内联定义模型工具函数
    def get_model_path(filename):
        """获取模型文件的路径，处理打包和非打包环境"""
        if getattr(sys, 'frozen', False):  # 检查是否为打包后的环境
            # 打包后环境，使用_MEIPASS临时目录
            base_path = sys._MEIPASS
        else:
            # 开发环境，使用当前脚本目录
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        # 首先检查模型目录
        model_dir = os.path.join(base_path, "models")
        model_path = os.path.join(model_dir, filename)
        
        # 如果模型目录中没有找到，则检查根目录
        if not os.path.exists(model_path):
            model_path = os.path.join(base_path, filename)
        
        return model_path

    def get_available_models():
        """获取可用的模型列表"""
        # 固定返回四个模型名称
        return ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
    
    def get_model_name(model_path):
        """从模型路径获取模型名称"""
        return os.path.basename(model_path)


class StyleSheet:
    MAIN_STYLE = """
        QMainWindow {
            background-color: #2b2b2b;
        }
        QLabel {
            color: #ffffff;
            font-size: 14px;
        }
        QPushButton {
            background-color: #0d47a1;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 14px;
            min-width: 100px;
        }
        QPushButton:hover {
            background-color: #1565c0;
        }
        QPushButton:pressed {
            background-color: #0a3d91;
        }
        QPushButton:disabled {
            background-color: #666666;
        }
        QComboBox {
            background-color: #424242;
            color: white;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 5px;
            min-width: 100px;
        }
        QComboBox:drop-down {
            border: none;
            width: 20px;
        }
        QSpinBox {
            background-color: #424242;
            color: white;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 5px;
        }
        QCheckBox {
            color: white;
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
        QGroupBox {
            color: white;
            font-weight: bold;
            border: 2px solid #555555;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
    """

class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人工智能物体检测与跟踪")  # 修改窗口标题
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(StyleSheet.MAIN_STYLE)

        # Initialize variables
        self.model = None
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.tracker = None
        self.is_playing = False
        self.available_models = get_available_models()  # 获取可用的模型列表

        # Video writer properties
        self.video_writer = None
        self.output_video_path = None  # 存储输出视频路径

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Create left panel for video display
        video_container = QFrame()
        video_container.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        video_container.setStyleSheet("QFrame { background-color: #1a1a1a; border: 2px solid #555555; border-radius: 8px; }")
        video_layout = QVBoxLayout(video_container)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(900, 700)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: none;")
        video_layout.addWidget(self.video_label)
        main_layout.addWidget(video_container, stretch=7)

        # Create right control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(15)
        main_layout.addWidget(control_panel, stretch=3)

        # Model Selection Group
        model_group = QGroupBox("模型设置")  # 修改分组标题
        model_layout = QVBoxLayout()
        
        model_label = QLabel("选择模型:")  # 修改标签文本
        self.model_combo = QComboBox()
        
        # 添加固定的四个模型选项
        for model_name in self.available_models:
            self.model_combo.addItem(model_name)
            
        self.model_combo.currentIndexChanged.connect(self.load_model)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        control_layout.addWidget(model_group)

        # Input Source Group
        source_group = QGroupBox("输入源")  # 修改分组标题
        source_layout = QVBoxLayout()
        
        self.camera_btn = QPushButton("📷 使用摄像头")  # 修改按钮文本
        self.camera_btn.clicked.connect(self.start_camera)
        
        self.file_btn = QPushButton("📁 打开视频文件")  # 修改按钮文本
        self.file_btn.clicked.connect(self.open_video_file)
        
        source_layout.addWidget(self.camera_btn)
        source_layout.addWidget(self.file_btn)
        source_group.setLayout(source_layout)
        control_layout.addWidget(source_group)

        # Detection Settings Group
        settings_group = QGroupBox("检测设置")  # 修改分组标题
        settings_layout = QVBoxLayout()
        
        steps_label = QLabel("预测步数:")  # 修改标签文本
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 30)
        self.steps_spin.setValue(15)
        
        # 修改保存视频选项，添加保存路径按钮
        save_layout = QHBoxLayout()
        self.save_check = QCheckBox("保存输出视频")
        self.save_check.stateChanged.connect(self.toggle_video_saving)
        self.save_path_btn = QPushButton("选择保存位置")
        self.save_path_btn.clicked.connect(self.select_output_path)
        save_layout.addWidget(self.save_check)
        save_layout.addWidget(self.save_path_btn)
        
        # 添加轨迹显示控制
        track_layout = QHBoxLayout()
        track_label = QLabel("历史轨迹长度:")
        self.track_length_spin = QSpinBox()
        self.track_length_spin.setRange(5, 50)
        self.track_length_spin.setValue(20)
        self.track_length_spin.valueChanged.connect(self.update_track_settings)
        track_layout.addWidget(track_label)
        track_layout.addWidget(self.track_length_spin)
        
        # 添加跟踪器销毁方式选项
        self.immediate_destroy_check = QCheckBox("物体消失时立即销毁跟踪")
        self.immediate_destroy_check.setChecked(True)  # 默认选中
        self.immediate_destroy_check.stateChanged.connect(self.update_tracking_mode)
        
        # 添加颜色图例说明
        legend_layout = QHBoxLayout()
        legend_label = QLabel("颜色图例:")
        history_color_label = QLabel("□ 历史轨迹")
        history_color_label.setStyleSheet("color: blue;")
        prediction_color_label = QLabel("□ 预测路径")
        prediction_color_label.setStyleSheet("color: red;")
        legend_layout.addWidget(legend_label)
        legend_layout.addWidget(history_color_label)
        legend_layout.addWidget(prediction_color_label)
        
        settings_layout.addWidget(steps_label)
        settings_layout.addWidget(self.steps_spin)
        settings_layout.addLayout(track_layout)
        settings_layout.addWidget(self.immediate_destroy_check)  # 添加到界面
        settings_layout.addLayout(legend_layout)
        settings_layout.addLayout(save_layout)  # 替换 settings_layout.addWidget(self.save_check)
        settings_group.setLayout(settings_layout)
        control_layout.addWidget(settings_group)

        # Playback Controls Group
        playback_group = QGroupBox("播放控制")  # 修改分组标题
        playback_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ 播放")  # 修改按钮文本
        self.play_btn.clicked.connect(self.toggle_playback)
        self.stop_btn = QPushButton("⏹ 停止")  # 修改按钮文本
        self.stop_btn.clicked.connect(self.stop_playback)
        
        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.stop_btn)
        playback_group.setLayout(playback_layout)
        control_layout.addWidget(playback_group)

        # Status Information
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        control_layout.addWidget(self.status_label)

        # 状态栏添加保存路径显示
        self.output_path_label = QLabel()
        self.output_path_label.setStyleSheet("color: #ffcc00;")
        control_layout.addWidget(self.output_path_label)

        # Add stretch to push controls to the top
        control_layout.addStretch()

        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.setStyleSheet("QStatusBar { color: white; }")

        # Initialize model
        self.load_model()
        self.update_status("准备就绪")  # 修改状态信息

    def update_status(self, message):
        self.status_label.setText(message)
        self.statusBar.showMessage(message)

    def load_model(self):
        try:
            model_index = self.model_combo.currentIndex()
            if (model_index < 0 or model_index >= len(self.available_models)):
                self.update_status("错误: 无效的模型选择")  # 修改错误信息
                return
                
            model_path = self.available_models[model_index]
            
            self.update_status(f"正在加载模型: {model_path}...")
            self.model = YOLO(model_path)
            self.update_status(f"模型 {model_path} 加载成功")
        except Exception as e:
            self.update_status(f"加载模型时出错: {str(e)}")

    def start_camera(self):
        if self.video_capture is not None:
            self.video_capture.release()
        self.video_capture = cv2.VideoCapture(0)
        if self.video_capture.isOpened():
            # 创建跟踪器，启用立即销毁选项
            self.tracker = ObjectTracker(max_disappeared=20, max_distance=100, dt=0.1, immediate_destroy=True)
            self.start_playback()
            self.update_status("摄像头已启动")  # 修改状态信息
        else:
            self.update_status("错误: 无法打开摄像头")  # 修改错误信息

    def open_video_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "打开视频文件",  # 修改对话框标题
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"  # 修改文件过滤器
        )
        if file_name:
            if self.video_capture is not None:
                self.video_capture.release()
            self.video_capture = cv2.VideoCapture(file_name)
            if self.video_capture.isOpened():
                # 创建跟踪器，启用立即销毁选项
                self.tracker = ObjectTracker(max_disappeared=20, max_distance=100, dt=0.1, immediate_destroy=True)
                self.start_playback()
                self.update_status(f"正在播放: {file_name}")  # 修改状态信息
            else:
                self.update_status("错误: 无法打开视频文件")  # 修改错误信息

    def toggle_playback(self):
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        self.is_playing = True
        self.play_btn.setText("⏸ 暂停")  # 修改按钮文本
        self.timer.start(30)
        
        # 如果启用了保存视频，设置视频写入器
        if self.save_check.isChecked():
            self.setup_video_writer()
            
        self.update_status("正在播放")  # 修改状态信息

    def pause_playback(self):
        self.is_playing = False
        self.play_btn.setText("▶ 播放")  # 修改按钮文本
        self.timer.stop()
        self.update_status("已暂停")  # 修改状态信息

    def stop_playback(self):
        self.pause_playback()
        
        # 释放视频写入器
        self.release_video_writer()
        
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.video_label.clear()
        self.update_status("已停止")  # 修改状态信息

    def update_frame(self):
        if self.video_capture is None or not self.video_capture.isOpened():
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.stop_playback()
            return

        # Process frame with YOLO and tracker
        results = self.model(frame, conf=0.5)
        
        # Get centroids of detected objects
        centroids = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                centroids.append((centroid_x, centroid_y))
                
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{self.model.names[cls]} {conf:.2f}"
                
                # Enhanced visualization
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add background to text for better visibility
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Update tracker and draw tracks
        self.tracker.update(centroids)
        processed_frame = self.tracker.draw_tracks(frame.copy(), future_steps=self.steps_spin.value())
        
        # 如果启用了视频保存，写入帧
        if self.video_writer is not None and self.save_check.isChecked():
            self.video_writer.write(processed_frame)
        
        # Convert frame to Qt format and display
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def update_track_settings(self):
        """更新轨迹显示设置"""
        if self.tracker:
            for tracker in self.tracker.objects.values():
                tracker.max_history = self.track_length_spin.value()
            self.update_status("轨迹设置已更新")
            
    def update_tracking_mode(self):
        """更新跟踪销毁模式"""
        if self.tracker:
            self.tracker.immediate_destroy = self.immediate_destroy_check.isChecked()
            mode = "立即销毁" if self.tracker.immediate_destroy else "渐进销毁"
            self.update_status(f"已更新跟踪模式: {mode}")

    def select_output_path(self):
        """选择视频输出保存路径"""
        default_name = f"output_{self.get_timestamp()}.mp4"
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "选择保存位置",
            default_name,
            "MP4 文件 (*.mp4);;AVI 文件 (*.avi)"
        )
        if file_name:
            self.output_video_path = file_name
            self.output_path_label.setText(f"输出: {os.path.basename(file_name)}")
            
            # 如果当前正在录制，重新创建写入器
            if self.save_check.isChecked() and self.video_capture is not None:
                self.setup_video_writer()
                
    def get_timestamp(self):
        """获取当前时间戳，用于默认文件名"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
            
    def toggle_video_saving(self, state):
        """开关视频保存功能"""
        if state == Qt.CheckState.Checked:
            if not self.output_video_path:
                # 如果没有设置输出路径，自动选择一个
                default_path = os.path.join(
                    os.path.expanduser("~"), 
                    "Videos" if os.path.exists(os.path.join(os.path.expanduser("~"), "Videos")) else "",
                    f"output_{self.get_timestamp()}.mp4"
                )
                self.output_video_path = default_path
                self.output_path_label.setText(f"输出: {os.path.basename(default_path)}")
                
            if self.video_capture is not None and self.is_playing:
                self.setup_video_writer()
                self.update_status("视频保存已开始")
        else:
            self.release_video_writer()
            self.update_status("视频保存已停止")
            
    def setup_video_writer(self):
        """设置视频写入器"""
        if self.video_capture is None or not self.video_capture.isOpened():
            return
            
        # 释放之前的视频写入器
        self.release_video_writer()
        
        # 获取视频源的属性
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # 默认帧率
            
        # 创建目录(如果不存在)
        output_dir = os.path.dirname(self.output_video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码
        self.video_writer = cv2.VideoWriter(
            self.output_video_path, fourcc, fps, (width, height)
        )
        
    def release_video_writer(self):
        """释放视频写入器"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def closeEvent(self, event):
        # 释放视频写入器
        self.release_video_writer()
        self.stop_playback()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better cross-platform appearance
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec())