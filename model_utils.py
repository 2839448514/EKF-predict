"""
模型工具模块 - 处理模型路径和加载
"""
import os
import sys

def get_model_path(filename):
    """获取模型文件的路径，处理打包和非打包环境
    
    Args:
        filename: 模型文件名
        
    Returns:
        str: 模型的完整路径
    """
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
    """获取可用的模型列表
    
    Returns:
        list: 可用模型的文件路径列表
    """
    # 返回固定的四个模型选项
    return ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]

def get_model_name(model_path):
    """从模型路径获取模型名称
    
    Args:
        model_path: 模型的路径
        
    Returns:
        str: 模型名称
    """
    return os.path.basename(model_path)
