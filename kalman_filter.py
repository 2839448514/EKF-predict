import numpy as np


class ExtendedKalmanFilter:
    """
    扩展卡尔曼滤波器实现，用于跟踪和预测目标运动
    状态向量 [x, y, vx, vy, ax, ay]
    x, y - 位置
    vx, vy - 速度
    ax, ay - 加速度
    """
    def __init__(self, dt=0.1):
        # 状态向量维度 [x, y, vx, vy, ax, ay]
        self.state_dim = 6
        # 测量向量维度 [x, y]
        self.meas_dim = 2
        
        # 初始化状态向量和协方差矩阵
        self.x = np.zeros((self.state_dim, 1))
        self.P = np.eye(self.state_dim) * 100  # 初始不确定性较大
        
        # 过程噪声协方差矩阵
        self.Q = np.eye(self.state_dim)
        self.Q[0:2, 0:2] *= 0.01  # 位置噪声小
        self.Q[2:4, 2:4] *= 0.1   # 速度噪声中等
        self.Q[4:6, 4:6] *= 1.0   # 加速度噪声大
        
        # 测量噪声协方差矩阵
        self.R = np.eye(self.meas_dim) * 10  # 假设测量噪声适中
        
        # 状态转移矩阵 F (根据运动学方程)
        self.dt = dt
        self.F = np.eye(self.state_dim)
        self.F[0, 2] = self.dt
        self.F[1, 3] = self.dt
        self.F[2, 4] = self.dt
        self.F[3, 5] = self.dt
        self.F[0, 4] = 0.5 * self.dt * self.dt
        self.F[1, 5] = 0.5 * self.dt * self.dt
        
        # 测量矩阵 H
        self.H = np.zeros((self.meas_dim, self.state_dim))
        self.H[0, 0] = 1.0  # x位置
        self.H[1, 1] = 1.0  # y位置
        
        # 初始化
        self.initialized = False
        self.history = []
    
    def initialize(self, measurement):
        """
        使用第一次测量初始化状态
        """
        self.x[0] = measurement[0]  # x位置
        self.x[1] = measurement[1]  # y位置
        # 其他状态保持为零
        
        self.initialized = True
        self.history = [np.copy(self.x[:2])]
    
    def predict(self):
        """
        预测步骤：预测下一个状态
        """
        # 使用状态转移矩阵预测新状态
        self.x = np.dot(self.F, self.x)
        
        # 更新协方差
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return np.copy(self.x)
    
    def update(self, measurement):
        """
        更新步骤：根据测量更新状态估计
        """
        if not self.initialized:
            self.initialize(measurement)
            return np.copy(self.x)
        
        # 计算残差
        z = np.array(measurement).reshape(self.meas_dim, 1)
        y = z - np.dot(self.H, self.x)
        
        # 计算卡尔曼增益
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # 更新状态估计
        self.x = self.x + np.dot(K, y)
        
        # 更新协方差矩阵
        I = np.eye(self.state_dim)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        
        # 保存历史状态用于绘图
        self.history.append(np.copy(self.x[:2]))
        
        return np.copy(self.x)
    
    def get_history(self):
        """
        返回历史轨迹点
        """
        return np.array(self.history)
    
    def predict_future(self, steps=10):
        """
        预测未来几步的位置
        """
        future_states = []
        current_x = np.copy(self.x)
        current_P = np.copy(self.P)
        
        for _ in range(steps):
            # 使用当前模型预测未来状态
            current_x = np.dot(self.F, current_x)
            future_states.append(np.copy(current_x[:2]))
        
        return np.array(future_states)
