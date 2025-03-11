import numpy as np
import cv2
from scipy.spatial import distance as dist
from collections import OrderedDict

class KalmanTracker:
    def __init__(self, centroid, dt=0.1):
        """初始化卡尔曼滤波器跟踪器
        
        Args:
            centroid: 目标的初始中心点(x,y)
            dt: 时间步长
        """
        # 4个状态变量(x, y, vx, vy)和2个测量变量(x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # 状态转移矩阵
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 测量矩阵 - 只测量位置，不直接测量速度
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # 过程噪声协方差矩阵
        # 调整这些值以适应目标的运动模式
        self.kf.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32) * 0.03
        
        # 测量噪声协方差矩阵
        self.kf.measurementNoiseCov = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.float32) * 0.1
        
        # 后验误差估计协方差矩阵
        self.kf.errorCovPost = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32) * 0.1
        
        # 初始状态
        self.kf.statePost = np.array([
            [centroid[0]],
            [centroid[1]],
            [0],
            [0]
        ], dtype=np.float32)
        
        self.prediction = None
        self.dt = dt
        self.history = []  # 存储历史位置
        self.max_history = 20  # 历史轨迹长度
        self.history.append((int(centroid[0]), int(centroid[1])))
        self.disappeared = 0
        self.prediction_quality = 1.0  # 预测质量评分(0-1)
        self.velocity_stability = 0.0  # 速度稳定性

    def predict(self):
        """预测下一个位置"""
        prediction = self.kf.predict()
        self.prediction = (int(prediction[0][0]), int(prediction[1][0]))
        return self.prediction

    def update(self, centroid):
        """用新的测量结果更新卡尔曼滤波器"""
        measurement = np.array([[centroid[0]], [centroid[1]]], dtype=np.float32)
        
        # 记录之前的状态
        prev_state = self.kf.statePost.copy()
        
        # 更新卡尔曼滤波器
        self.kf.correct(measurement)
        
        # 计算速度的变化幅度，用于评估预测质量
        if prev_state[2][0] != 0 or prev_state[3][0] != 0:
            old_vel = np.array([prev_state[2][0], prev_state[3][0]])
            new_vel = np.array([self.kf.statePost[2][0], self.kf.statePost[3][0]])
            vel_mag_old = np.linalg.norm(old_vel)
            vel_mag_new = np.linalg.norm(new_vel)
            
            if vel_mag_old > 0 and vel_mag_new > 0:
                # 计算速度方向变化
                vel_cosine = np.dot(old_vel, new_vel) / (vel_mag_old * vel_mag_new)
                vel_cosine = np.clip(vel_cosine, -1, 1)  # 确保值在[-1, 1]范围内
                
                # 速度稳定度 (1表示完全稳定，0表示不稳定)
                self.velocity_stability = (vel_cosine + 1) / 2
                
                # 速度幅度变化
                vel_mag_ratio = min(vel_mag_old, vel_mag_new) / max(vel_mag_old, vel_mag_new)
                
                # 更新预测质量 (结合方向稳定性和幅度稳定性)
                self.prediction_quality = 0.7 * self.velocity_stability + 0.3 * vel_mag_ratio
                # 历史加权平均，新的预测质量占30%，历史占70%
                self.prediction_quality = 0.3 * self.prediction_quality + 0.7 * self.prediction_quality
        
        # 重置消失计数
        self.disappeared = 0
        
        # 更新历史轨迹
        current_pos = (int(centroid[0]), int(centroid[1]))
        self.history.append(current_pos)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        return current_pos

    def predict_future(self, steps=10):
        """预测未来多步的位置"""
        # 复制当前卡尔曼滤波器状态进行预测
        kf_copy = cv2.KalmanFilter(4, 2)
        kf_copy.statePost = self.kf.statePost.copy()
        kf_copy.errorCovPost = self.kf.errorCovPost.copy()
        kf_copy.transitionMatrix = self.kf.transitionMatrix.copy()
        
        future_positions = []
        for _ in range(steps):
            prediction = kf_copy.predict()
            x, y = int(prediction[0][0]), int(prediction[1][0])
            future_positions.append((x, y))
            
        return future_positions


class ObjectTracker:
    def __init__(self, max_disappeared=10, max_distance=50, dt=0.1, immediate_destroy=True):
        """初始化跟踪器
        
        Args:
            max_disappeared: 在目标被视为消失之前，最大连续帧数
            max_distance: 分配目标的最大距离阈值
            dt: 时间步长
            immediate_destroy: 如果为True，当物体未被检测到时立即销毁跟踪器
        """
        self.nextObjectID = 0
        self.objects = OrderedDict()  # 存储活跃的跟踪器
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.dt = dt
        self.colors = {}  # 为每个ID分配一个颜色
        self.immediate_destroy = immediate_destroy  # 新增参数，控制是否立即销毁
        
    def register(self, centroid):
        """注册一个新的目标"""
        tracker = KalmanTracker(centroid, dt=self.dt)
        self.objects[self.nextObjectID] = tracker
        self.colors[self.nextObjectID] = tuple([np.random.randint(0, 255) for _ in range(3)])
        self.nextObjectID += 1
        
    def deregister(self, objectID):
        """注销一个消失的目标"""
        del self.objects[objectID]
        del self.colors[objectID]
        
    def update(self, centroids):
        """更新跟踪器
        
        Args:
            centroids: 当前帧中检测到的所有对象的中心点列表
        """
        # 如果没有中心点，将所有现有对象标记为消失
        if len(centroids) == 0:
            # 如果设置为立即销毁，则所有跟踪器都被销毁
            if self.immediate_destroy:
                object_ids = list(self.objects.keys())
                for object_id in object_ids:
                    self.deregister(object_id)
            else:
                # 否则使用原有的消失计数机制
                for objectID in list(self.objects.keys()):
                    self.objects[objectID].disappeared += 1
                    
                    # 如果对象消失太久，注销它
                    if self.objects[objectID].disappeared > self.max_disappeared:
                        self.deregister(objectID)
                    
            return self.objects
        
        # 如果当前没有跟踪任何对象，注册每个中心点
        if len(self.objects) == 0:
            for i in range(len(centroids)):
                self.register(centroids[i])
        
        # 否则，尝试匹配中心点与现有对象
        else:
            objectIDs = list(self.objects.keys())
            objectTrackers = list(self.objects.values())
            
            # 首先预测所有现有对象的新位置
            predicted_centroids = [tracker.predict() for tracker in objectTrackers]
            
            # 计算所有预测中心点和新检测到中心点之间的距离
            D = dist.cdist(predicted_centroids, centroids)
            
            # 找到行(现有对象)和列(新中心点)之间的最小距离匹配
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            usedRows = set()
            usedCols = set()
            
            # 遍历行列配对
            for (row, col) in zip(rows, cols):
                # 如果已经检查过这一行或这一列，继续
                if row in usedRows or col in usedCols:
                    continue
                
                # 如果距离大于最大阈值，继续
                if D[row, col] > self.max_distance:
                    continue
                
                # 更新跟踪器
                objectID = objectIDs[row]
                self.objects[objectID].update(centroids[col])
                
                # 标记为已使用
                usedRows.add(row)
                usedCols.add(col)
                
            # 检查哪些行和列还未被使用
            unusedRows = set(range(D.shape[0])) - usedRows
            unusedCols = set(range(D.shape[1])) - usedCols
            
            # 处理未匹配的追踪器 (这些物体在当前帧未被检测到)
            if self.immediate_destroy:
                # 如果设置为立即销毁，则未匹配到的跟踪器全部销毁
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.deregister(objectID)
            else:
                # 否则使用原有的消失计数机制
                if len(self.objects) >= len(centroids):
                    for row in unusedRows:
                        objectID = objectIDs[row]
                        self.objects[objectID].disappeared += 1
                        
                        if self.objects[objectID].disappeared > self.max_disappeared:
                            self.deregister(objectID)
                            
            # 注册尚未使用的新中心点（这些是新出现的物体）
            for col in unusedCols:
                self.register(centroids[col])
                    
        return self.objects
    
    def draw_tracks(self, frame, future_steps=10):
        """在帧上绘制轨迹和预测
        
        Args:
            frame: 要绘制的图像
            future_steps: 预测的未来步数
            
        Returns:
            带有轨迹的图像
        """
        for objectID, tracker in self.objects.items():
            # 如果物体已经消失，不绘制它
            if tracker.disappeared > 0:
                continue
                
            # 使用固定颜色：蓝色用于历史轨迹，红色用于预测
            history_color = (255, 0, 0)  # 蓝色 (BGR格式)
            prediction_color = (0, 0, 255)  # 红色 (BGR格式)
            
            # 获取对象的原始颜色用于ID标识
            original_color = self.colors[objectID]
            
            # 绘制ID和当前位置，使用原始标识颜色
            if tracker.history:
                current_position = tracker.history[-1]
                cv2.circle(frame, current_position, 5, original_color, -1)
                cv2.putText(frame, f"ID {objectID}", 
                          (current_position[0] - 10, current_position[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, original_color, 2)
            
            # 绘制历史轨迹 - 使用蓝色实线
            for i in range(1, len(tracker.history)):
                pt1 = tracker.history[i-1]
                pt2 = tracker.history[i]
                cv2.line(frame, pt1, pt2, history_color, 2)
            
            # 绘制预测的未来轨迹 - 使用红色
            future_positions = tracker.predict_future(future_steps)
            
            # 根据预测质量调整预测线的样式
            line_thickness = max(1, min(3, int(2 * tracker.prediction_quality)))
            
            # 从当前位置到第一个预测位置的红色虚线
            if tracker.history and future_positions:
                pt1 = tracker.history[-1]
                pt2 = future_positions[0]
                self.draw_dashed_line(frame, pt1, pt2, prediction_color, thickness=line_thickness)
            
            # 绘制预测的未来轨迹 - 使用红色虚线
            for i in range(1, len(future_positions)):
                pt1 = future_positions[i-1]
                pt2 = future_positions[i]
                
                # 根据预测质量调整虚线密度
                if tracker.prediction_quality < 0.5:
                    # 预测质量低，使用更稀疏的虚线
                    self.draw_dashed_line(frame, pt1, pt2, prediction_color, 
                                         thickness=line_thickness, gap=6)
                else:
                    # 预测质量高，使用较密的虚线
                    self.draw_dashed_line(frame, pt1, pt2, prediction_color, 
                                         thickness=line_thickness, gap=4)
                
                # 每个预测点画一个小圆点
                cv2.circle(frame, pt2, 3, prediction_color, -1)
                
        return frame
        
    def draw_dashed_line(self, img, pt1, pt2, color, thickness=1, gap=5):
        """在两点之间绘制虚线
        
        Args:
            img: 图像
            pt1, pt2: 起点和终点
            color: 线条颜色
            thickness: 线条厚度
            gap: 虚线间隔
        """
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        pts = []
        
        # 计算线段两端之间的单位向量
        if dist == 0:
            return  # 防止除以零错误
            
        dx, dy = (pt2[0] - pt1[0]) / dist, (pt2[1] - pt1[1]) / dist
        
        # 计算线段上的所有点
        step = gap
        for i in range(0, int(dist), step):
            p = (int(pt1[0] + dx * i), int(pt1[1] + dy * i))
            pts.append(p)
            
        # 每隔一个点绘制一条短线段
        for i in range(len(pts) - 1):
            if i % 2 == 0:
                cv2.line(img, pts[i], pts[i + 1], color, thickness)
