"""
HOG特征角点检测实现
Histogram of Oriented Gradients (HOG) 用于角点检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

class HOGCornerDetector:
    def __init__(self, cell_size=8, block_size=2, nbins=9):
        """
        初始化HOG角点检测器
        
        参数:
            cell_size: 单元格大小（像素）
            block_size: 块大小（单元格数）
            nbins: 方向直方图的bin数量
        """
        self.cell_size = cell_size
        self.block_size = block_size
        self.nbins = nbins
        
    def compute_gradients(self, image):
        """计算图像的梯度"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 使用Sobel算子计算梯度
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值和方向
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi
        angle[angle < 0] += 180  # 转换为0-180度
        
        return magnitude, angle
    
    def compute_hog_features(self, magnitude, angle):
        """计算HOG特征"""
        h, w = magnitude.shape
        
        # 计算单元格数量
        cell_h = h // self.cell_size
        cell_w = w // self.cell_size
        
        # 初始化HOG特征矩阵
        hog_features = np.zeros((cell_h, cell_w, self.nbins))
        
        # 为每个单元格计算方向直方图
        for i in range(cell_h):
            for j in range(cell_w):
                # 提取单元格区域
                cell_mag = magnitude[i*self.cell_size:(i+1)*self.cell_size,
                                    j*self.cell_size:(j+1)*self.cell_size]
                cell_angle = angle[i*self.cell_size:(i+1)*self.cell_size,
                                  j*self.cell_size:(j+1)*self.cell_size]
                
                # 计算方向直方图
                hist, _ = np.histogram(cell_angle, bins=self.nbins, 
                                      range=(0, 180), weights=cell_mag)
                hog_features[i, j] = hist
        
        # 块归一化
        normalized_features = []
        for i in range(cell_h - self.block_size + 1):
            for j in range(cell_w - self.block_size + 1):
                # 提取块
                block = hog_features[i:i+self.block_size, j:j+self.block_size]
                block_flat = block.flatten()
                
                # L2归一化
                norm = np.linalg.norm(block_flat)
                if norm > 0:
                    block_flat = block_flat / norm
                
                normalized_features.append(block_flat)
        
        return np.concatenate(normalized_features)
    
    def detect_corners(self, image, threshold=0.1):
        """
        使用HOG特征检测角点
        
        参数:
            image: 输入图像
            threshold: 角点响应阈值
            
        返回:
            corners: 检测到的角点坐标列表
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 计算梯度
        magnitude, angle = self.compute_gradients(gray)
        
        # 计算角点响应
        h, w = gray.shape
        response = np.zeros((h, w))
        
        # 使用梯度幅值的变化作为角点响应
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算Hessian矩阵的特征值
        gxx = gx * gx
        gxy = gx * gy
        gyy = gy * gy
        
        # 使用Harris角点检测的变体
        k = 0.04
        det = gxx * gyy - gxy**2
        trace = gxx + gyy
        response = det - k * trace**2
        
        # 非极大值抑制
        corners = []
        for i in range(1, h-1):
            for j in range(1, w-1):
                if response[i, j] > threshold * response.max():
                    # 检查是否为局部极大值
                    if (response[i, j] >= response[i-1:i+2, j-1:j+2]).all():
                        corners.append((j, i, response[i, j]))
        
        # 按响应强度排序
        corners.sort(key=lambda x: x[2], reverse=True)
        
        return [(x, y) for x, y, _ in corners]
    
    def visualize_corners(self, image, corners, max_corners=100):
        """可视化检测到的角点"""
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # 绘制角点
        for i, (x, y) in enumerate(corners[:max_corners]):
            cv2.circle(vis_image, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.circle(vis_image, (int(x), int(y)), 5, (0, 255, 0), 1)
        
        return vis_image

def main():
    """主函数：测试HOG角点检测"""
    # 创建测试图像
    test_image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(test_image, (50, 50), (150, 150), 255, -1)
    cv2.circle(test_image, (100, 100), 30, 128, -1)
    
    # 创建检测器
    detector = HOGCornerDetector(cell_size=8, block_size=2, nbins=9)
    
    # 检测角点
    corners = detector.detect_corners(test_image, threshold=0.01)
    
    # 可视化结果
    result = detector.visualize_corners(test_image, corners)
    
    # 显示结果
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(test_image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    
    # 计算并显示梯度
    magnitude, angle = detector.compute_gradients(test_image)
    plt.subplot(132)
    plt.imshow(magnitude, cmap='hot')
    plt.title('梯度幅值')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(result)
    plt.title(f'检测到的角点 ({len(corners)}个)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/hog_corner_detection.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"检测到 {len(corners)} 个角点")
    print("前10个角点坐标:")
    for i, (x, y) in enumerate(corners[:10]):
        print(f"  角点 {i+1}: ({x:.1f}, {y:.1f})")

if __name__ == "__main__":
    main()