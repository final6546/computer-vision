"""
边缘检测算子比较
比较不同的边缘检测算子及其效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import time

class EdgeDetector:
    def __init__(self):
        """初始化边缘检测器"""
        pass
    
    def sobel_edge_detection(self, image, ksize=3):
        """Sobel边缘检测"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Sobel算子
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # 计算梯度幅值
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 归一化到0-255
        magnitude = np.uint8(255 * magnitude / magnitude.max())
        
        # 计算梯度方向
        direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        
        return magnitude, direction, sobel_x, sobel_y
    
    def prewitt_edge_detection(self, image):
        """Prewitt边缘检测"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Prewitt算子
        kernel_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
        
        kernel_y = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]])
        
        prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        
        # 计算梯度幅值
        magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
        magnitude = np.uint8(255 * magnitude / magnitude.max())
        
        # 计算梯度方向
        direction = np.arctan2(prewitt_y, prewitt_x) * 180 / np.pi
        
        return magnitude, direction, prewitt_x, prewitt_y
    
    def roberts_edge_detection(self, image):
        """Roberts边缘检测"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Roberts算子
        kernel_x = np.array([[1, 0],
                            [0, -1]])
        
        kernel_y = np.array([[0, 1],
                            [-1, 0]])
        
        roberts_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        roberts_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        
        # 计算梯度幅值
        magnitude = np.sqrt(roberts_x**2 + roberts_y**2)
        magnitude = np.uint8(255 * magnitude / magnitude.max())
        
        # 计算梯度方向
        direction = np.arctan2(roberts_y, roberts_x) * 180 / np.pi
        
        return magnitude, direction, roberts_x, roberts_y
    
    def canny_edge_detection(self, image, low_threshold=50, high_threshold=150):
        """Canny边缘检测"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        return edges
    
    def laplacian_edge_detection(self, image, ksize=3):
        """Laplacian边缘检测"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # Laplacian算子
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize)
        
        # 取绝对值并归一化
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(255 * laplacian / laplacian.max())
        
        return laplacian
    
    def zero_crossing_edge_detection(self, image, threshold=0.01):
        """零交叉边缘检测"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 应用LoG (Laplacian of Gaussian)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        
        # 零交叉检测
        edges = np.zeros_like(laplacian, dtype=np.uint8)
        h, w = laplacian.shape
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                # 检查3x3邻域中的零交叉
                neighborhood = laplacian[i-1:i+2, j-1:j+2]
                max_val = neighborhood.max()
                min_val = neighborhood.min()
                
                # 如果最大值和最小值符号相反且差值超过阈值
                if max_val > threshold and min_val < -threshold:
                    edges[i, j] = 255
        
        return edges
    
    def evaluate_edge_detection(self, edge_image, ground_truth=None):
        """评估边缘检测结果"""
        if ground_truth is None:
            # 如果没有真实标签，使用Canny作为参考
            ground_truth = self.canny_edge_detection(edge_image)
        
        # 确保二值化
        edge_binary = (edge_image > 0).astype(np.uint8)
        gt_binary = (ground_truth > 0).astype(np.uint8)
        
        # 计算评估指标
        tp = np.sum((edge_binary == 1) & (gt_binary == 1))  # 真阳性
        fp = np.sum((edge_binary == 1) & (gt_binary == 0))  # 假阳性
        fn = np.sum((edge_binary == 0) & (gt_binary == 1))  # 假阴性
        tn = np.sum((edge_binary == 0) & (gt_binary == 0))  # 真阴性
        
        # 避免除以零
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        metrics = {
            'true_positive': tp,
            'false_positive': fp,
            'false_negative': fn,
            'true_negative': tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy
        }
        
        return metrics
    
    def compute_edge_thickness(self, edge_image):
        """计算边缘厚度（平均宽度）"""
        # 使用形态学操作估计边缘厚度
        from skimage.morphology import binary_dilation, binary_erosion
        
        edge_binary = (edge_image > 0)
        
        # 计算边缘像素的连通区域
        num_labels, labels = cv2.connectedComponents(edge_binary.astype(np.uint8))
        
        if num_labels <= 1:
            return 0
        
        thicknesses = []
        for label in range(1, num_labels):
            mask = (labels == label).astype(np.uint8)
            
            # 计算区域面积和周长
            area = np.sum(mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    # 估计平均厚度（假设为圆形区域）
                    thickness = 4 * area / perimeter
                    thicknesses.append(thickness)
        
        return np.mean(thicknesses) if thicknesses else 0

def create_test_images():
    """创建测试图像"""
    images = []
    
    # 图像1：简单几何形状
    img1 = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (250, 250), 255, 5)
    cv2.circle(img1, (150, 150), 80, 200, 3)
    cv2.line(img1, (50, 150), (250, 150), 150, 2)
    images.append(('几何形状', img1))
    
    # 图像2：添加噪声
    img2 = img1.copy()
    noise = np.random.normal(0, 30, img2.shape).astype(np.uint8)
    img2 = cv2.add(img2, noise)
    images.append(('带噪声的几何形状', img2))
    
    # 图像3：纹理图像
    img3 = np.zeros((300, 300), dtype=np.uint8)
    for i in range(0, 300, 20):
        cv2.line(img3, (i, 0), (i, 300), 255, 2)
        cv2.line(img3, (0, i), (300, i), 200, 2)
    images.append(('网格纹理', img3))
    
    return images

def main():
    """主函数：比较不同的边缘检测算子"""
    # 创建测试图像
    test_images = create_test_images()
    
    # 创建边缘检测器
    detector = EdgeDetector()
    
    # 定义要测试的边缘检测方法
    edge_methods = [
        ('Sobel', detector.sobel_edge_detection),
        ('Prewitt', detector.prewitt_edge_detection),
        ('Roberts', detector.roberts_edge_detection),
        ('Canny', detector.canny_edge_detection),
        ('Laplacian', detector.laplacian_edge_detection),
        ('Zero-Crossing', detector.zero_crossing_edge_detection)
    ]
    
    # 存储所有结果
    all_results = {}
    execution_times = {}
    
    # 对每幅图像应用所有边缘检测方法
    for img_name, image in test_images:
        print(f"\n处理图像: {img_name}")
        all_results[img_name] = {}
        execution_times[img_name] = {}
        
        # 使用Canny作为参考（ground truth）
        start_time = time.time()
        canny_edges = detector.canny_edge_detection(image)
        canny_time = time.time() - start_time
        execution_times[img_name]['Canny'] = canny_time
        
        for method_name, method_func in edge_methods:
            print(f"  应用 {method_name}...")
            
            start_time = time.time()
            
            if method_name in ['Sobel', 'Prewitt', 'Roberts']:
                # 这些方法返回多个值
                result = method_func(image)
                edge_image = result[0]  # 梯度幅值
            elif method_name == 'Canny':
                edge_image = canny_edges
            else:
                # 其他方法直接返回边缘图像
                if method_name == 'Laplacian':
                    edge_image = method_func(image, ksize=3)
                elif method_name == 'Zero-Crossing':
                    edge_image = method_func(image, threshold=0.01)
                else:
                    edge_image = method_func(image)
            
            exec_time = time.time() - start_time
            execution_times[img_name][method_name] = exec_time
            
            # 评估结果（使用Canny作为参考）
            metrics = detector.evaluate_edge_detection(edge_image, canny_edges)
            
            # 计算边缘厚度
            thickness = detector.compute_edge_thickness(edge_image)
            
            all_results[img_name][method_name] = {
                'image': edge_image,
                'metrics': metrics,
                'thickness': thickness,
                'execution_time': exec_time
            }
    
    # 可视化结果
    for img_name, image in test_images:
        plt.figure(figsize=(15, 10))
        
        # 显示原始图像
        plt.subplot(3, 4, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'原始图像: {img_name}')
        plt.axis('off')
        
        # 显示每种方法的边缘检测结果
        for idx, (method_name, _) in enumerate(edge_methods):
            row = idx // 3 + 1
            col = idx % 3 + 2
            
            plt.subplot(3, 4, row * 4 + col)
            
            if method_name in all_results[img_name]:
                result = all_results[img_name][method_name]
                edge_img = result['image']
                metrics = result['metrics']
                
                plt.imshow(edge_img, cmap='gray')
                plt.title(f'{method_name}\nF1={metrics["f1_score"]:.3f}')
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results/edge_detection_{img_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # 创建综合比较图
    plt.figure(figsize=(12, 8))
    
    # F1分数比较
    plt.subplot(221)
    methods = [m[0] for m in edge_methods]
    
    for img_name, _ in test_images:
        f1_scores = []
        for method_name in methods:
            if method_name in all_results[img_name]:
                f1_scores.append(all_results[img_name][method_name]['metrics']['f1_score'])
            else:
                f1_scores.append(0)
        
        plt.plot(methods, f1_scores, marker='o', label=img_name)
    
    plt.xlabel('边缘检测方法')
    plt.ylabel('F1分数')
    plt.title('不同方法的F1分数比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 执行时间比较
    plt.subplot(222)
    for img_name, _ in test_images:
        exec_times = []
        for method_name in methods:
            if method_name in execution_times[img_name]:
                exec_times.append(execution_times[img_name][method_name] * 1000)  # 转换为毫秒
            else:
                exec_times.append(0)
        
        plt.plot(methods, exec_times, marker='s', label=img_name)
    
    plt.xlabel('边缘检测方法')
    plt.ylabel('执行时间 (ms)')
    plt.title('不同方法的执行时间比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 边缘厚度比较
    plt.subplot(223)
    for img_name, _ in test_images:
        thicknesses = []
        for method_name in methods:
            if method_name in all_results[img_name]:
                thicknesses.append(all_results[img_name][method_name]['thickness'])
            else:
                thicknesses.append(0)
        
        plt.bar(np.arange(len(methods)) + 0.2 * (test_images.index((img_name, _)) - 1), 
                thicknesses, width=0.2, label=img_name)
    
    plt.xlabel('边缘检测方法')
    plt.ylabel('平均边缘厚度 (像素)')
    plt.title('不同方法的边缘厚度比较')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(np.arange(len(methods)), methods, rotation=45)
    
    # 精度-召回率曲线
    plt.subplot(224)
    for method_name in methods:
        precisions = []
        recalls = []
        
        for img_name, _ in test_images:
            if method_name in all_results[img_name]:
                metrics = all_results[img_name][method_name]['metrics']
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
        
        if precisions and recalls:
            plt.scatter(recalls, precisions, label=method_name, s=100)
    
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精度 (Precision)')
    plt.title('精度-召回率散点图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/edge_detection_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 保存详细结果
    with open('results/edge_detection_results.txt', 'w') as f:
        f.write("边缘检测算子比较实验结果\n")
        f.write("=" * 60 + "\n\n")
        
        for img_name, image in test_images:
            f.write(f"图像: {img_name}\n")
            f.write("-" * 40 + "\n")
            
            for method_name in methods:
                if method_name in all_results[img_name]:
                    result = all_results[img_name][method_name]
                    metrics = result['metrics']
                    
                    f.write(f"\n{method_name}:\n")
                    f.write(f"  执行时间: {result['execution_time']:.4f} 秒\n")
                    f.write(f"  边缘厚度: {result['thickness']:.2f} 像素\n")
                    f.write(f"  真阳性: {metrics['true_positive']}\n")
                    f.write(f"  假阳性: {metrics['false_positive']}\n")
                    f.write(f"  假阴性: {metrics['false_negative']}\n")
                    f.write(f"  精度: {metrics['precision']:.4f}\n")
                    f.write(f"  召回率: {metrics['recall']:.4f}\n")
                    f.write(f"  F1分数: {metrics['f1_score']:.4f}\n")
                    f.write(f"  准确率: {metrics['accuracy']:.4f}\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
    
    print("\n实验完成！结果已保存到 results/ 目录")

if __name__ == "__main__":
    main()