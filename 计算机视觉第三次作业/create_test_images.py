"""
创建测试图像
用于计算机视觉实验
"""

import cv2
import numpy as np
import os

def create_geometric_shapes():
    """创建几何形状测试图像"""
    # 图像1：简单几何形状
    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # 绘制矩形
    cv2.rectangle(img1, (50, 50), (200, 200), (255, 0, 0), 3)
    
    # 绘制圆形
    cv2.circle(img1, (300, 150), 60, (0, 255, 0), 3)
    
    # 绘制三角形
    pts = np.array([[100, 250], [150, 150], [200, 250]], np.int32)
    cv2.polylines(img1, [pts], True, (0, 0, 255), 3)
    
    # 绘制线条
    cv2.line(img1, (250, 50), (350, 250), (255, 255, 0), 2)
    
    cv2.imwrite('images/geometric_shapes.jpg', img1)
    print("创建图像: geometric_shapes.jpg")
    
    # 灰度版本
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('images/geometric_shapes_gray.jpg', gray1)
    
    return img1, gray1

def create_building_facade():
    """创建建筑立面测试图像"""
    img2 = np.zeros((400, 500, 3), dtype=np.uint8)
    
    # 建筑主体
    cv2.rectangle(img2, (100, 100), (400, 350), (200, 200, 200), -1)
    cv2.rectangle(img2, (100, 100), (400, 350), (100, 100, 100), 3)
    
    # 窗户
    for i in range(3):
        for j in range(4):
            x = 120 + j * 70
            y = 120 + i * 70
            cv2.rectangle(img2, (x, y), (x+40, y+40), (100, 150, 200), -1)
            cv2.rectangle(img2, (x, y), (x+40, y+40), (50, 100, 150), 2)
    
    # 门
    cv2.rectangle(img2, (250, 300), (300, 350), (150, 100, 50), -1)
    cv2.rectangle(img2, (250, 300), (300, 350), (100, 70, 30), 3)
    
    # 屋顶
    pts = np.array([[80, 100], [250, 50], [420, 100]], np.int32)
    cv2.fillPoly(img2, [pts], (180, 120, 80))
    
    cv2.imwrite('images/building_facade.jpg', img2)
    print("创建图像: building_facade.jpg")
    
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('images/building_facade_gray.jpg', gray2)
    
    return img2, gray2

def create_natural_scene():
    """创建自然场景测试图像"""
    img3 = np.zeros((350, 450, 3), dtype=np.uint8)
    
    # 天空
    cv2.rectangle(img3, (0, 0), (450, 200), (135, 206, 235), -1)
    
    # 地面
    cv2.rectangle(img3, (0, 200), (450, 350), (34, 139, 34), -1)
    
    # 山
    pts1 = np.array([[50, 200], [150, 100], [250, 200]], np.int32)
    pts2 = np.array([[200, 200], [300, 120], [400, 200]], np.int32)
    cv2.fillPoly(img3, [pts1], (139, 69, 19))
    cv2.fillPoly(img3, [pts2], (160, 82, 45))
    
    # 树
    for x in [80, 180, 280, 380]:
        # 树干
        cv2.rectangle(img3, (x-5, 220), (x+5, 250), (101, 67, 33), -1)
        # 树冠
        cv2.circle(img3, (x, 200), 25, (0, 100, 0), -1)
        cv2.circle(img3, (x, 180), 20, (0, 120, 0), -1)
        cv2.circle(img3, (x, 160), 15, (0, 140, 0), -1)
    
    # 云
    cv2.circle(img3, (100, 80), 20, (255, 255, 255), -1)
    cv2.circle(img3, (120, 70), 25, (255, 255, 255), -1)
    cv2.circle(img3, (140, 80), 20, (255, 255, 255), -1)
    
    cv2.circle(img3, (300, 100), 20, (255, 255, 255), -1)
    cv2.circle(img3, (320, 90), 25, (255, 255, 255), -1)
    cv2.circle(img3, (340, 100), 20, (255, 255, 255), -1)
    
    cv2.imwrite('images/natural_scene.jpg', img3)
    print("创建图像: natural_scene.jpg")
    
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('images/natural_scene_gray.jpg', gray3)
    
    return img3, gray3

def create_texture_pattern():
    """创建纹理模式测试图像"""
    img4 = np.zeros((300, 300), dtype=np.uint8)
    
    # 创建棋盘格纹理
    for i in range(0, 300, 30):
        for j in range(0, 300, 30):
            if (i//30 + j//30) % 2 == 0:
                img4[i:i+30, j:j+30] = 255
    
    # 添加圆形纹理
    for i in range(75, 300, 150):
        for j in range(75, 300, 150):
            cv2.circle(img4, (j, i), 40, 150, -1)
    
    cv2.imwrite('images/texture_pattern.jpg', img4)
    print("创建图像: texture_pattern.jpg")
    
    return img4, img4.copy()

def create_noisy_version(image, noise_level=25):
    """创建带噪声的图像版本"""
    if len(image.shape) == 3:
        h, w, c = image.shape
        noise = np.random.normal(0, noise_level, (h, w, c)).astype(np.uint8)
    else:
        h, w = image.shape
        noise = np.random.normal(0, noise_level, (h, w)).astype(np.uint8)
    
    noisy_image = cv2.add(image, noise)
    return noisy_image

def create_transformed_versions(image, base_name):
    """创建变换后的图像版本"""
    h, w = image.shape[:2]
    
    # 旋转版本
    M_rotate = cv2.getRotationMatrix2D((w/2, h/2), 15, 1.0)
    rotated = cv2.warpAffine(image, M_rotate, (w, h))
    cv2.imwrite(f'images/{base_name}_rotated.jpg', rotated)
    
    # 缩放版本
    scaled = cv2.resize(image, None, fx=0.8, fy=0.8)
    # 调整到原始大小
    scaled_resized = np.zeros_like(image)
    sh, sw = scaled.shape[:2]
    scaled_resized[:sh, :sw] = scaled
    cv2.imwrite(f'images/{base_name}_scaled.jpg', scaled_resized)
    
    # 平移版本
    M_translate = np.float32([[1, 0, 20], [0, 1, 15]])
    translated = cv2.warpAffine(image, M_translate, (w, h))
    cv2.imwrite(f'images/{base_name}_translated.jpg', translated)
    
    print(f"为 {base_name} 创建了旋转、缩放和平移版本")

def main():
    """主函数"""
    print("创建测试图像...")
    
    # 确保images目录存在
    if not os.path.exists('images'):
        os.makedirs('images')
    
    # 创建各种测试图像
    print("\n1. 创建几何形状图像...")
    img1, gray1 = create_geometric_shapes()
    create_transformed_versions(gray1, 'geometric_shapes')
    
    print("\n2. 创建建筑立面图像...")
    img2, gray2 = create_building_facade()
    create_transformed_versions(gray2, 'building_facade')
    
    print("\n3. 创建自然场景图像...")
    img3, gray3 = create_natural_scene()
    create_transformed_versions(gray3, 'natural_scene')
    
    print("\n4. 创建纹理模式图像...")
    img4, gray4 = create_texture_pattern()
    create_transformed_versions(gray4, 'texture_pattern')
    
    print("\n5. 创建带噪声的图像版本...")
    # 为每幅图像创建噪声版本
    noisy1 = create_noisy_version(gray1, 20)
    cv2.imwrite('images/geometric_shapes_noisy.jpg', noisy1)
    
    noisy2 = create_noisy_version(gray2, 30)
    cv2.imwrite('images/building_facade_noisy.jpg', noisy2)
    
    noisy3 = create_noisy_version(gray3, 25)
    cv2.imwrite('images/natural_scene_noisy.jpg', noisy3)
    
    noisy4 = create_noisy_version(gray4, 15)
    cv2.imwrite('images/texture_pattern_noisy.jpg', noisy4)
    
    print("\n6. 创建边缘检测测试图像...")
    # 专门用于边缘检测测试的图像
    edge_test = np.zeros((300, 300), dtype=np.uint8)
    
    # 不同宽度的线条
    cv2.line(edge_test, (50, 50), (250, 50), 255, 1)   # 细线
    cv2.line(edge_test, (50, 100), (250, 100), 255, 3)  # 中等线
    cv2.line(edge_test, (50, 150), (250, 150), 255, 5)  # 粗线
    
    # 不同对比度的边缘
    cv2.rectangle(edge_test, (50, 200), (150, 250), 200, -1)  # 中等对比度
    cv2.rectangle(edge_test, (200, 200), (250, 250), 255, -1) # 高对比度
    
    cv2.imwrite('images/edge_test_pattern.jpg', edge_test)
    
    print("\n所有测试图像创建完成！")
    print("图像保存在 'images/' 目录中")
    
    # 显示图像统计信息
    print("\n图像统计:")
    image_files = os.listdir('images')
    print(f"总共创建了 {len(image_files)} 个图像文件")
    
    for img_file in sorted(image_files):
        img_path = os.path.join('images', img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape
            print(f"  {img_file}: {w}x{h} 像素")

if __name__ == "__main__":
    main()