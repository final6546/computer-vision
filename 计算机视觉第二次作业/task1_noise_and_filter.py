"""
任务一：对原始图像添加噪声并进行不同滤波方法的对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class NoiseGenerator:
    """噪声生成器"""
    
    @staticmethod
    def add_gaussian_noise(image, mean=0, sigma=25):
        """添加高斯噪声"""
        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
        """添加椒盐噪声"""
        noisy_image = image.copy()
        
        # 盐噪声（白点）
        salt_mask = np.random.random(image.shape[:2]) < salt_prob
        noisy_image[salt_mask] = 255
        
        # 椒噪声（黑点）
        pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
        noisy_image[pepper_mask] = 0
        
        return noisy_image


class ImageFilter:
    """图像滤波器"""
    
    @staticmethod
    def gaussian_filter(image, kernel_size=5, sigma=1.0):
        """高斯平滑滤波"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    @staticmethod
    def median_filter(image, kernel_size=5):
        """中值滤波"""
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def fourier_filter(image, cutoff_frequency=30):
        """傅里叶频域滤波（低通滤波）"""
        # 转换为灰度图（如果是彩色图）
        if len(image.shape) == 3:
            # 对每个通道分别处理
            filtered_channels = []
            for i in range(image.shape[2]):
                filtered_channel = ImageFilter._fourier_filter_single_channel(
                    image[:, :, i], cutoff_frequency
                )
                filtered_channels.append(filtered_channel)
            return np.stack(filtered_channels, axis=2)
        else:
            return ImageFilter._fourier_filter_single_channel(image, cutoff_frequency)
    
    @staticmethod
    def _fourier_filter_single_channel(channel, cutoff_frequency):
        """对单通道进行傅里叶滤波"""
        # 傅里叶变换
        f_transform = fftpack.fft2(channel)
        f_shift = fftpack.fftshift(f_transform)
        
        # 创建低通滤波器
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols))
        
        # 创建圆形低通滤波器
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - crow)**2 + (j - ccol)**2) <= cutoff_frequency:
                    mask[i, j] = 1
        
        # 应用滤波器
        f_shift_filtered = f_shift * mask
        
        # 逆傅里叶变换
        f_ishift = fftpack.ifftshift(f_shift_filtered)
        img_back = fftpack.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        return np.clip(img_back, 0, 255).astype(np.uint8)


def evaluate_image_quality(original, processed):
    """评估图像质量"""
    psnr_value = psnr(original, processed)
    
    # 计算SSIM，处理小图像的情况
    min_side = min(original.shape[0], original.shape[1])
    if min_side < 7:
        # 图像太小，使用默认值
        ssim_value = 0.0
    else:
        # 根据图像大小调整win_size
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        try:
            ssim_value = ssim(
                original, 
                processed, 
                win_size=win_size,
                channel_axis=2 if len(original.shape) == 3 else None
            )
        except Exception as e:
            print(f"SSIM计算警告: {e}")
            ssim_value = 0.0
    
    return psnr_value, ssim_value


def select_image_file():
    """使用文件对话框选择图像"""
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()  # 隐藏主窗口
        root.attributes('-topmost', True)  # 置顶
        
        file_path = filedialog.askopenfilename(
            title='选择图像文件',
            filetypes=[
                ('图像文件', '*.jpg *.jpeg *.png *.bmp *.tiff'),
                ('JPEG', '*.jpg *.jpeg'),
                ('PNG', '*.png'),
                ('所有文件', '*.*')
            ]
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"无法打开文件选择对话框: {e}")
        return None


def task1_main(image_path=None):
    """任务一主函数"""
    # 如果没有指定图像路径，尝试选择图像
    if image_path is None:
        print("请选择要处理的图像...")
        image_path = select_image_file()
        
        if image_path:
            print(f"已选择图像: {image_path}")
        else:
            print("未选择图像，使用默认测试图像...")
            image_path = 'test_image.jpg'
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        # 如果没有图像，创建一个测试图像
        print("未找到图像，创建测试图像...")
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(image_path, image)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1. 添加噪声
    print("添加噪声...")
    gaussian_noisy = NoiseGenerator.add_gaussian_noise(image_rgb, sigma=25)
    salt_pepper_noisy = NoiseGenerator.add_salt_pepper_noise(image_rgb, salt_prob=0.02, pepper_prob=0.02)
    
    # 2. 对加噪声图像进行滤波
    print("应用滤波器...")
    filters = {
        '高斯平滑': lambda img: ImageFilter.gaussian_filter(img, kernel_size=5, sigma=1.5),
        '中值滤波': lambda img: ImageFilter.median_filter(img, kernel_size=5),
        '傅里叶频域滤波': lambda img: ImageFilter.fourier_filter(img, cutoff_frequency=30)
    }
    
    # 对高斯噪声图像进行滤波
    gaussian_filtered = {}
    for name, filter_func in filters.items():
        gaussian_filtered[name] = filter_func(gaussian_noisy)
    
    # 对椒盐噪声图像进行滤波
    salt_pepper_filtered = {}
    for name, filter_func in filters.items():
        salt_pepper_filtered[name] = filter_func(salt_pepper_noisy)
    
    # 3. 可视化结果
    print("生成对比图...")
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    # 第一行：原图和高斯噪声相关
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('原始图像', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gaussian_noisy)
    axes[0, 1].set_title('高斯噪声', fontsize=12)
    axes[0, 1].axis('off')
    
    for idx, (name, filtered_img) in enumerate(gaussian_filtered.items(), start=2):
        axes[0, idx].imshow(filtered_img)
        psnr_val, ssim_val = evaluate_image_quality(image_rgb, filtered_img)
        axes[0, idx].set_title(f'{name}\nPSNR: {psnr_val:.2f}dB\nSSIM: {ssim_val:.4f}', fontsize=10)
        axes[0, idx].axis('off')
    
    # 第二行：椒盐噪声相关
    axes[1, 0].imshow(image_rgb)
    axes[1, 0].set_title('原始图像', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(salt_pepper_noisy)
    axes[1, 1].set_title('椒盐噪声', fontsize=12)
    axes[1, 1].axis('off')
    
    for idx, (name, filtered_img) in enumerate(salt_pepper_filtered.items(), start=2):
        axes[1, idx].imshow(filtered_img)
        psnr_val, ssim_val = evaluate_image_quality(image_rgb, filtered_img)
        axes[1, idx].set_title(f'{name}\nPSNR: {psnr_val:.2f}dB\nSSIM: {ssim_val:.4f}', fontsize=10)
        axes[1, idx].axis('off')
    
    # 第三行：定量对比分析
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')
    
    # PSNR对比
    filter_names = list(filters.keys())
    gaussian_psnr = [evaluate_image_quality(image_rgb, gaussian_filtered[name])[0] for name in filter_names]
    salt_pepper_psnr = [evaluate_image_quality(image_rgb, salt_pepper_filtered[name])[0] for name in filter_names]
    
    x = np.arange(len(filter_names))
    width = 0.35
    
    axes[2, 2].bar(x - width/2, gaussian_psnr, width, label='高斯噪声', alpha=0.8)
    axes[2, 2].bar(x + width/2, salt_pepper_psnr, width, label='椒盐噪声', alpha=0.8)
    axes[2, 2].set_ylabel('PSNR (dB)', fontsize=10)
    axes[2, 2].set_title('PSNR对比', fontsize=12)
    axes[2, 2].set_xticks(x)
    axes[2, 2].set_xticklabels(filter_names, rotation=15, ha='right', fontsize=8)
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    # SSIM对比
    gaussian_ssim = [evaluate_image_quality(image_rgb, gaussian_filtered[name])[1] for name in filter_names]
    salt_pepper_ssim = [evaluate_image_quality(image_rgb, salt_pepper_filtered[name])[1] for name in filter_names]
    
    axes[2, 3].bar(x - width/2, gaussian_ssim, width, label='高斯噪声', alpha=0.8)
    axes[2, 3].bar(x + width/2, salt_pepper_ssim, width, label='椒盐噪声', alpha=0.8)
    axes[2, 3].set_ylabel('SSIM', fontsize=10)
    axes[2, 3].set_title('SSIM对比', fontsize=12)
    axes[2, 3].set_xticks(x)
    axes[2, 3].set_xticklabels(filter_names, rotation=15, ha='right', fontsize=8)
    axes[2, 3].legend()
    axes[2, 3].grid(True, alpha=0.3)
    
    axes[2, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig('task1_results.png', dpi=300, bbox_inches='tight')
    print("结果已保存到 task1_results.png")
    plt.show()
    
    # 打印分析结果
    print("\n" + "="*60)
    print("定性分析结果：")
    print("="*60)
    print("\n1. 高斯噪声去噪效果：")
    for name in filter_names:
        psnr_val, ssim_val = evaluate_image_quality(image_rgb, gaussian_filtered[name])
        print(f"   {name}: PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.4f}")
    
    print("\n2. 椒盐噪声去噪效果：")
    for name in filter_names:
        psnr_val, ssim_val = evaluate_image_quality(image_rgb, salt_pepper_filtered[name])
        print(f"   {name}: PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.4f}")
    
    print("\n3. 结论：")
    print("   - 对于高斯噪声：高斯平滑和傅里叶频域滤波效果较好")
    print("   - 对于椒盐噪声：中值滤波效果最佳")
    print("   - 中值滤波对脉冲噪声（椒盐）有很好的抑制作用")
    print("   - 高斯滤波和频域滤波更适合处理连续的高斯噪声")


if __name__ == '__main__':
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 如果提供了图像路径参数
        task1_main(image_path=sys.argv[1])
    else:
        # 否则打开文件选择对话框
        task1_main()
