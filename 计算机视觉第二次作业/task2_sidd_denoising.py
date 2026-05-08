"""
任务二：使用SIDD数据集进行图像去噪对比分析
包含传统方法和深度学习方法
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import json


class DnCNN(nn.Module):
    """DnCNN深度学习去噪网络"""
    def __init__(self, channels=3, num_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        
        layers = []
        layers.append(nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise


class SIDDDataset(Dataset):
    """SIDD数据集加载器"""
    def __init__(self, noisy_dir, clean_dir=None):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir) if clean_dir else None
        
        # 获取所有图像文件
        self.noisy_images = sorted(list(self.noisy_dir.glob('*.png')) + 
                                   list(self.noisy_dir.glob('*.jpg')))
        
        if self.clean_dir:
            self.clean_images = sorted(list(self.clean_dir.glob('*.png')) + 
                                      list(self.clean_dir.glob('*.jpg')))
        else:
            self.clean_images = None
    
    def __len__(self):
        return len(self.noisy_images)
    
    def __getitem__(self, idx):
        noisy_img = cv2.imread(str(self.noisy_images[idx]))
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
        
        if self.clean_images:
            clean_img = cv2.imread(str(self.clean_images[idx]))
            clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
            return noisy_img, clean_img, str(self.noisy_images[idx])
        
        return noisy_img, None, str(self.noisy_images[idx])


class TraditionalDenoiser:
    """传统去噪方法集合"""
    
    @staticmethod
    def gaussian_denoise(image, kernel_size=5, sigma=1.5):
        """高斯平滑去噪"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    @staticmethod
    def median_denoise(image, kernel_size=5):
        """中值滤波去噪"""
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def bilateral_denoise(image, d=9, sigma_color=75, sigma_space=75):
        """双边滤波去噪"""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def nlm_denoise(image, h=10, template_window_size=7, search_window_size=21):
        """非局部均值去噪"""
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 
                                               template_window_size, search_window_size)


class DeepLearningDenoiser:
    """深度学习去噪方法"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DnCNN(channels=3, num_layers=17).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"已加载模型: {model_path}")
        else:
            print("使用未训练的模型（仅用于演示）")
        
        self.model.eval()
    
    def denoise(self, image):
        """使用DnCNN进行去噪"""
        # 归一化
        img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            denoised = self.model(img_tensor)
        
        # 反归一化
        denoised = denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
        
        return denoised


def calculate_metrics(clean_img, denoised_img):
    """计算图像质量评价指标"""
    # PSNR
    psnr_value = psnr(clean_img, denoised_img)
    
    # SSIM - 处理小图像
    min_side = min(clean_img.shape[0], clean_img.shape[1])
    if min_side < 7:
        ssim_value = 0.0
    else:
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        try:
            ssim_value = ssim(clean_img, denoised_img, win_size=win_size, channel_axis=2)
        except Exception as e:
            print(f"SSIM计算警告: {e}")
            ssim_value = 0.0
    
    # MSE
    mse_value = np.mean((clean_img.astype(float) - denoised_img.astype(float)) ** 2)
    
    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'MSE': mse_value
    }


def qualitative_analysis(noisy_images, clean_images, num_samples=5):
    """定性分析：显示5张图像的去噪结果"""
    print("\n开始定性分析...")
    
    # 初始化去噪器
    traditional_methods = {
        '高斯平滑': TraditionalDenoiser.gaussian_denoise,
        '中值滤波': TraditionalDenoiser.median_denoise,
        '双边滤波': TraditionalDenoiser.bilateral_denoise,
        '非局部均值': TraditionalDenoiser.nlm_denoise
    }
    
    dl_denoiser = DeepLearningDenoiser()
    
    # 选择样本
    indices = np.linspace(0, len(noisy_images) - 1, num_samples, dtype=int)
    
    for idx in indices:
        noisy_img = noisy_images[idx]
        clean_img = clean_images[idx] if clean_images else None
        
        # 应用各种去噪方法
        results = {'原始噪声图像': noisy_img}
        
        if clean_img is not None:
            results['真实清晰图像'] = clean_img
        
        for name, method in traditional_methods.items():
            results[name] = method(noisy_img)
        
        results['DnCNN(深度学习)'] = dl_denoiser.denoise(noisy_img)
        
        # 可视化
        num_methods = len(results)
        fig, axes = plt.subplots(2, (num_methods + 1) // 2, figsize=(20, 8))
        axes = axes.flatten()
        
        for i, (name, img) in enumerate(results.items()):
            axes[i].imshow(img)
            
            # 如果有真实图像，计算指标
            if clean_img is not None and name not in ['原始噪声图像', '真实清晰图像']:
                metrics = calculate_metrics(clean_img, img)
                title = f'{name}\nPSNR: {metrics["PSNR"]:.2f}dB\nSSIM: {metrics["SSIM"]:.4f}'
            else:
                title = name
            
            axes[i].set_title(title, fontsize=10)
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(results), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'qualitative_sample_{idx}.png', dpi=200, bbox_inches='tight')
        print(f"已保存样本 {idx} 的对比图")
        plt.close()


def quantitative_analysis(dataset, output_file='quantitative_results.json'):
    """定量分析：对整个数据集进行评估"""
    print("\n开始定量分析...")
    
    # 初始化去噪器
    traditional_methods = {
        '高斯平滑': TraditionalDenoiser.gaussian_denoise,
        '中值滤波': TraditionalDenoiser.median_denoise,
        '双边滤波': TraditionalDenoiser.bilateral_denoise,
        '非局部均值': TraditionalDenoiser.nlm_denoise
    }
    
    dl_denoiser = DeepLearningDenoiser()
    
    # 存储所有方法的指标
    all_metrics = {name: {'PSNR': [], 'SSIM': [], 'MSE': []} 
                   for name in list(traditional_methods.keys()) + ['DnCNN(深度学习)']}
    
    # 遍历数据集
    for noisy_img, clean_img, img_path in tqdm(dataset, desc="处理图像"):
        if clean_img is None:
            print("警告：没有真实清晰图像，跳过定量分析")
            continue
        
        # 传统方法
        for name, method in traditional_methods.items():
            denoised = method(noisy_img)
            metrics = calculate_metrics(clean_img, denoised)
            
            for key, value in metrics.items():
                all_metrics[name][key].append(value)
        
        # 深度学习方法
        denoised_dl = dl_denoiser.denoise(noisy_img)
        metrics_dl = calculate_metrics(clean_img, denoised_dl)
        
        for key, value in metrics_dl.items():
            all_metrics['DnCNN(深度学习)'][key].append(value)
    
    # 计算平均值
    avg_metrics = {}
    for method_name, metrics in all_metrics.items():
        avg_metrics[method_name] = {
            'PSNR_mean': np.mean(metrics['PSNR']),
            'PSNR_std': np.std(metrics['PSNR']),
            'SSIM_mean': np.mean(metrics['SSIM']),
            'SSIM_std': np.std(metrics['SSIM']),
            'MSE_mean': np.mean(metrics['MSE']),
            'MSE_std': np.std(metrics['MSE'])
        }
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(avg_metrics, f, indent=4, ensure_ascii=False)
    
    print(f"\n定量分析结果已保存到 {output_file}")
    
    # 打印结果
    print("\n" + "="*80)
    print("定量分析结果汇总：")
    print("="*80)
    for method_name, metrics in avg_metrics.items():
        print(f"\n{method_name}:")
        print(f"  PSNR: {metrics['PSNR_mean']:.2f} ± {metrics['PSNR_std']:.2f} dB")
        print(f"  SSIM: {metrics['SSIM_mean']:.4f} ± {metrics['SSIM_std']:.4f}")
        print(f"  MSE:  {metrics['MSE_mean']:.2f} ± {metrics['MSE_std']:.2f}")
    
    # 可视化对比
    visualize_quantitative_results(avg_metrics)
    
    return avg_metrics


def visualize_quantitative_results(avg_metrics):
    """可视化定量分析结果"""
    methods = list(avg_metrics.keys())
    psnr_means = [avg_metrics[m]['PSNR_mean'] for m in methods]
    psnr_stds = [avg_metrics[m]['PSNR_std'] for m in methods]
    ssim_means = [avg_metrics[m]['SSIM_mean'] for m in methods]
    ssim_stds = [avg_metrics[m]['SSIM_std'] for m in methods]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # PSNR对比
    x = np.arange(len(methods))
    axes[0].bar(x, psnr_means, yerr=psnr_stds, capsize=5, alpha=0.8, color='steelblue')
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('不同方法的PSNR对比', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # SSIM对比
    axes[1].bar(x, ssim_means, yerr=ssim_stds, capsize=5, alpha=0.8, color='coral')
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('不同方法的SSIM对比', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('quantitative_comparison.png', dpi=300, bbox_inches='tight')
    print("定量对比图已保存到 quantitative_comparison.png")
    plt.close()


def create_sample_dataset(output_dir='sample_sidd_data'):
    """创建示例数据集（如果没有真实SIDD数据）"""
    print(f"创建示例数据集到 {output_dir}...")
    
    noisy_dir = Path(output_dir) / 'noisy'
    clean_dir = Path(output_dir) / 'clean'
    
    noisy_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成10张示例图像
    for i in range(10):
        # 创建清晰图像
        clean_img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        
        # 添加噪声
        noise = np.random.normal(0, 25, clean_img.shape)
        noisy_img = clean_img + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        # 保存
        cv2.imwrite(str(clean_dir / f'clean_{i:03d}.png'), 
                   cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(noisy_dir / f'noisy_{i:03d}.png'), 
                   cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))
    
    print(f"已创建 {len(list(noisy_dir.glob('*.png')))} 张示例图像")
    return str(noisy_dir), str(clean_dir)


def select_directory(title="选择文件夹"):
    """使用文件对话框选择文件夹"""
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        dir_path = filedialog.askdirectory(title=title)
        root.destroy()
        return dir_path if dir_path else None
    except Exception as e:
        print(f"无法打开文件夹选择对话框: {e}")
        return None


def main():
    """主函数"""
    print("="*80)
    print("任务二：SIDD数据集图像去噪对比分析")
    print("="*80)
    
    # 检查是否有SIDD数据集
    sidd_noisy_dir = 'SIDD_noisy'
    sidd_clean_dir = 'SIDD_clean'
    
    if not os.path.exists(sidd_noisy_dir):
        print("\n未找到SIDD数据集")
        print("选项1: 选择已有的噪声图像文件夹")
        print("选项2: 创建示例数据集")
        
        choice = input("\n请选择 (1/2，直接回车使用选项2): ").strip()
        
        if choice == '1':
            print("\n请选择噪声图像文件夹...")
            sidd_noisy_dir = select_directory("选择噪声图像文件夹")
            
            if sidd_noisy_dir:
                print(f"已选择噪声图像文件夹: {sidd_noisy_dir}")
                
                has_clean = input("是否有对应的清晰图像文件夹？(y/n，直接回车为n): ").strip().lower()
                
                if has_clean == 'y':
                    print("\n请选择清晰图像文件夹...")
                    sidd_clean_dir = select_directory("选择清晰图像文件夹")
                    if sidd_clean_dir:
                        print(f"已选择清晰图像文件夹: {sidd_clean_dir}")
                    else:
                        print("未选择清晰图像文件夹，将只进行定性分析")
                        sidd_clean_dir = None
                else:
                    sidd_clean_dir = None
            else:
                print("未选择文件夹，创建示例数据集...")
                sidd_noisy_dir, sidd_clean_dir = create_sample_dataset()
        else:
            print("\n创建示例数据集...")
            sidd_noisy_dir, sidd_clean_dir = create_sample_dataset()
    
    # 加载数据集
    dataset = SIDDDataset(sidd_noisy_dir, sidd_clean_dir)
    print(f"\n数据集大小: {len(dataset)} 张图像")
    
    # 准备数据
    noisy_images = []
    clean_images = []
    
    for noisy, clean, _ in dataset:
        noisy_images.append(noisy)
        if clean is not None:
            clean_images.append(clean)
    
    # 1. 定性分析
    qualitative_analysis(noisy_images, clean_images if clean_images else None, num_samples=min(5, len(dataset)))
    
    # 2. 定量分析
    if clean_images:
        quantitative_analysis(dataset)
    else:
        print("\n警告：没有真实清晰图像，无法进行定量分析")
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == '__main__':
    main()
