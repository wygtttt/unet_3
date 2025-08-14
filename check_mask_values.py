import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from potsdam_dataset import PostdamDataset

def check_raw_mask_file(mask_dir, num_samples=5):
    """
    直接检查mask文件的原始像素值，不经过任何转换
    """
    print("\n=== 检查原始Mask文件 ===")
    
    if not os.path.exists(mask_dir):
        print(f"错误: 目录 {mask_dir} 不存在")
        return
    
    # 获取mask文件列表
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.tif', '.png', '.jpg', '.jpeg'))])
    
    if not mask_files:
        print(f"错误: 在 {mask_dir} 中没有找到图像文件")
        return
    
    print(f"找到 {len(mask_files)} 个mask文件")
    
    # 检查前几个文件
    samples = min(num_samples, len(mask_files))
    
    for i in range(samples):
        mask_path = os.path.join(mask_dir, mask_files[i])
        print(f"\n检查文件: {mask_files[i]}")
        
        # 使用PIL读取原始图像
        mask_img = Image.open(mask_path)
        mask_array = np.array(mask_img)
        
        # 打印基本信息
        print(f"图像形状: {mask_array.shape}")
        print(f"数据类型: {mask_array.dtype}")
        print(f"最小值: {mask_array.min()}")
        print(f"最大值: {mask_array.max()}")
        print(f"唯一值: {np.unique(mask_array)}")
        
        # 计算唯一值的分布
        unique, counts = np.unique(mask_array, return_counts=True)
        for u, c in zip(unique, counts):
            percentage = (c / mask_array.size) * 100
            print(f"  值 {u}: {c} 像素 ({percentage:.2f}%)")
        
        # 为前两个样本绘制直方图
        if i < 2:
            plt.figure(figsize=(10, 6))
            plt.hist(mask_array.flatten(), bins=50)
            plt.title(f'Mask Pixel Value Distribution - {mask_files[i]}')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.savefig(f'mask_histogram_{i+1}.png')
            print(f"已保存直方图: mask_histogram_{i+1}.png")
            plt.close()

def check_processed_masks(data_path):
    """
    检查经过PostdamDataset处理后的mask值
    """
    print("\n=== 检查处理后的Mask ===")
    
    # 使用简单转换
    class SimpleTransform:
        def __init__(self):
            self.transform = T.ToTensor()
            
        def __call__(self, img, mask=None):
            img_tensor = self.transform(img)
            if mask is not None:
                if isinstance(mask, torch.Tensor):
                    return img_tensor, mask
                mask_tensor = torch.from_numpy(np.array(mask)).long()
                return img_tensor, mask_tensor
            return img_tensor
    
    # 创建训练和测试数据集
    train_dataset = PostdamDataset(
        data_path, 
        train=True,
        transforms=SimpleTransform(),
        ir_transforms=SimpleTransform()
    )
    
    test_dataset = PostdamDataset(
        data_path, 
        train=False,
        transforms=SimpleTransform(),
        ir_transforms=SimpleTransform()
    )
    
    # 检查训练集样本
    print("\n训练集mask检查:")
    for i in range(min(3, len(train_dataset))):
        _, _, mask, _ = train_dataset[i]
        print(f"\n样本 {i+1}:")
        print(f"Mask形状: {mask.shape}")
        print(f"数据类型: {mask.dtype}")
        print(f"最小值: {mask.min().item()}")
        print(f"最大值: {mask.max().item()}")
        print(f"唯一值: {torch.unique(mask).tolist()}")
        
        # 保存原始mask图像用于可视化
        plt.figure(figsize=(8, 8))
        plt.imshow(mask.numpy(), cmap='tab20')
        plt.colorbar(label='类别')
        plt.title(f'Processed Mask - Train Sample {i+1}')
        plt.savefig(f'processed_mask_train_{i+1}.png')
        print(f"已保存处理后的mask图像: processed_mask_train_{i+1}.png")
        plt.close()
    
    # 检查测试集样本
    print("\n测试集mask检查:")
    for i in range(min(3, len(test_dataset))):
        _, _, mask, _ = test_dataset[i]
        print(f"\n样本 {i+1}:")
        print(f"Mask形状: {mask.shape}")
        print(f"数据类型: {mask.dtype}")
        print(f"最小值: {mask.min().item()}")
        print(f"最大值: {mask.max().item()}")
        print(f"唯一值: {torch.unique(mask).tolist()}")
        
        # 保存处理后的mask图像
        plt.figure(figsize=(8, 8))
        plt.imshow(mask.numpy(), cmap='tab20')
        plt.colorbar(label='类别')
        plt.title(f'Processed Mask - Test Sample {i+1}')
        plt.savefig(f'processed_mask_test_{i+1}.png')
        print(f"已保存处理后的mask图像: processed_mask_test_{i+1}.png")
        plt.close()

def check_mask_preprocessing(data_path):
    """
    检查PostdamDataset中mask处理的每一步
    """
    print("\n=== 检查Mask预处理步骤 ===")
    
    # 训练集mask目录
    mask_dir = os.path.join(data_path, "postdam", "mask_train")
    
    if not os.path.exists(mask_dir):
        print(f"错误: 目录 {mask_dir} 不存在")
        return
    
    # 获取mask文件列表
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.tif', '.png', '.jpg', '.jpeg'))])
    
    if not mask_files:
        print(f"错误: 在 {mask_dir} 中没有找到图像文件")
        return
    
    # 选择第一个文件进行详细分析
    mask_path = os.path.join(mask_dir, mask_files[0])
    print(f"分析文件: {mask_files[0]}")
    
    # 步骤1: 使用PIL加载
    mask = Image.open(mask_path).convert('L')
    mask_array_1 = np.array(mask)
    print("\n步骤1 - PIL加载:")
    print(f"形状: {mask_array_1.shape}")
    print(f"数据类型: {mask_array_1.dtype}")
    print(f"值范围: [{mask_array_1.min()}, {mask_array_1.max()}]")
    print(f"唯一值: {np.unique(mask_array_1)}")
    
    # 步骤2: 除以255
    mask_array_2 = mask_array_1 / 255
    print("\n步骤2 - 除以255:")
    print(f"形状: {mask_array_2.shape}")
    print(f"数据类型: {mask_array_2.dtype}")
    print(f"值范围: [{mask_array_2.min()}, {mask_array_2.max()}]")
    print(f"唯一值: {np.unique(mask_array_2)}")
    
    # 步骤3: 转回PIL
    mask_pil = Image.fromarray(mask_array_2)
    mask_array_3 = np.array(mask_pil)
    print("\n步骤3 - 转回PIL:")
    print(f"形状: {mask_array_3.shape}")
    print(f"数据类型: {mask_array_3.dtype}")
    print(f"值范围: [{mask_array_3.min()}, {mask_array_3.max()}]")
    print(f"唯一值: {np.unique(mask_array_3)}")
    
    # 步骤4: 转为tensor
    mask_tensor = torch.from_numpy(mask_array_2).long()
    print("\n步骤4 - 转为tensor:")
    print(f"形状: {mask_tensor.shape}")
    print(f"数据类型: {mask_tensor.dtype}")
    print(f"值范围: [{mask_tensor.min().item()}, {mask_tensor.max().item()}]")
    print(f"唯一值: {torch.unique(mask_tensor).tolist()}")
    
    # 可视化原始mask和处理后的mask
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(mask_array_1, cmap='gray')
    axes[0].set_title('Original Mask')
    axes[0].axis('off')
    
    axes[1].imshow(mask_array_2, cmap='gray')
    axes[1].set_title('After Division by 255')
    axes[1].axis('off')
    
    axes[2].imshow(mask_tensor.numpy(), cmap='tab20')
    axes[2].set_title('Final Tensor (as Long)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('mask_processing_steps.png')
    print("已保存处理步骤图像: mask_processing_steps.png")
    plt.close()

def main():
    # 更改为您的数据路径
    data_path = "./"
    
    # 检查训练集和测试集的mask目录
    train_mask_dir = os.path.join(data_path, "postdam", "mask_train")
    test_mask_dir = os.path.join(data_path, "postdam", "mask_test")
    
    # 检查原始mask文件
    print("\n===== 检查训练集mask =====")
    check_raw_mask_file(train_mask_dir)
    
    print("\n===== 检查测试集mask =====")
    check_raw_mask_file(test_mask_dir)
    
    # 检查经过处理的mask
    check_processed_masks(data_path)
    
    # 检查预处理步骤
    check_mask_preprocessing(data_path)

if __name__ == "__main__":
    main() 