import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

from src import DualDecoderUNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def save_colored_mask(mask, filename):
    """
    将类别索引转换为彩色图像并保存
    
    Args:
        mask: 预测的分割掩码, shape [H, W], 值范围 0-5
        filename: 保存路径
    """
    # 定义颜色映射 - 与训练代码保持一致
    mask_colors = np.array([
        [0, 0, 0],          # 类别0 - 黑色
        [0, 255, 0],        # 类别1 - 绿色
        [0, 0, 255],        # 类别2 - 蓝色
        [255, 255, 0],      # 类别3 - 黄色
        [255, 0, 255],      # 类别4 - 紫色
        [255, 0, 0]         # 类别5 - 红色
    ], dtype=np.uint8)
    
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 为每个类别分配颜色
    for class_idx in range(6):  # 有6个类别 (0-5)
        colored_mask[mask == class_idx] = mask_colors[class_idx]
    
    # 保存为图像
    Image.fromarray(colored_mask).save(filename)


def visualize_results(vi_img, ir_img, seg_pred, fusion_output, save_path):
    """
    将原始图像、分割结果和融合结果可视化并保存
    
    Args:
        vi_img: 可见光图像, [C, H, W]
        ir_img: 红外图像, [1, H, W]
        seg_pred: 分割预测结果, [H, W], 值范围 0-5
        fusion_output: 融合结果, [1, H, W]
        save_path: 保存路径
    """
    # 转换为numpy数组用于可视化
    vi_img = vi_img.permute(1, 2, 0).cpu().numpy()
    ir_img = ir_img[0].cpu().numpy()
    fusion_output = fusion_output[0].cpu().numpy()
    
    # 归一化图像用于显示
    vi_img = (vi_img - vi_img.min()) / (vi_img.max() - vi_img.min() + 1e-8)
    ir_img = (ir_img - ir_img.min()) / (ir_img.max() - ir_img.min() + 1e-8)
    fusion_output = (fusion_output - fusion_output.min()) / (fusion_output.max() - fusion_output.min() + 1e-8)
    
    # 定义颜色映射
    mask_colors = np.array([
        [0, 0, 0],          # 类别0 - 黑色
        [0, 255, 0],        # 类别1 - 绿色
        [0, 0, 255],        # 类别2 - 蓝色
        [255, 255, 0],      # 类别3 - 黄色
        [255, 0, 255],      # 类别4 - 紫色
        [255, 0, 0]         # 类别5 - 红色
    ]) / 255.0
    
    # 创建彩色分割掩码
    h, w = seg_pred.shape
    colored_mask = np.zeros((h, w, 3))
    for class_idx in range(6):
        colored_mask[seg_pred == class_idx] = mask_colors[class_idx]
    
    # 创建可视化图像
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(vi_img)
    axes[0].set_title('VI Image')
    axes[0].axis('off')
    
    axes[1].imshow(ir_img, cmap='gray')
    axes[1].set_title('IR Image')
    axes[1].axis('off')
    
    axes[2].imshow(colored_mask)
    axes[2].set_title('Segmentation')
    axes[2].axis('off')
    
    axes[3].imshow(fusion_output, cmap='gray')
    axes[3].set_title('Fusion Result')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # 参数设置
    num_classes = 6  # 0-5共6个类别
    weights_path = "/ifs/root/ipa01/101/user_101003/Xdj/unet/save_weights/best_dual_model.pth"
    
    # 设置预测用的样本路径
    # 这些路径需要根据实际情况调整
    ir_img_path = "./postdam/ir_test/1.png"  # 红外图像路径
    vi_img_path = "./postdam/vi_test/1.png"  # 可见光图像路径
    
    # 检查文件是否存在
    assert os.path.exists(weights_path), f"模型权重 {weights_path} 不存在."
    assert os.path.exists(ir_img_path), f"IR图像 {ir_img_path} 不存在."
    assert os.path.exists(vi_img_path), f"VI图像 {vi_img_path} 不存在."
    
    # 使用训练时计算的均值和标准差
    vi_mean = (0.3405778515395395, 0.3637113326614421, 0.33767444875971564)
    vi_std = (0.14075637061449023, 0.13888050726764334, 0.14433405425755172)
    ir_mean = (0.38396125844223883,)
    ir_std = (0.14174455530391647,)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用 {device} 设备.")
    
    # 创建模型
    model = DualDecoderUNet(num_classes=num_classes, base_c=32)
    
    # 加载权重
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    
    # 加载图像
    ir_img = Image.open(ir_img_path).convert('L')
    vi_img = Image.open(vi_img_path).convert('RGB')
    
    # 图像预处理
    ir_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=ir_mean, std=ir_std)
    ])
    
    vi_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=vi_mean, std=vi_std)
    ])
    
    # 应用变换
    ir_tensor = ir_transform(ir_img).unsqueeze(0)  # 添加批次维度
    vi_tensor = vi_transform(vi_img).unsqueeze(0)  # 添加批次维度
    
    # 确保IR图像有正确的通道维度 [B, 1, H, W]
    if len(ir_tensor.shape) == 3:
        ir_tensor = ir_tensor.unsqueeze(1)
    
    # 进入评估模式
    model.eval()
    with torch.no_grad():
        # 初始化模型
        ir_height, ir_width = ir_tensor.shape[-2:]
        vi_height, vi_width = vi_tensor.shape[-2:]
        
        # 确保两个图像尺寸相同
        assert ir_height == vi_height and ir_width == vi_width, "IR和VI图像尺寸不一致"
        
        # 初始化模型（可选）
        init_ir = torch.zeros((1, 1, ir_height, ir_width), device=device)
        init_vi = torch.zeros((1, 3, vi_height, vi_width), device=device)
        model(init_ir, init_vi)
        
        # 推理
        t_start = time_synchronized()
        ir_tensor = ir_tensor.to(device)
        vi_tensor = vi_tensor.to(device)
        outputs = model(ir_tensor, vi_tensor)
        t_end = time_synchronized()
        
        print(f"推理时间: {t_end - t_start:.4f}秒")
        
        # 处理分割输出
        seg_output = outputs['seg']
        seg_pred = seg_output.argmax(1).squeeze(0).cpu().numpy()
        
        # 处理融合输出
        fusion_output = outputs['fusion'].cpu()
        
        # 保存结果
        save_colored_mask(seg_pred, "segmentation_result.png")
        
        # 将融合结果保存为图像
        fusion_img = fusion_output.squeeze(0).squeeze(0).numpy()
        fusion_img = (fusion_img - fusion_img.min()) / (fusion_img.max() - fusion_img.min()) * 255.0
        fusion_img = fusion_img.astype(np.uint8)
        Image.fromarray(fusion_img).save("fusion_result.png")
        
        # 可视化并保存对比图
        visualize_results(vi_tensor[0].cpu(), ir_tensor[0].cpu(), 
                         seg_pred, fusion_output[0], "dual_results.png")
        
        print("预测结果已保存.")


if __name__ == '__main__':
    main() 