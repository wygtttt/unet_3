import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

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


def visualize_comparison(vi_img, ir_img, gt_mask, seg_pred, fusion_output, save_path):
    """
    将原始图像、真实标签、分割预测结果和融合结果可视化并保存
    
    Args:
        vi_img: 可见光图像, [C, H, W]
        ir_img: 红外图像, [1, H, W]
        gt_mask: 真实标签, [H, W]
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
    
    # 创建彩色分割掩码 - GT
    h, w = gt_mask.shape
    colored_gt_mask = np.zeros((h, w, 3))
    for class_idx in range(6):
        colored_gt_mask[gt_mask == class_idx] = mask_colors[class_idx]
    
    # 创建彩色分割掩码 - 预测
    h, w = seg_pred.shape
    colored_pred_mask = np.zeros((h, w, 3))
    for class_idx in range(6):
        colored_pred_mask[seg_pred == class_idx] = mask_colors[class_idx]
    
    # 创建可视化图像 - 5个面板：VI图像、IR图像、GT掩码、预测掩码、融合结果
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    axes[0].imshow(vi_img)
    axes[0].set_title('VI Image')
    axes[0].axis('off')
    
    axes[1].imshow(ir_img, cmap='gray')
    axes[1].set_title('IR Image')
    axes[1].axis('off')
    
    axes[2].imshow(colored_gt_mask)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    axes[3].imshow(colored_pred_mask)
    axes[3].set_title('Prediction')
    axes[3].axis('off')
    
    axes[4].imshow(fusion_output, cmap='gray')
    axes[4].set_title('Fusion Result')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def create_output_dirs(output_base_dir):
    """
    创建输出目录结构
    """
    dirs = {
        'seg_colored': os.path.join(output_base_dir, 'segmentation_colored'),
        'fusion': os.path.join(output_base_dir, 'fusion'),
        'comparison': os.path.join(output_base_dir, 'comparison')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def process_batch(model, data_root, output_dir, device):
    """
    批量处理测试集并保存结果
    
    Args:
        model: 加载好权重的模型
        data_root: 数据集根目录
        output_dir: 输出目录
        device: 使用的设备
    """
    # 创建输出目录
    output_dirs = create_output_dirs(output_dir)
    
    # 设置数据目录
    ir_dir = os.path.join(data_root, 'ir_test')
    vi_dir = os.path.join(data_root, 'vi_test')
    mask_dir = os.path.join(data_root, 'mask_test')
    
    # 获取测试集中的所有图像名称
    ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.tif'))])
    
    # 使用训练时计算的均值和标准差
    vi_mean = (0.3405778515395395, 0.3637113326614421, 0.33767444875971564)
    vi_std = (0.14075637061449023, 0.13888050726764334, 0.14433405425755172)
    ir_mean = (0.38396125844223883,)
    ir_std = (0.14174455530391647,)
    
    # 图像预处理
    ir_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=ir_mean, std=ir_std)
    ])
    
    vi_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=vi_mean, std=vi_std)
    ])
    
    # 批量处理图像
    total_time = 0
    num_images = len(ir_files)
    
    for i, ir_file in enumerate(tqdm(ir_files, desc="处理测试图像")):
        # 文件名（不含扩展名）
        file_base = os.path.splitext(ir_file)[0]
        
        # 构建文件路径
        ir_path = os.path.join(ir_dir, ir_file)
        vi_path = os.path.join(vi_dir, f"{file_base}.png")  # 假定VI图像与IR图像具有相同的基本名称
        mask_path = os.path.join(mask_dir, f"{file_base}.png")  # 假定掩码与IR图像具有相同的基本名称
        
        # 检查文件是否存在
        if not os.path.exists(vi_path):
            print(f"找不到VI图像: {vi_path}, 跳过.")
            continue
        
        # 加载图像
        ir_img = Image.open(ir_path).convert('L')
        vi_img = Image.open(vi_path).convert('RGB')
        
        # 加载ground truth掩码（如果存在）
        if os.path.exists(mask_path):
            gt_mask = np.array(Image.open(mask_path).convert('L'))
        else:
            print(f"找不到掩码: {mask_path}, 将使用全黑掩码.")
            gt_mask = np.zeros((ir_img.height, ir_img.width), dtype=np.uint8)
        
        # 应用变换
        ir_tensor = ir_transform(ir_img).unsqueeze(0)  # 添加批次维度
        vi_tensor = vi_transform(vi_img).unsqueeze(0)  # 添加批次维度
        
        # 确保IR图像有正确的通道维度 [B, 1, H, W]
        if len(ir_tensor.shape) == 3:
            ir_tensor = ir_tensor.unsqueeze(1)
        
        # 模型推理
        with torch.no_grad():
            t_start = time_synchronized()
            ir_tensor = ir_tensor.to(device)
            vi_tensor = vi_tensor.to(device)
            outputs = model(ir_tensor, vi_tensor)
            t_end = time_synchronized()
            
            inference_time = t_end - t_start
            total_time += inference_time
            
            # 处理分割输出
            seg_output = outputs['seg']
            seg_pred = seg_output.argmax(1).squeeze(0).cpu().numpy()
            
            # 处理融合输出
            fusion_output = outputs['fusion'].cpu()
        
        # 保存彩色分割结果
        save_colored_mask(seg_pred, os.path.join(output_dirs['seg_colored'], f"{file_base}_seg.png"))
        
        # 保存融合结果
        fusion_img = fusion_output.squeeze(0).squeeze(0).numpy()
        fusion_img = (fusion_img - fusion_img.min()) / (fusion_img.max() - fusion_img.min()) * 255.0
        fusion_img = fusion_img.astype(np.uint8)
        Image.fromarray(fusion_img).save(os.path.join(output_dirs['fusion'], f"{file_base}_fusion.png"))
        
        # 保存对比可视化
        visualize_comparison(vi_tensor[0].cpu(), ir_tensor[0].cpu(), gt_mask, seg_pred, 
                           fusion_output[0], os.path.join(output_dirs['comparison'], f"{file_base}_comparison.png"))
    
    # 打印平均推理时间
    avg_time = total_time / num_images
    print(f"平均推理时间: {avg_time:.4f}秒/图像")
    
    return output_dirs


def main():
    # 参数设置
    num_classes = 6  # 0-5共6个类别
    weights_path = "/ifs/root/ipa01/101/user_101003/Xdj/unet/save_weights/best_dual_model.pth"
    data_root = "./postdam"
    output_dir = "./prediction_results"
    
    # 检查权重文件是否存在
    assert os.path.exists(weights_path), f"模型权重 {weights_path} 不存在."
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用 {device} 设备.")
    
    # 创建模型
    model = DualDecoderUNet(num_classes=num_classes, base_c=32)
    
    # 加载权重
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # 批量处理测试集
    output_dirs = process_batch(model, data_root, output_dir, device)
    
    print(f"预测结果已保存到: {output_dir}")
    print(f"- 彩色分割结果: {output_dirs['seg_colored']}")
    print(f"- 融合结果: {output_dirs['fusion']}")
    print(f"- 对比可视化: {output_dirs['comparison']}")


if __name__ == '__main__':
    main() 