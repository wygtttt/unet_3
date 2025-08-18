import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from src import DualDecoderUNet
from Evaluator.Evaluator import Evaluator


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


def visualize_comparison(vi_img, ir_img, gt_mask, ir_seg_pred, vi_seg_pred, ir_recon, vi_recon, fusion_output, save_path):
    """
    将原始图像、真实标签、分割预测结果、重建结果和融合结果可视化并保存
    
    Args:
        vi_img: 可见光图像, [C, H, W]
        ir_img: 红外图像, [1, H, W]
        gt_mask: 真实标签, [H, W]
        ir_seg_pred: IR分割预测结果, [H, W], 值范围 0-5
        vi_seg_pred: VI分割预测结果, [H, W], 值范围 0-5
        ir_recon: IR重建结果, [1, H, W]
        vi_recon: VI重建结果, [1, H, W]
        fusion_output: 融合结果, [1, H, W]
        save_path: 保存路径
    """
    # 转换为numpy数组用于可视化
    vi_img = vi_img[0].cpu().numpy()  # VI现在也是单通道
    ir_img = ir_img[0].cpu().numpy()
    ir_recon = ir_recon[0].cpu().numpy()
    vi_recon = vi_recon[0].cpu().numpy()
    fusion_output = fusion_output[0].cpu().numpy()
    
    # 归一化图像用于显示
    vi_img = (vi_img - vi_img.min()) / (vi_img.max() - vi_img.min() + 1e-8)
    ir_img = (ir_img - ir_img.min()) / (ir_img.max() - ir_img.min() + 1e-8)
    ir_recon = (ir_recon - ir_recon.min()) / (ir_recon.max() - ir_recon.min() + 1e-8)
    vi_recon = (vi_recon - vi_recon.min()) / (vi_recon.max() - vi_recon.min() + 1e-8)
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
    
    # 创建彩色分割掩码 - IR预测
    h, w = ir_seg_pred.shape
    colored_ir_pred_mask = np.zeros((h, w, 3))
    for class_idx in range(6):
        colored_ir_pred_mask[ir_seg_pred == class_idx] = mask_colors[class_idx]
    
    # 创建彩色分割掩码 - VI预测
    h, w = vi_seg_pred.shape
    colored_vi_pred_mask = np.zeros((h, w, 3))
    for class_idx in range(6):
        colored_vi_pred_mask[vi_seg_pred == class_idx] = mask_colors[class_idx]
    
    # 创建可视化图像 - 8个面板：原始图像、重建图像、分割结果、融合结果
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 第一行：原始图像和重建图像
    axes[0, 0].imshow(vi_img, cmap='gray')
    axes[0, 0].set_title('VI Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ir_img, cmap='gray')
    axes[0, 1].set_title('IR Original')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(vi_recon, cmap='gray')
    axes[0, 2].set_title('VI Reconstruction')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(ir_recon, cmap='gray')
    axes[0, 3].set_title('IR Reconstruction')
    axes[0, 3].axis('off')
    
    # 第二行：分割结果和融合结果
    axes[1, 0].imshow(colored_gt_mask)
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(colored_ir_pred_mask)
    axes[1, 1].set_title('IR Segmentation')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(colored_vi_pred_mask)
    axes[1, 2].set_title('VI Segmentation')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(fusion_output, cmap='gray')
    axes[1, 3].set_title('Fusion Result')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def create_output_dirs(output_base_dir):
    """
    创建输出目录结构
    """
    dirs = {
        'ir_seg_colored': os.path.join(output_base_dir, 'ir_segmentation_colored'),
        'vi_seg_colored': os.path.join(output_base_dir, 'vi_segmentation_colored'),
        'ir_recon': os.path.join(output_base_dir, 'ir_reconstruction'),
        'vi_recon': os.path.join(output_base_dir, 'vi_reconstruction'),
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
    vi_mean = (0.38396125844223883,)  # Y channel typical normalization value
    vi_std = (0.14174455530391647,)   # Y channel typical normalization value
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
    scd_scores = []  # 存储SCD指标
    
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
        vi_img = Image.open(vi_path).convert('RGB')  # VI也转换为灰度图像，因为模型期望1通道输入
        vi_img = vi_img.convert('YCbCr')
        # Extract only Y channel (luminance)
        vi_img = vi_img.split()[0]  # Get Y channel only
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
            
            # 处理IR分割输出
            ir_seg_output = outputs['ir_seg']
            ir_seg_pred = ir_seg_output.argmax(1).squeeze(0).cpu().numpy()
            
            # 处理VI分割输出
            vi_seg_output = outputs['vi_seg']
            vi_seg_pred = vi_seg_output.argmax(1).squeeze(0).cpu().numpy()
            
            # 处理重建输出
            ir_recon_output = outputs['ir_recon'].cpu()
            vi_recon_output = outputs['vi_recon'].cpu()
            
            # 处理融合输出
            fusion_output = outputs['fusion'].cpu()
        
        # 保存IR分割结果
        save_colored_mask(ir_seg_pred, os.path.join(output_dirs['ir_seg_colored'], f"{file_base}_ir_seg.png"))
        
        # 保存VI分割结果
        save_colored_mask(vi_seg_pred, os.path.join(output_dirs['vi_seg_colored'], f"{file_base}_vi_seg.png"))
        
        # 保存IR重建结果
        ir_recon_img = ir_recon_output.squeeze(0).squeeze(0).numpy()
        ir_recon_img = (ir_recon_img - ir_recon_img.min()) / (ir_recon_img.max() - ir_recon_img.min()) * 255.0
        ir_recon_img = ir_recon_img.astype(np.uint8)
        Image.fromarray(ir_recon_img).save(os.path.join(output_dirs['ir_recon'], f"{file_base}_ir_recon.png"))
        
        # 保存VI重建结果
        vi_recon_img = vi_recon_output.squeeze(0).squeeze(0).numpy()
        vi_recon_img = (vi_recon_img - vi_recon_img.min()) / (vi_recon_img.max() - vi_recon_img.min()) * 255.0
        vi_recon_img = vi_recon_img.astype(np.uint8)
        Image.fromarray(vi_recon_img).save(os.path.join(output_dirs['vi_recon'], f"{file_base}_vi_recon.png"))
        
        # 保存融合结果
        fusion_img = fusion_output.squeeze(0).squeeze(0).numpy()
        fusion_img = (fusion_img - fusion_img.min()) / (fusion_img.max() - fusion_img.min()) * 255.0
        fusion_img = fusion_img.astype(np.uint8)
        Image.fromarray(fusion_img).save(os.path.join(output_dirs['fusion'], f"{file_base}_fusion.png"))
        
        # 计算SCD指标
        # 将原始图像转换为numpy数组用于SCD计算
        ir_img_array = np.array(ir_img, dtype=np.float32)
        vi_img_array = np.array(vi_img, dtype=np.float32)
        fusion_img_float = fusion_img.astype(np.float32)
        
        # 计算SCD指标
        scd_score = Evaluator.SCD(fusion_img_float, ir_img_array, vi_img_array)
        scd_scores.append(scd_score)
        
        print(f"图像 {file_base}: SCD = {scd_score:.4f}")
        
        # 保存对比可视化
        visualize_comparison(vi_tensor[0].cpu(), ir_tensor[0].cpu(), gt_mask, ir_seg_pred, vi_seg_pred,
                           ir_recon_output[0], vi_recon_output[0], fusion_output[0], 
                           os.path.join(output_dirs['comparison'], f"{file_base}_comparison.png"))
    
    # 打印平均推理时间
    avg_time = total_time / num_images
    print(f"平均推理时间: {avg_time:.4f}秒/图像")
    
    # 计算并打印SCD指标统计信息
    if scd_scores:
        avg_scd = np.mean(scd_scores)
        max_scd = np.max(scd_scores)
        min_scd = np.min(scd_scores)
        std_scd = np.std(scd_scores)
        
        print("\n=== SCD指标统计 ===")
        print(f"平均SCD: {avg_scd:.4f}")
        print(f"最大SCD: {max_scd:.4f}")
        print(f"最小SCD: {min_scd:.4f}")
        print(f"标准差: {std_scd:.4f}")
        print(f"处理图像数量: {len(scd_scores)}")
        
        # 保存SCD结果到文件
        scd_results_path = os.path.join(output_dir, 'scd_results.txt')
        with open(scd_results_path, 'w', encoding='utf-8') as f:
            f.write("SCD指标计算结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"平均SCD: {avg_scd:.4f}\n")
            f.write(f"最大SCD: {max_scd:.4f}\n")
            f.write(f"最小SCD: {min_scd:.4f}\n")
            f.write(f"标准差: {std_scd:.4f}\n")
            f.write(f"处理图像数量: {len(scd_scores)}\n\n")
            f.write("各图像详细结果:\n")
            for i, (ir_file, scd) in enumerate(zip(ir_files[:len(scd_scores)], scd_scores)):
                file_base = os.path.splitext(ir_file)[0]
                f.write(f"{i+1:3d}. {file_base}: {scd:.4f}\n")
        
        print(f"SCD结果已保存到: {scd_results_path}")
    
    return output_dirs


def main():
    # 参数设置
    num_classes = 6  # 0-5共6个类别
    weights_path = "/ifs/root/ipa01/101/user_101003/Xdj/unet_3/save_weights/best_dual_model.pth"
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
    print(f"- IR分割结果: {output_dirs['ir_seg_colored']}")
    print(f"- VI分割结果: {output_dirs['vi_seg_colored']}")
    print(f"- IR重建结果: {output_dirs['ir_recon']}")
    print(f"- VI重建结果: {output_dirs['vi_recon']}")
    print(f"- 融合结果: {output_dirs['fusion']}")
    print(f"- 对比可视化: {output_dirs['comparison']}")
    print(f"- SCD指标结果: {os.path.join(output_dir, 'scd_results.txt')}")


if __name__ == '__main__':
    main()