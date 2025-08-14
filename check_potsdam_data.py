import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid

from potsdam_dataset import PostdamDataset

def visualize_batch(ir_imgs, vi_imgs, masks, fusion_targets, save_path=None):
    """
    Visualize a batch of data from the Potsdam dataset
    
    Args:
        ir_imgs: IR images [B, 1, H, W]
        vi_imgs: VI images [B, 3, H, W]
        masks: Segmentation masks [B, H, W]
        fusion_targets: Fusion targets [B, 1, H, W]
        save_path: Path to save the visualization, if None, just display
    """
    batch_size = ir_imgs.shape[0]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(batch_size, 4, figsize=(20, 5*batch_size))
    
    # Handle case of single image
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    # 定义mask颜色映射
    # 6类 - 使用有区分度的颜色
    mask_colors = np.array([
        [0, 0, 0],          # 类别0 - 黑色
        [0, 255, 0],        # 类别1 - 绿色
        [0, 0, 255],        # 类别2 - 蓝色
        [255, 255, 0],      # 类别3 - 黄色
        [255, 0, 255],      # 类别4 - 紫色
        [255, 0, 0]         # 类别5 - 红色
    ]) / 255.0
    
    # 类别名称
    class_names = ['类别0', '类别1', '类别2', '类别3', '类别4', '类别5']
    
    # Iterate over each image in the batch
    for i in range(batch_size):
        # IR image (grayscale)
        ir_img = ir_imgs[i, 0].cpu().numpy()
        axes[i, 0].imshow(ir_img, cmap='gray')
        axes[i, 0].set_title(f'IR Image {i+1}')
        axes[i, 0].axis('off')
        
        # VI image (RGB)
        vi_img = vi_imgs[i].permute(1, 2, 0).cpu().numpy()
        # Normalize for display
        vi_img = (vi_img - vi_img.min()) / (vi_img.max() - vi_img.min() + 1e-8)
        axes[i, 1].imshow(vi_img)
        axes[i, 1].set_title(f'VI Image {i+1}')
        axes[i, 1].axis('off')
        
        # Mask - 使用彩色映射显示多类别
        mask = masks[i].cpu().numpy()
        
        # 创建彩色mask
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3))
        
        # 为每个类别分配颜色
        for class_idx in range(6):  # 有6个类别 (0-5)
            if class_idx in np.unique(mask):  # 只处理存在的类别
                colored_mask[mask == class_idx] = mask_colors[class_idx]
        
        axes[i, 2].imshow(colored_mask)
        axes[i, 2].set_title(f'Mask {i+1}')
        axes[i, 2].axis('off')
        
        # 添加类别图例
        legend_elements = [plt.Rectangle((0, 0), 1, 1, color=mask_colors[j]) for j in range(6)]
        if i == 0:  # 只在第一行添加图例
            axes[i, 2].legend(legend_elements, class_names, loc='upper right', 
                           bbox_to_anchor=(1.3, 1), fontsize='small')
        
        # Fusion target
        fusion_target = fusion_targets[i, 0].cpu().numpy()
        axes[i, 3].imshow(fusion_target, cmap='gray')
        axes[i, 3].set_title(f'Fusion Target {i+1}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def display_data_stats(ir_imgs, vi_imgs, masks, fusion_targets):
    """
    Display statistics about the data
    
    Args:
        ir_imgs: IR images [B, 1, H, W]
        vi_imgs: VI images [B, 3, H, W]
        masks: Segmentation masks [B, H, W]
        fusion_targets: Fusion targets [B, 1, H, W]
    """
    print("\n=== Data Statistics ===")
    
    # IR images
    print(f"\nIR Images:")
    print(f"Shape: {ir_imgs.shape}")
    print(f"Data type: {ir_imgs.dtype}")
    print(f"Min value: {ir_imgs.min().item():.4f}")
    print(f"Max value: {ir_imgs.max().item():.4f}")
    print(f"Mean value: {ir_imgs.mean().item():.4f}")
    print(f"Std deviation: {ir_imgs.std().item():.4f}")
    
    # VI images
    print(f"\nVI Images:")
    print(f"Shape: {vi_imgs.shape}")
    print(f"Data type: {vi_imgs.dtype}")
    print(f"Min value: {vi_imgs.min().item():.4f}")
    print(f"Max value: {vi_imgs.max().item():.4f}")
    print(f"Mean value: {vi_imgs.mean().item():.4f}")
    print(f"Std deviation: {vi_imgs.std().item():.4f}")
    
    # Masks
    print(f"\nMasks:")
    print(f"Shape: {masks.shape}")
    print(f"Data type: {masks.dtype}")
    print(f"Min value: {masks.min().item()}")
    print(f"Max value: {masks.max().item()}")
    print(f"Unique values: {torch.unique(masks).tolist()}")
    
    # Count occurrences of each class
    for class_idx in range(6):  # 0-5类别
        class_count = (masks == class_idx).sum().item()
        percentage = 100.0 * class_count / masks.numel()
        print(f"  类别 {class_idx}: {class_count} 像素 ({percentage:.2f}%)")
    
    # Fusion targets
    print(f"\nFusion Targets:")
    print(f"Shape: {fusion_targets.shape}")
    print(f"Data type: {fusion_targets.dtype}")
    print(f"Min value: {fusion_targets.min().item():.4f}")
    print(f"Max value: {fusion_targets.max().item():.4f}")
    print(f"Mean value: {fusion_targets.mean().item():.4f}")
    print(f"Std deviation: {fusion_targets.std().item():.4f}")

def check_fusion_calculation(ir_imgs, vi_imgs, fusion_targets):
    """
    Verify that fusion targets are correctly calculated as max(ir_img, vi_mean)
    
    Args:
        ir_imgs: IR images [B, 1, H, W]
        vi_imgs: VI images [B, 3, H, W]
        fusion_targets: Fusion targets [B, 1, H, W]
    """
    print("\n=== Fusion Target Verification ===")
    
    # Manually calculate fusion targets
    vi_mean = torch.mean(vi_imgs, dim=1, keepdim=True)
    calculated_fusion = torch.maximum(ir_imgs, vi_mean)
    
    # Check if calculated fusion matches the provided fusion targets
    fusion_match = torch.allclose(calculated_fusion, fusion_targets, rtol=1e-4, atol=1e-4)
    if fusion_match:
        print("✓ Fusion targets are correctly calculated as max(ir_img, vi_mean)")
    else:
        print("✗ Fusion targets may not be correctly calculated")
        mse = torch.mean((calculated_fusion - fusion_targets) ** 2).item()
        print(f"Mean Squared Error: {mse:.6f}")

def main():
    # Data path
    data_path = "./"  # Change this to your data path
    
    # Create a simple transformation (just convert to tensor, no normalization)
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
    
    # Create dataset (both train and test)
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
    
    # Print dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create output directory for visualizations if it doesn't exist
    vis_dir = "data_visualization"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Check different samples from the dataset
    batch_size = 4
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=test_dataset.collate_fn
    )
    
    # Visualize and check train data
    print("\n== Checking Training Data ==")
    train_batch = next(iter(train_loader))
    ir_imgs, vi_imgs, masks, fusion_targets = train_batch
    
    # Check shapes and values
    display_data_stats(ir_imgs, vi_imgs, masks, fusion_targets)
    
    # Verify fusion calculation
    check_fusion_calculation(ir_imgs, vi_imgs, fusion_targets)
    
    # Visualize
    visualize_batch(
        ir_imgs, vi_imgs, masks, fusion_targets,
        save_path=os.path.join(vis_dir, "train_batch.png")
    )
    
    # Visualize and check test data
    print("\n== Checking Test Data ==")
    test_batch = next(iter(test_loader))
    ir_imgs, vi_imgs, masks, fusion_targets = test_batch
    
    # Check shapes and values
    display_data_stats(ir_imgs, vi_imgs, masks, fusion_targets)
    
    # Verify fusion calculation
    check_fusion_calculation(ir_imgs, vi_imgs, fusion_targets)
    
    # Visualize
    visualize_batch(
        ir_imgs, vi_imgs, masks, fusion_targets,
        save_path=os.path.join(vis_dir, "test_batch.png")
    )
    
    # Individual visualizations (better to see specific examples clearly)
    print("\n== Visualizing Individual Examples ==")
    for i in range(min(3, len(train_dataset))):
        ir_img, vi_img, mask, fusion_target = train_dataset[i]
        
        # Convert to batch format for visualization
        ir_img = ir_img.unsqueeze(0)
        vi_img = vi_img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        fusion_target = fusion_target.unsqueeze(0)
        
        visualize_batch(
            ir_img, vi_img, mask, fusion_target,
            save_path=os.path.join(vis_dir, f"train_example_{i+1}.png")
        )

if __name__ == "__main__":
    main() 