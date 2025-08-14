import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import train_utils.distributed_utils as utils

from src import DualDecoderUNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, FusionLoss
from potsdam_dataset import PostdamDataset
import transforms as T


class SegmentationPresetTrain:
    def __init__(self, base_size=None, crop_size=None, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 移除随机缩放和随机裁剪，只保留翻转和标准化
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # 不再使用base_size和crop_size参数
    if train:
        return SegmentationPresetTrain(mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


class IRSegmentationPresetTrain:
    def __init__(self, base_size=None, crop_size=None, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.5,), std=(0.5,)):
        # 移除随机缩放和随机裁剪，只保留翻转和标准化
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class IRSegmentationPresetEval:
    def __init__(self, mean=(0.5,), std=(0.5,)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_ir_transform(train, mean=(0.5,), std=(0.5,)):
    # 不再使用base_size和crop_size参数
    if train:
        return IRSegmentationPresetTrain(mean=mean, std=std)
    else:
        return IRSegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    model = DualDecoderUNet(num_classes=num_classes, base_c=32)
    return model


def criterion(outputs, targets, num_classes, fusion_loss_fn):
    """
    Combined loss function for segmentation and fusion tasks
    Args:
        outputs: Dictionary containing 'seg' and 'fusion' outputs
        targets: Tuple of (ir_images, vi_images, masks, fusion_targets)
        num_classes: Number of segmentation classes
        fusion_loss_fn: Fusion loss function (combination of L1 and SSIM)
    Returns:
        Combined loss (segmentation loss + fusion loss)
    """
    ir_imgs, vi_imgs, masks, fusion_targets = targets
    
    # Segmentation loss (cross entropy + dice)
    seg_outputs = outputs['seg']
    segmentation_loss = torch.nn.functional.cross_entropy(
        seg_outputs, masks, ignore_index=255
    )
    
    # Add dice loss if needed for segmentation
    from train_utils.dice_coefficient_loss import dice_loss, build_target
    dice_target = build_target(masks, num_classes, ignore_index=255)
    segmentation_loss += dice_loss(seg_outputs, dice_target, multiclass=True, ignore_index=255)
    
    # Fusion loss
    fusion_outputs = outputs['fusion']
    fusion_loss = fusion_loss_fn(fusion_outputs, fusion_targets)
    
    # Combined loss (you can adjust weights as needed)
    total_loss = segmentation_loss + fusion_loss
    
    return {
        'total': total_loss,
        'seg': segmentation_loss,
        'fusion': fusion_loss
    }


def evaluate_model(model, data_loader, device, num_classes, fusion_loss_fn):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_fusion_loss = 0.0
    total_ssim = 0.0
    total_l1_loss = 0.0
    samples = 0
    
    with torch.no_grad():
        for ir_imgs, vi_imgs, masks, fusion_targets in metric_logger.log_every(data_loader, 100, header):
            ir_imgs, vi_imgs = ir_imgs.to(device), vi_imgs.to(device)
            masks, fusion_targets = masks.to(device), fusion_targets.to(device)
            
            outputs = model(ir_imgs, vi_imgs)
            seg_output = outputs['seg']
            fusion_output = outputs['fusion']
            
            # Update segmentation metrics
            confmat.update(masks.flatten(), seg_output.argmax(1).flatten())
            dice.update(seg_output, masks)
            
            # 计算融合指标
            fusion_loss = fusion_loss_fn(fusion_output, fusion_targets)
            
            # 计算SSIM值 (SSIM接近1表示相似度高)
            ssim_value = fusion_loss_fn.ssim_loss(fusion_output, fusion_targets)
            
            # 计算L1损失
            l1_loss = fusion_loss_fn.l1_loss(fusion_output, fusion_targets)
            
            # 累加指标
            total_fusion_loss += fusion_loss.item() * ir_imgs.size(0)
            total_ssim += ssim_value.item() * ir_imgs.size(0)
            total_l1_loss += l1_loss.item() * ir_imgs.size(0)
            samples += ir_imgs.size(0)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()
        
        # 计算平均值
        avg_fusion_loss = total_fusion_loss / samples
        avg_ssim = total_ssim / samples
        avg_l1_loss = total_l1_loss / samples

    return confmat, dice.value.item(), avg_fusion_loss, avg_ssim, avg_l1_loss


def save_segmentation_visualization(images, masks, predictions, epoch, step, output_dir):
    """
    Save visualization of segmentation results.
    
    Args:
        images: Input VI images (B, C, H, W)
        masks: Ground truth masks (B, H, W)
        predictions: Model predictions (B, C, H, W)
        epoch: Current epoch
        step: Current step
        output_dir: Directory to save visualizations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Take only the first few images in the batch to visualize
    num_images = min(4, images.shape[0])
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    # If there's only one image, make sure axes is 2D
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    # 定义mask颜色映射 - 6个类别
    mask_colors = np.array([
        [0, 0, 0],          # 类别0 - 黑色
        [0, 255, 0],        # 类别1 - 绿色
        [0, 0, 255],        # 类别2 - 蓝色
        [255, 255, 0],      # 类别3 - 黄色
        [255, 0, 255],      # 类别4 - 紫色
        [255, 0, 0]         # 类别5 - 红色
    ]) / 255.0
    
    # Iterate through images
    for i in range(num_images):
        # Original image
        img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
        # Normalize for display
        img = (img - img.min()) / (img.max() - img.min())
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask - 彩色显示
        mask = masks[i].detach().cpu().numpy()
        h, w = mask.shape
        colored_mask_gt = np.zeros((h, w, 3))
        
        # 为每个类别分配颜色
        for class_idx in range(6):  # 有6个类别 (0-5)
            if class_idx in np.unique(mask):  # 只处理存在的类别
                colored_mask_gt[mask == class_idx] = mask_colors[class_idx]
                
        axes[i, 1].imshow(colored_mask_gt)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Predicted mask - 彩色显示
        pred = torch.argmax(predictions[i], dim=0).detach().cpu().numpy()
        h, w = pred.shape
        colored_mask_pred = np.zeros((h, w, 3))
        
        # 为每个类别分配颜色
        for class_idx in range(6):  # 有6个类别 (0-5)
            if class_idx in np.unique(pred):  # 只处理存在的类别
                colored_mask_pred[pred == class_idx] = mask_colors[class_idx]
                
        axes[i, 2].imshow(colored_mask_pred)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'segmentation_epoch{epoch}_step{step}.png'))
    plt.close()


def train_one_epoch_dual(model, optimizer, data_loader, device, epoch, num_classes,
                         lr_scheduler, fusion_loss_fn, print_freq=10, scaler=None, 
                         vis_dir='segmentation_vis'):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('seg_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('fusion_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # 根据类别数设置loss权重
    loss_weight = None

    # Create visualization directory if it doesn't exist
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        
    # Step counter
    step = 0

    for ir_imgs, vi_imgs, masks, fusion_targets in metric_logger.log_every(data_loader, print_freq, header):
        step += 1
        ir_imgs, vi_imgs = ir_imgs.to(device), vi_imgs.to(device)
        masks, fusion_targets = masks.to(device), fusion_targets.to(device)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(ir_imgs, vi_imgs)
            losses = criterion(outputs, (ir_imgs, vi_imgs, masks, fusion_targets), 
                              num_classes=num_classes, fusion_loss_fn=fusion_loss_fn)
            loss = losses['total']

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), 
                            seg_loss=losses['seg'].item(),
                            fusion_loss=losses['fusion'].item(),
                            lr=lr)
        
        # Save visualization every 100 steps
        if step % 100 == 0:
            with torch.no_grad():
                # Get segmentation outputs
                seg_output = outputs['seg']
                
                # Save visualization
                save_segmentation_visualization(
                    vi_imgs, masks, seg_output, 
                    epoch, step, vis_dir
                )
                print(f"Saved segmentation visualization at epoch {epoch}, step {step}")

    return (metric_logger.meters["loss"].global_avg, 
            metric_logger.meters["seg_loss"].global_avg,
            metric_logger.meters["fusion_loss"].global_avg, 
            lr)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    
    # segmentation num_classes - 现在有6个类别 (0-5)
    num_classes = args.num_classes

    # Computed mean and std for normalization
    # RGB mean and std for VI images
    vi_mean = (0.3405778515395395, 0.3637113326614421, 0.33767444875971564)
    vi_std = (0.14075637061449023, 0.13888050726764334, 0.14433405425755172)
    
    # Single channel mean and std for IR images
    ir_mean = (0.38396125844223883,)
    ir_std = (0.14174455530391647,)

    results_file = "results_dual{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = PostdamDataset(args.data_path,
                                  train=True,
                                  transforms=get_transform(train=True, mean=vi_mean, std=vi_std),
                                  ir_transforms=get_ir_transform(train=True, mean=ir_mean, std=ir_std))

    val_dataset = PostdamDataset(args.data_path,
                                train=False,
                                transforms=get_transform(train=False, mean=vi_mean, std=vi_std),
                                ir_transforms=get_ir_transform(train=False, mean=ir_mean, std=ir_std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              shuffle=True,
                                              pin_memory=True,
                                              collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=1,
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # Initialize fusion loss function
    fusion_loss_fn = FusionLoss(ssim_weight=0.5, l1_weight=0.5)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Create learning rate update strategy
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # Create directory for segmentation visualizations
    vis_dir = os.path.join("segmentation_vis", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        total_loss, seg_loss, fusion_loss, lr = train_one_epoch_dual(
            model, optimizer, train_loader, device, epoch, num_classes,
            lr_scheduler=lr_scheduler, fusion_loss_fn=fusion_loss_fn, 
            print_freq=args.print_freq, scaler=scaler, vis_dir=vis_dir
        )

        # 更新评估函数调用，获取SSIM和L1损失
        confmat, dice, val_fusion_loss, val_ssim, val_l1_loss = evaluate_model(
            model, val_loader, device=device, num_classes=num_classes,
            fusion_loss_fn=fusion_loss_fn
        )
        
        val_info = str(confmat)
        print(val_info)
        print(f"Dice coefficient: {dice:.3f}")
        print(f"Validation fusion loss: {val_fusion_loss:.4f}")
        
        # 打印融合解码器的SSIM和L1指标
        print(f"Fusion Decoder - SSIM: {val_ssim:.4f} (higher is better)")
        print(f"Fusion Decoder - L1 Loss: {val_l1_loss:.4f} (lower is better)")
        
        # Write to results file
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {total_loss:.4f}\n" \
                         f"seg_loss: {seg_loss:.4f}\n" \
                         f"fusion_loss: {fusion_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n" \
                         f"val_fusion_loss: {val_fusion_loss:.4f}\n" \
                         f"val_ssim: {val_ssim:.4f}\n" \
                         f"val_l1_loss: {val_l1_loss:.4f}\n"
            f.write(train_info + val_info + "\n\n")

        # Save best model
        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args
        }
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "save_weights/best_dual_model.pth")
        else:
            torch.save(save_file, "save_weights/dual_model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet dual decoder training")

    parser.add_argument("--data-path", default="./", help="Potsdam dataset root")
    # 修改为6个类别 (0-5)
    parser.add_argument("--num-classes", default=6, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args) 