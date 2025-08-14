import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from potsdam_dataset import PostdamDataset


def compute_mean_std(data_loader, is_rgb=True):
    """
    Compute mean and std for a dataset.
    Args:
        data_loader: DataLoader for the dataset
        is_rgb: True for RGB images (VI), False for single channel images (IR)
    Returns:
        mean and std values
    """
    channels_sum = 0
    channels_squared_sum = 0
    num_batches = 0
    num_pixels = 0

    # Use tqdm for a progress bar
    for idx, data in enumerate(tqdm(data_loader, desc=f"Computing {'RGB' if is_rgb else 'IR'} stats")):
        if is_rgb:
            # Process VI images (RGB)
            _, vi_imgs, _, _ = data
            # Only need the VI images
            data = vi_imgs
        else:
            # Process IR images (single channel)
            ir_imgs, _, _, _ = data
            # Only need the IR images
            data = ir_imgs

        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
        num_pixels += data.shape[0] * data.shape[2] * data.shape[3]  # batch_size * height * width
        
        # For a quicker estimate, you can comment out the break
        # break  # uncomment to test with just one batch

    mean = channels_sum / num_batches
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean.tolist(), std.tolist()


def main():
    # Define paths and parameters
    data_path = './'  # or whatever your dataset root is
    batch_size = 16  # larger batch size for faster computation
    num_workers = 4  # adjust according to your CPU

    # Simple transforms that only convert to tensor without normalization
    to_tensor = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create a dummy transform that applies ToTensor only
    # This avoids any normalization which would affect our stats
    class SimpleTransform:
        def __init__(self):
            self.transform = transforms.ToTensor()
            
        def __call__(self, img, mask=None):
            img_tensor = self.transform(img)
            if mask is not None:
                if isinstance(mask, torch.Tensor):
                    return img_tensor, mask
                mask_tensor = torch.from_numpy(np.array(mask)).long()
                return img_tensor, mask_tensor
            return img_tensor

    # Create datasets and dataloaders for both training and test sets
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
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=test_dataset.collate_fn
    )
    
    # Compute stats for training set
    print("Computing statistics for the training set...")
    vi_train_mean, vi_train_std = compute_mean_std(train_loader, is_rgb=True)
    ir_train_mean, ir_train_std = compute_mean_std(train_loader, is_rgb=False)
    
    # Compute stats for test set
    print("\nComputing statistics for the test set...")
    vi_test_mean, vi_test_std = compute_mean_std(test_loader, is_rgb=True)
    ir_test_mean, ir_test_std = compute_mean_std(test_loader, is_rgb=False)
    
    # Print results
    print("\n=== Results ===")
    print("Training Set:")
    print(f"VI (RGB) Mean: {vi_train_mean}")
    print(f"VI (RGB) Std: {vi_train_std}")
    print(f"IR (Grayscale) Mean: {ir_train_mean}")
    print(f"IR (Grayscale) Std: {ir_train_std}")
    
    print("\nTest Set:")
    print(f"VI (RGB) Mean: {vi_test_mean}")
    print(f"VI (RGB) Std: {vi_test_std}")
    print(f"IR (Grayscale) Mean: {ir_test_mean}")
    print(f"IR (Grayscale) Std: {ir_test_std}")
    
    print("\n=== Combined (train+test) ===")
    # Compute combined stats (weighted average based on dataset sizes)
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    total_size = train_size + test_size
    
    vi_combined_mean = [(m1 * train_size + m2 * test_size) / total_size for m1, m2 in zip(vi_train_mean, vi_test_mean)]
    vi_combined_std = [(s1 * train_size + s2 * test_size) / total_size for s1, s2 in zip(vi_train_std, vi_test_std)]
    
    ir_combined_mean = [(m1 * train_size + m2 * test_size) / total_size for m1, m2 in zip(ir_train_mean, ir_test_mean)]
    ir_combined_std = [(s1 * train_size + s2 * test_size) / total_size for s1, s2 in zip(ir_train_std, ir_test_std)]
    
    print(f"VI (RGB) Mean: {vi_combined_mean}")
    print(f"VI (RGB) Std: {vi_combined_std}")
    print(f"IR (Grayscale) Mean: {ir_combined_mean}")
    print(f"IR (Grayscale) Std: {ir_combined_std}")
    
    # Save results to a file
    with open('potsdam_mean_std.txt', 'w') as f:
        f.write("=== Results ===\n")
        f.write("Training Set:\n")
        f.write(f"VI (RGB) Mean: {vi_train_mean}\n")
        f.write(f"VI (RGB) Std: {vi_train_std}\n")
        f.write(f"IR (Grayscale) Mean: {ir_train_mean}\n")
        f.write(f"IR (Grayscale) Std: {ir_train_std}\n")
        
        f.write("\nTest Set:\n")
        f.write(f"VI (RGB) Mean: {vi_test_mean}\n")
        f.write(f"VI (RGB) Std: {vi_test_std}\n")
        f.write(f"IR (Grayscale) Mean: {ir_test_mean}\n")
        f.write(f"IR (Grayscale) Std: {ir_test_std}\n")
        
        f.write("\n=== Combined (train+test) ===\n")
        f.write(f"VI (RGB) Mean: {vi_combined_mean}\n")
        f.write(f"VI (RGB) Std: {vi_combined_std}\n")
        f.write(f"IR (Grayscale) Mean: {ir_combined_mean}\n")
        f.write(f"IR (Grayscale) Std: {ir_combined_std}\n")
    
    print("\nResults saved to potsdam_mean_std.txt")
    
    # Print code to directly use in your script
    print("\n=== Code to copy into your script ===")
    print("# RGB mean and std for VI images")
    print(f"vi_mean = {tuple(vi_combined_mean)}")
    print(f"vi_std = {tuple(vi_combined_std)}")
    print("# Single channel mean and std for IR images")
    print(f"ir_mean = {tuple(ir_combined_mean)}")
    print(f"ir_std = {tuple(ir_combined_std)}")


if __name__ == "__main__":
    main() 