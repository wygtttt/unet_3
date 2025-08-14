import os
from PIL import Image
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class PostdamDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None, ir_transforms=None):
        super(PostdamDataset, self).__init__()
        self.flag = "train" if train else "test"
        data_root = os.path.join(root, "postdam")
        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        self.transforms = transforms  # For VI images (RGB) and masks
        self.ir_transforms = ir_transforms  # For IR images (single channel)
        
        # Define paths to the directory
        self.ir_dir = os.path.join(data_root, f"ir_{self.flag}")
        self.vi_dir = os.path.join(data_root, f"vi_{self.flag}")
        self.mask_dir = os.path.join(data_root, f"mask_{self.flag}")
        
        # Check if directories exist
        assert os.path.exists(self.ir_dir), f"path '{self.ir_dir}' does not exist."
        assert os.path.exists(self.vi_dir), f"path '{self.vi_dir}' does not exist."
        assert os.path.exists(self.mask_dir), f"path '{self.mask_dir}' does not exist."
        
        # Get image names with their extensions
        ir_img_names = sorted([i for i in os.listdir(self.ir_dir) if i.endswith((".tif", ".png", ".jpg", ".jpeg"))])
        vi_img_names = sorted([i for i in os.listdir(self.vi_dir) if i.endswith((".tif", ".png", ".jpg", ".jpeg"))])
        mask_img_names = sorted([i for i in os.listdir(self.mask_dir) if i.endswith((".tif", ".png", ".jpg", ".jpeg"))])
        
        # Find common image names across all directories
        # Remove extensions to compare base names
        ir_base_names = set([os.path.splitext(img)[0] for img in ir_img_names])
        vi_base_names = set([os.path.splitext(img)[0] for img in vi_img_names])
        mask_base_names = set([os.path.splitext(img)[0] for img in mask_img_names])
        
        # Find the intersection of all three sets
        common_base_names = sorted(list(ir_base_names.intersection(vi_base_names).intersection(mask_base_names)))
        
        print(f"Found {len(common_base_names)} common images across IR, VI, and mask directories")
        print(f"Original counts - IR: {len(ir_img_names)}, VI: {len(vi_img_names)}, Mask: {len(mask_img_names)}")
        
        # Ensure we have at least some common images
        assert len(common_base_names) > 0, "No common images found across all three directories"
        
        # Create mapping from base names back to full filenames with extensions
        ir_name_mapping = {os.path.splitext(name)[0]: name for name in ir_img_names}
        vi_name_mapping = {os.path.splitext(name)[0]: name for name in vi_img_names}
        mask_name_mapping = {os.path.splitext(name)[0]: name for name in mask_img_names}
        
        # Create full paths for each common image
        self.ir_img_list = [os.path.join(self.ir_dir, ir_name_mapping[base_name]) for base_name in common_base_names]
        self.vi_img_list = [os.path.join(self.vi_dir, vi_name_mapping[base_name]) for base_name in common_base_names]
        self.mask_img_list = [os.path.join(self.mask_dir, mask_name_mapping[base_name]) for base_name in common_base_names]

    def __getitem__(self, idx):
        # Load IR image (single channel)
        ir_img = Image.open(self.ir_img_list[idx]).convert('L')
        
        # Load VI image and extract Y channel (luminance) from YCbCr
        vi_img = Image.open(self.vi_img_list[idx]).convert('RGB')
        vi_img = vi_img.convert('YCbCr')
        # Extract only Y channel (luminance)
        vi_img = vi_img.split()[0]  # Get Y channel only
        
        # Load mask - 直接加载，保持原始类别值（0-5）
        mask = Image.open(self.mask_img_list[idx]).convert('L')
        
        # Apply transforms if available
        if self.transforms is not None:
            # Apply transforms to VI image and mask
            vi_img, mask = self.transforms(vi_img, mask)
            
            # Apply IR-specific transforms to IR image if available
            if self.ir_transforms is not None:
                # Create a dummy mask for IR since transforms expect an image and mask pair
                dummy_mask = Image.fromarray(np.zeros_like(np.array(ir_img)))
                ir_img, _ = self.ir_transforms(ir_img, dummy_mask)
            else:
                # If no specific IR transforms provided, convert to tensor without normalization
                from torchvision import transforms
                ir_transform = transforms.Compose([transforms.ToTensor()])
                ir_img = ir_transform(ir_img)
        else:
            # Convert to tensors
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            vi_img = to_tensor(vi_img)
            ir_img = to_tensor(ir_img)
            # 将mask转换为长整型张量，不需要除以255
            mask = torch.from_numpy(np.array(mask)).long()
        
        # Create fusion target: max(ir_img, vi_mean)
        vi_mean = torch.mean(vi_img, dim=0, keepdim=True)  # Average the RGB channels to get a single channel
        
        # Make sure IR image has the right shape for element-wise operations (1, H, W)
        if len(ir_img.shape) == 2:
            ir_img = ir_img.unsqueeze(0)
        
        fusion_target = torch.maximum(ir_img, vi_mean)
        
        return ir_img, vi_img, mask, fusion_target

    def __len__(self):
        return len(self.ir_img_list)

    @staticmethod
    def collate_fn(batch):
        ir_images, vi_images, masks, fusion_targets = list(zip(*batch))
        
        batched_ir_imgs = cat_list(ir_images, fill_value=0)
        batched_vi_imgs = cat_list(vi_images, fill_value=0)
        batched_masks = cat_list(masks, fill_value=255)
        batched_fusion_targets = cat_list(fusion_targets, fill_value=0)
        
        return batched_ir_imgs, batched_vi_imgs, batched_masks, batched_fusion_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs