import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gaussian(window_size, sigma):
    """
    Generates a 1D Gaussian kernel.
    """
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    """
    Creates a 2D Gaussian kernel.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    """
    Calculates the SSIM index between img1 and img2.
    """
    # Value range can be different from 255. Other typical ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    else:
        # Ensure window is on the same device as the input
        window = window.to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Register buffer to not optimize
        self.register_buffer('window', create_window(window_size))

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.window.size()[0]:
            window = self.window.to(img1.device)
        else:
            window = create_window(self.window_size, channel).to(img1.device)
            # Update the buffer with the new window
            self.window = window.to(self.window.device)

        return ssim(img1, img2, window=window, window_size=self.window_size,
                    size_average=self.size_average, val_range=self.val_range)


class FusionLoss(nn.Module):
    def __init__(self, ssim_weight=0.5, l1_weight=0.5):
        """
        Fusion loss combining SSIM and L1 loss
        Args:
            ssim_weight: Weight for SSIM loss (higher values focus more on structural similarity)
            l1_weight: Weight for L1 loss (higher values focus more on absolute pixel differences)
        """
        super(FusionLoss, self).__init__()
        self.ssim_loss = SSIM(window_size=11, size_average=True)
        self.l1_loss = nn.L1Loss()
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight

    def forward(self, pred, target):
        """
        Calculate the fusion loss
        Args:
            pred: Predicted fusion image from the model
            target: Target fusion image (max of IR and VI)
        Returns:
            Weighted loss combining SSIM and L1 loss
        """
        ssim_value = self.ssim_loss(pred, target)
        ssim_loss = 1 - ssim_value  # Convert SSIM to a loss (0 is perfect, 1 is bad)
        l1_loss = self.l1_loss(pred, target)
        
        # Combine losses with weights
        return self.ssim_weight * ssim_loss + self.l1_weight * l1_loss 