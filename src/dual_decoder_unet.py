from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class DualDecoderUNet(nn.Module):
    def __init__(self,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(DualDecoderUNet, self).__init__()
        self.num_classes = num_classes
        self.bilinear = bilinear

        # Encoder for combined IR (1 channel) + VI (3 channels) = 4 channels
        self.in_conv = DoubleConv(4, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        # Segmentation Decoder
        self.seg_up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.seg_up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.seg_up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.seg_up4 = Up(base_c * 2, base_c, bilinear)
        self.seg_out_conv = OutConv(base_c, num_classes)

        # Fusion Decoder
        self.fusion_up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.fusion_up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.fusion_up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.fusion_up4 = Up(base_c * 2, base_c, bilinear)
        self.fusion_out_conv = OutConv(base_c, 1)  # Output is a single-channel fused image

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Concatenate IR and VI along channel dimension
        x = torch.cat([ir, vi], dim=1)
        
        # Encoder
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Segmentation Decoder
        seg_x = self.seg_up1(x5, x4)
        seg_x = self.seg_up2(seg_x, x3)
        seg_x = self.seg_up3(seg_x, x2)
        seg_x = self.seg_up4(seg_x, x1)
        seg_logits = self.seg_out_conv(seg_x)
        
        # Fusion Decoder
        fusion_x = self.fusion_up1(x5, x4)
        fusion_x = self.fusion_up2(fusion_x, x3)
        fusion_x = self.fusion_up3(fusion_x, x2)
        fusion_x = self.fusion_up4(fusion_x, x1)
        fusion_out = self.fusion_out_conv(fusion_x)
        #fusion_out = torch.sigmoid(fusion_out)  # Normalize fusion output to [0, 1]
        
        return {
            "seg": seg_logits,
            "fusion": fusion_out
        } 