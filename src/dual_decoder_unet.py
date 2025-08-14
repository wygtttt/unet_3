from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x_source = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        
        return self.sigmoid(x) * x_source + x_source


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
        factor = 2 if bilinear else 1

        # IR Encoder (1 channel input)
        self.ir_in_conv = DoubleConv(1, base_c)
        self.ir_down1 = Down(base_c, base_c * 2)
        self.ir_down2 = Down(base_c * 2, base_c * 4)
        self.ir_down3 = Down(base_c * 4, base_c * 8)
        self.ir_down4 = Down(base_c * 8, base_c * 16 // factor)

        # VI Encoder (1 channel input - Y channel only)
        self.vi_in_conv = DoubleConv(1, base_c)
        self.vi_down1 = Down(base_c, base_c * 2)
        self.vi_down2 = Down(base_c * 2, base_c * 4)
        self.vi_down3 = Down(base_c * 4, base_c * 8)
        self.vi_down4 = Down(base_c * 8, base_c * 16 // factor)

        # Spatial Attention modules for each encoding level
        self.ir_attention1 = SpatialAttention()
        self.ir_attention2 = SpatialAttention()
        self.ir_attention3 = SpatialAttention()
        self.ir_attention4 = SpatialAttention()
        self.ir_attention5 = SpatialAttention()
        
        self.vi_attention1 = SpatialAttention()
        self.vi_attention2 = SpatialAttention()
        self.vi_attention3 = SpatialAttention()
        self.vi_attention4 = SpatialAttention()
        self.vi_attention5 = SpatialAttention()

        # IR Segmentation Decoder
        self.ir_seg_up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.ir_seg_up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.ir_seg_up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.ir_seg_up4 = Up(base_c * 2, base_c, bilinear)
        self.ir_seg_out_conv = OutConv(base_c, num_classes)

        # IR Reconstruction Decoder
        self.ir_recon_up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.ir_recon_up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.ir_recon_up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.ir_recon_up4 = Up(base_c * 2, base_c, bilinear)
        self.ir_recon_out_conv = OutConv(base_c, 1)  # Reconstruct IR image

        # VI Segmentation Decoder
        self.vi_seg_up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.vi_seg_up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.vi_seg_up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.vi_seg_up4 = Up(base_c * 2, base_c, bilinear)
        self.vi_seg_out_conv = OutConv(base_c, num_classes)

        # VI Reconstruction Decoder
        self.vi_recon_up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.vi_recon_up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.vi_recon_up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.vi_recon_up4 = Up(base_c * 2, base_c, bilinear)
        self.vi_recon_out_conv = OutConv(base_c, 1)  # Reconstruct VI image
        
        # Fusion Decoder (using concatenated features)
        self.fusion_up1 = Up(base_c * 32, base_c * 16 // factor, bilinear)  # Double channels due to concatenation
        self.fusion_up2 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.fusion_up3 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.fusion_up4 = Up(base_c * 4, base_c * 2, bilinear)
        self.fusion_out_conv = OutConv(base_c * 2, 1)  # Output fused image

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> Dict[str, torch.Tensor]:
        # IR Encoder with spatial attention
        ir_x1 = self.ir_in_conv(ir)
        ir_x2 = self.ir_down1(ir_x1)
        ir_x3 = self.ir_down2(ir_x2)
        ir_x4 = self.ir_down3(ir_x3)
        ir_x5 = self.ir_down4(ir_x4)
        
        # VI Encoder with spatial attention
        vi_x1 = self.vi_in_conv(vi)
        vi_x2 = self.vi_down1(vi_x1)
        vi_x3 = self.vi_down2(vi_x2)
        vi_x4 = self.vi_down3(vi_x3)
        vi_x5 = self.vi_down4(vi_x4)
        
        # Apply spatial attention to IR and VI features at each level
        ir_att_x1 = self.ir_attention1(ir_x1)
        ir_att_x2 = self.ir_attention2(ir_x2)
        ir_att_x3 = self.ir_attention3(ir_x3)
        ir_att_x4 = self.ir_attention4(ir_x4)
        ir_att_x5 = self.ir_attention5(ir_x5)
        
        vi_att_x1 = self.vi_attention1(vi_x1)
        vi_att_x2 = self.vi_attention2(vi_x2)
        vi_att_x3 = self.vi_attention3(vi_x3)
        vi_att_x4 = self.vi_attention4(vi_x4)
        vi_att_x5 = self.vi_attention5(vi_x5)
        
        # Concatenate IR and VI features for fusion
        fusion_x1 = torch.cat([ir_att_x1, vi_att_x1], dim=1)
        fusion_x2 = torch.cat([ir_att_x2, vi_att_x2], dim=1)
        fusion_x3 = torch.cat([ir_att_x3, vi_att_x3], dim=1)
        fusion_x4 = torch.cat([ir_att_x4, vi_att_x4], dim=1)
        fusion_x5 = torch.cat([ir_att_x5, vi_att_x5], dim=1)
        
        # IR Segmentation Decoder
        ir_seg_x = self.ir_seg_up1(ir_x5, ir_x4)
        ir_seg_x = self.ir_seg_up2(ir_seg_x, ir_x3)
        ir_seg_x = self.ir_seg_up3(ir_seg_x, ir_x2)
        ir_seg_x = self.ir_seg_up4(ir_seg_x, ir_x1)
        ir_seg_logits = self.ir_seg_out_conv(ir_seg_x)
        
        # IR Reconstruction Decoder
        ir_recon_x = self.ir_recon_up1(ir_x5, ir_x4)
        ir_recon_x = self.ir_recon_up2(ir_recon_x, ir_x3)
        ir_recon_x = self.ir_recon_up3(ir_recon_x, ir_x2)
        ir_recon_x = self.ir_recon_up4(ir_recon_x, ir_x1)
        ir_recon_out = self.ir_recon_out_conv(ir_recon_x)
        
        # VI Segmentation Decoder
        vi_seg_x = self.vi_seg_up1(vi_x5, vi_x4)
        vi_seg_x = self.vi_seg_up2(vi_seg_x, vi_x3)
        vi_seg_x = self.vi_seg_up3(vi_seg_x, vi_x2)
        vi_seg_x = self.vi_seg_up4(vi_seg_x, vi_x1)
        vi_seg_logits = self.vi_seg_out_conv(vi_seg_x)
        
        # VI Reconstruction Decoder
        vi_recon_x = self.vi_recon_up1(vi_x5, vi_x4)
        vi_recon_x = self.vi_recon_up2(vi_recon_x, vi_x3)
        vi_recon_x = self.vi_recon_up3(vi_recon_x, vi_x2)
        vi_recon_x = self.vi_recon_up4(vi_recon_x, vi_x1)
        vi_recon_out = self.vi_recon_out_conv(vi_recon_x)
        
        # Fusion Decoder using concatenated attention-weighted features
        fusion_x = self.fusion_up1(fusion_x5, fusion_x4)
        fusion_x = self.fusion_up2(fusion_x, fusion_x3)
        fusion_x = self.fusion_up3(fusion_x, fusion_x2)
        fusion_x = self.fusion_up4(fusion_x, fusion_x1)
        fusion_out = self.fusion_out_conv(fusion_x)
        
        return {
            "ir_seg": ir_seg_logits,
            "ir_recon": ir_recon_out,
            "vi_seg": vi_seg_logits,
            "vi_recon": vi_recon_out,
            "fusion": fusion_out
        }