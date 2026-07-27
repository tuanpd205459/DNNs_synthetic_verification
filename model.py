import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => GroupNorm => ReLU) * 2
    GroupNorm is fully FP16-compatible under AMP (unlike BatchNorm2d which
    forces FP32 internally, breaking the FP16 pipeline).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # num_groups=8: works for any out_channels divisible by 8 (64,128,256,512)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard / Basic U-Net Architecture (Ronneberger et al.)
    Input:  [N, in_channels, H, W] (Stacked holograms H1, H2)
    Output: [N, out_channels, H, W] (Unit-normalized continuous representation: [sin(phi), cos(phi)])
    """
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # Decoder (Expanding Path)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = DoubleConv(512 + 256, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = DoubleConv(256 + 128, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = DoubleConv(128 + 64, 64)

        # Final 1x1 Convolution
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder with Skip Connections
        x = self.conv_up1(torch.cat([self.up1(x4), x3], dim=1))
        x = self.conv_up2(torch.cat([self.up2(x), x2], dim=1))
        x = self.conv_up3(torch.cat([self.up3(x), x1], dim=1))

        logits = self.outc(x)

        # Manual unit-normalization on S^1: sin²+cos²=1
        # Uses basic arithmetic (stays FP16 under AMP, unlike F.normalize which is FP32-only)
        norm = torch.sqrt((logits * logits).sum(dim=1, keepdim=True) + 1e-8)
        return logits / norm


# Alias for backward compatibility with PhaseUNet naming in training scripts
PhaseUNet = UNet
