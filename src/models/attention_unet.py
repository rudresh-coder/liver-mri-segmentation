# src/models/attention_unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Building Blocks --------

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling: MaxPool + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.pool_conv(x)


class UpNoAttention(nn.Module):
    """Upscaling then DoubleConv (no attention branch)"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1: decoder input, x2: encoder skip (already attention-refined)
        x1 = self.up(x1)

        # match sizes
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    """
    Attention Gate as in Attention U-Net:
    - g: gating (decoder feature)
    - x: skip connection (encoder feature)
    Output: attention-weighted x
    """
    def __init__(self, in_g, in_x, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_g, inter_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_ch)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_x, inter_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_ch)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: decoder feature map (coarser, gating signal)
        x: encoder feature map (skip connection)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # upsample g1 to x1 size if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # apply attention map
        

# -------- Attention U-Net Model --------

class AttentionUNet(nn.Module):
    """
    Attention U-Net for 1-channel input and 1-channel output.
    Uses standard U-Net encoder depth with attention gates on skip connections.
    """
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Attention blocks
        self.att1 = AttentionBlock(in_g=512,          in_x=512,          inter_ch=256)
        self.att2 = AttentionBlock(in_g=256,          in_x=256,          inter_ch=128)
        self.att3 = AttentionBlock(in_g=128,          in_x=128,          inter_ch=64)
        self.att4 = AttentionBlock(in_g=64,           in_x=64,           inter_ch=32)

        # Decoder (upsampling + conv, using attention-refined skips)
        self.up1 = UpNoAttention(1024, 512 // factor, bilinear)
        self.up2 = UpNoAttention(512, 256 // factor, bilinear)
        self.up3 = UpNoAttention(256, 128 // factor, bilinear)
        self.up4 = UpNoAttention(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)      # 64
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024//factor

        # Attention on skips
        g1 = x5
        x4_att = self.att1(g1, x4)

        g2 = x4_att
        x3_att = self.att2(g2, x3)

        g3 = x3_att
        x2_att = self.att3(g3, x2)

        g4 = x2_att
        x1_att = self.att4(g4, x1)

        # Decoder path with attention-refined skips
        x = self.up1(x5, x4_att)
        x = self.up2(x,  x3_att)
        x = self.up3(x,  x2_att)
        x = self.up4(x,  x1_att)

        logits = self.outc(x)
        return logits
