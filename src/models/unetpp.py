import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpSample(nn.Module):
    def __init__(self, scale=2, mode="bilinear"):
        super().__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x, size=None):
        if size is not None:
            return F.interpolate(x, size=size, mode=self.mode, align_corners=False)
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)


class UNetPP(nn.Module):

    def __init__(self, n_channels=1, n_classes=1, filters=(64, 128, 256, 512, 512), deep_supervision=False):
        super().__init__()
        assert len(filters) == 5, "filters must be a tuple of length 5"
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.filters = filters
        self.deep_supervision = deep_supervision

        f0, f1, f2, f3, f4 = filters

        # Encoder (same as U-Net)
        self.conv00 = ConvBlock(n_channels, f0)
        self.pool0 = nn.MaxPool2d(2)
        self.conv10 = ConvBlock(f0, f1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv20 = ConvBlock(f1, f2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv30 = ConvBlock(f2, f3)
        self.pool3 = nn.MaxPool2d(2)
        self.conv40 = ConvBlock(f3, f4)

        # Upsample operator
        self.up = UpSample()

        # Nested decoder blocks (UNet++)
        # Level 0
        self.conv01 = ConvBlock(f0 + f1, f0)
        self.conv02 = ConvBlock(f0*2 + f1, f0)
        self.conv03 = ConvBlock(f0*3 + f1, f0)
        self.conv04 = ConvBlock(f0*4 + f1, f0)

        # Level 1
        self.conv11 = ConvBlock(f1 + f2, f1)
        self.conv12 = ConvBlock(f1*2 + f2, f1)
        self.conv13 = ConvBlock(f1*3 + f2, f1)

        # Level 2
        self.conv21 = ConvBlock(f2 + f3, f2)
        self.conv22 = ConvBlock(f2*2 + f3, f2)

        # Level 3
        self.conv31 = ConvBlock(f3 + f4, f3)

        # Final output convs
        if self.deep_supervision:
            self.final1 = nn.Conv2d(f0, n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(f0, n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(f0, n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(f0, n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(f0, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x00 = self.conv00(x)           # level 0
        x10 = self.conv10(self.pool0(x00))
        x20 = self.conv20(self.pool1(x10))
        x30 = self.conv30(self.pool2(x20))
        x40 = self.conv40(self.pool3(x30))

        # Decode - nested skip connections
        # x01
        x01 = self.conv01(torch.cat([x00, self.up(x10, size=x00.shape[2:])], dim=1))

        # x11
        x11 = self.conv11(torch.cat([x10, self.up(x20, size=x10.shape[2:])], dim=1))

        # x02
        x02 = self.conv02(torch.cat([
            x00,
            x01,
            self.up(x11, size=x00.shape[2:])
        ], dim=1))

        # x21
        x21 = self.conv21(torch.cat([x20, self.up(x30, size=x20.shape[2:])], dim=1))

        # x12
        x12 = self.conv12(torch.cat([
            x10,
            x11,
            self.up(x21, size=x10.shape[2:])
        ], dim=1))

        # x03
        x03 = self.conv03(torch.cat([
            x00,
            x01,
            x02,
            self.up(x12, size=x00.shape[2:])
        ], dim=1))

        # x31
        x31 = self.conv31(torch.cat([x30, self.up(x40, size=x30.shape[2:])], dim=1))

        # x22
        x22 = self.conv22(torch.cat([
            x20,
            x21,
            self.up(x31, size=x20.shape[2:])
        ], dim=1))

        # x13
        x13 = self.conv13(torch.cat([
            x10,
            x11,
            x12,
            self.up(x22, size=x10.shape[2:])
        ], dim=1))

        # x04
        x04 = self.conv04(torch.cat([
            x00,
            x01,
            x02,
            x03,
            self.up(x13, size=x00.shape[2:])
        ], dim=1))

        if self.deep_supervision:
            out1 = self.final1(x01)
            out2 = self.final2(x02)
            out3 = self.final3(x03)
            out4 = self.final4(x04)
            # return list of outputs (for deep supervision loss computation)
            return [out1, out2, out3, out4]
        else:
            out = self.final(x04)
            return out


# Utility: parameter count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Self-test: create model, check forward pass
    device = torch.device('cpu')
    model = UNetPP(n_channels=1, n_classes=1, filters=(32, 64, 128, 256, 256), deep_supervision=False)
    print('UNet++ param count:', count_parameters(model))
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    print('Input shape:', x.shape)
    print('Output shape:', y.shape)
