import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURES = (32, 64, 128, 256, 512)


class ConvBlock(nn.Module):
    """(conv3x3 -> InstanceNorm -> LeakyReLU) x 2."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            concat_channels = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            concat_channels = in_channels // 2 + skip_channels
        self.conv = ConvBlock(concat_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, n_channels=1, features=FEATURES):
        super().__init__()
        self.inc = ConvBlock(n_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[4])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]


class Decoder(nn.Module):
    """U-Net decoder. In training mode with deep supervision it returns
    [logits, ds1, ds2, ds3, ds4] (all at full resolution), otherwise logits only."""

    def __init__(self, n_classes=2, bilinear=False, deep_supervision=True, features=FEATURES):
        super().__init__()
        self.deep_supervision = deep_supervision
        factor = 2 if bilinear else 1
        self.up1 = Up(features[4], features[3], features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], features[0], bilinear)
        self.outc = OutConv(features[0], n_classes)
        if deep_supervision:
            self.deep_outputs = nn.ModuleList([
                nn.Conv2d(features[1], n_classes, kernel_size=1),
                nn.Conv2d(features[2], n_classes, kernel_size=1),
                nn.Conv2d(features[3], n_classes, kernel_size=1),
                nn.Conv2d(features[4], n_classes, kernel_size=1),
            ])

    def forward(self, features):
        x1, x2, x3, x4, x5 = features
        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)
        logits = self.outc(d1)
        if self.deep_supervision and self.training:
            size = x1.shape[2:]
            ds = [F.interpolate(head(feat), size=size, mode='bilinear', align_corners=True)
                  for head, feat in zip(self.deep_outputs, (d2, d3, d4, x5))]
            return [logits] + ds
        return logits


class TemporalDecoder(Decoder):
    """Decoder applied independently to every frame of a feature sequence."""

    def forward_sequence(self, features):
        T = features['x1'].shape[1]
        outputs = [self.forward([features[k][:, t] for k in ('x1', 'x2', 'x3', 'x4', 'x5')]) for t in range(T)]
        if isinstance(outputs[0], list):
            return [torch.stack([out[i] for out in outputs], dim=1) for i in range(len(outputs[0]))]
        return torch.stack(outputs, dim=1)

    def forward_key_frame(self, features):
        return self.forward([features[k][:, -1] for k in ('x1', 'x2', 'x3', 'x4', 'x5')])
