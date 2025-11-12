import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        p1 = self.pool(x1)
        x2 = self.down2(p1)
        p2 = self.pool(x2)
        x3 = self.down3(p2)
        p3 = self.pool(x3)
        x4 = self.down4(p3)
        p4 = self.pool(x4)

        b = self.bottleneck(p4)

        u4 = self.upconv4(b)
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.up4(u4)

        u3 = self.upconv3(u4)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.up3(u3)

        u2 = self.upconv2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up2(u2)

        u1 = self.upconv1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.up1(u1)

        out = self.final_conv(u1)
        return out
