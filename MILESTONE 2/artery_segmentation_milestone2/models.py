import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inter_ch, 1, 1), nn.Sigmoid())

    def forward(self, gate, skip):
        alpha = self.psi(self.gate(gate) + self.skip(skip))
        return skip * alpha


class UNet(nn.Module):
    def __init__(self, attention=False, in_channels=1, out_channels=1, base=16):
        super().__init__()
        self.attention = attention
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        if attention:
            self.att3 = AttentionGate(base * 4, base * 4, base * 2)
            self.att2 = AttentionGate(base * 2, base * 2, base)
            self.att1 = AttentionGate(base, base, max(base // 2, 1))
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        s3 = self.att3(d3, e3) if self.attention else e3
        d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = self.up2(d3)
        s2 = self.att2(d2, e2) if self.attention else e2
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = self.up1(d2)
        s1 = self.att1(d1, e1) if self.attention else e1
        d1 = self.dec1(torch.cat([d1, s1], dim=1))
        return self.out(d1)


def build_model(name):
    if name == "unet":
        return UNet(attention=False)
    if name == "attention_unet":
        return UNet(attention=True)
    raise ValueError(f"Unknown model: {name}")
