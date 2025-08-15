import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, n_classes=3):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = ConvBlock(128, 256)

        self.up1 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.dec1 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.out = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        d1 = self.up1(e3)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        out = self.out(d2).squeeze(-1)
        return self.fc(out)