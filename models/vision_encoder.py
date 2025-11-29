import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)


class VisionEncoder(nn.Module):
    def __init__(self, dim, img_size, channels):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(channels, 32),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),
        )
        # 64 -> 32 -> 16 -> 8
        self.proj = nn.Linear((img_size // 2 // 2 // 2) ** 2 * 128, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.cnn(x)
        # [B, C, H, W] -> [B, C * H * W]
        x = x.view(B, -1)
        return self.proj(x)
