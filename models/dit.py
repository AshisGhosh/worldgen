import torch
import torch.nn as nn

from .transformer import TransformerBlock
from .film import FiLM


def patchify(x, patch_size):
    B, C, H, W = x.shape
    P = patch_size
    h = H // P
    w = W // P
    x = x.reshape(B, C, h, P, w, P)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, h * w, P * P * C)
    return x


def unpatchify(x, patch_size, patches_per_side, img_size, channels):
    B, N, _ = x.shape
    h = w = patches_per_side
    H = W = img_size
    P = patch_size
    C = channels

    x = x.reshape(B, h, w, P, P, C)
    x = x.permute(0, 5, 1, 3, 2, 4)
    x = x.reshape(B, C, H, W)
    return x


class DiT(nn.Module):
    def __init__(self, dim=128, depth=3, patch_size=4, img_size=64, img_channels=3):
        super().__init__()
        assert (
            img_size % patch_size == 0
        ), f"Image size {img_size} must be divisible by patch size {patch_size}"
        self.dim = dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_per_side = self.img_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.img_channels = img_channels

        # [1, N, D]
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, dim))

        # [1, D]
        self.time_emb = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        # [B, D]
        self.film = FiLM(dim)

        # proj [B, N, C * P * P] -> [B, N, D]
        self.proj = nn.Linear(img_channels * self.patch_size * self.patch_size, dim)

        # attn
        self.attn_blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])

        # unproj
        self.unproj = nn.Linear(dim, img_channels * self.patch_size * self.patch_size)

    def forward(self, x, t):
        B, C, H, W = x.shape

        # [B, C, H, W] -> [B, N, C * P * P]
        x = patchify(x, self.patch_size)

        # [B, N, C * P * P] -> [B, N, D]
        x = self.proj(x)

        # [B, N, D] + [1, N, D] -> [B, N, D]
        x = x + self.pos_emb

        # [B] -> [B, 1]
        if t.dim() == 1:
            t = t.unsqueeze(1)

        # [B, 1] -> [B, D]
        t = self.time_emb(t)

        # [B, N, D], [B, D] -> [B, N, D]
        x = self.film(x, t)

        for block in self.attn_blocks:
            x = block(x)

        # [B, N, D] -> [B, N, C * P * P]
        x = self.unproj(x)

        # [B, N, C * P * P] -> [B, C, H, W]
        x = unpatchify(
            x,
            self.patch_size,
            self.num_patches_per_side,
            self.img_size,
            self.img_channels,
        )
        return x
