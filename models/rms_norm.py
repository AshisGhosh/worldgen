import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # sqrt((1/D * x^2) + eps)
        # [B, N, D]
        x_squared = x.pow(2)
        # [B, N, D] -> [B, N]
        x_mean_squared = x_squared.mean(-1)
        # [B, N]
        rms = (x_mean_squared + self.eps).sqrt()
        # [B, N] -> [B, N, 1]
        rms = rms.unsqueeze(-1)
        # [B, N, D] / [B, N, 1]
        x_norm = x / rms
        return x_norm * self.weight
