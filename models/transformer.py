import torch
import torch.nn as nn

from .rms_norm import RMSNorm
from .film import FiLM


class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.film1 = FiLM(dim)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = RMSNorm(dim)
        self.film2 = FiLM(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, cond):
        B, N, D = x.shape
        x_in = x

        x = self.norm1(x)
        x = self.film1(x, cond)

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = torch.bmm(Q, K.transpose(1, 2))
        scores = scores * (D**-0.5)
        attn = scores.softmax(dim=-1)
        out = torch.bmm(attn, V)
        out = self.proj(out)

        x = x_in + out
        x_in = x

        x = self.norm2(x)
        x = self.film2(x, cond)
        x = self.mlp(x)
        return x_in + x
