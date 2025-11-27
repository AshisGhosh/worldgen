import torch.nn as nn


class AdaptiveLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, cond):
        # [B, N, D]
        x = self.norm(x)
        # [B, D]
        scale = 1 + self.gamma(cond)
        shift = self.beta(cond)
        # [B, N, D]
        return x * scale.unsqueeze(1) + shift.unsqueeze(1)
