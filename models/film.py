import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)

    def forward(self, x, cond):
        # [B, D]
        scale = 1 + self.gamma(cond)
        shift = self.beta(cond)
        # [B, N, D] * [B, 1, D] + [B, 1, D]
        return x * scale.unsqueeze(1) + shift.unsqueeze(1)
