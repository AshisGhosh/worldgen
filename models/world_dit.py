import torch
import torch.nn as nn
from .dit import patchify, unpatchify
from .transformer import TransformerBlock
from .siglip import SigLIPEncoder  # noqa: F401
from .vision_encoder import VisionEncoder


class WorldDiT(nn.Module):
    def __init__(
        self, dim=256, depth=3, img_size=64, patch_size=4, channels=3, enable_cfg=False
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.patchify = patchify
        self.channels = channels
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.num_patches_per_side = self.img_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2

        self.proj = nn.Linear(channels * self.patch_size * self.patch_size, dim)

        # [1, N, D]
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim))

        # [B, 1] -> [B, D]
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

        # [B, 1] -> [B, D]
        self.action_embed = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

        # self.vision_embed = SigLIPEncoder(dim, freeze=True)
        self.vision_embed = VisionEncoder(dim, self.img_size, self.channels)

        self.cond_proj = nn.Linear(dim * 3, dim)

        self.enable_cfg = enable_cfg
        self.cond_dropout = 0.1
        if self.enable_cfg:
            self.null_start = nn.Parameter(torch.zeros(dim))
            self.null_action = nn.Parameter(torch.zeros(dim))

        self.attn_blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])

        self.unproj = nn.Linear(dim, channels * self.patch_size * self.patch_size)
        self.unpatchify = unpatchify

    def forward(self, x, t, starts, actions, cfg=False):
        """
        x: noised version of the end state [B, C, H, W]
        t: time conditioning of the step [B]
        starts: start conditioning [B, C, H, W]
        actions: action conditioning [B, 1]
        """
        B, C, H, W = x.shape

        # [B, C, H, W] -> [B, N, C * P * P]
        x = self.patchify(x, self.patch_size)

        # [B, N, C * P * P] -> [B, N, D]
        x = self.proj(x)

        # [B, N, D] + [1, N, D] -> [B, N, D]
        x = x + self.pos_embed

        # [B] -> [B, 1]
        t = t.unsqueeze(1)
        # [B, 1] -> [B, D]
        t = self.time_embed(t)

        # [B, C, H, W] -> [B, D]
        starts = self.vision_embed(starts)

        # [B] -> [B, 1]
        actions = actions.unsqueeze(1)
        # [B, 1] -> [B, D]
        actions = self.action_embed(actions)

        if self.enable_cfg:
            if self.training and self.cond_dropout > 0:
                # [B] -> [B, 1]
                drop_mask = (
                    torch.rand((B,), device=x.device).unsqueeze(-1) < self.cond_dropout
                )
                null_start = self.null_start.unsqueeze(0).expand_as(starts)  # [B, D]
                null_action = self.null_action.unsqueeze(0).expand_as(actions)  # [B, D]
                starts = torch.where(drop_mask, null_start, starts)
                actions = torch.where(drop_mask, null_action, actions)
            elif cfg:
                null_start = self.null_start.unsqueeze(0).expand_as(starts)  # [B, D]
                null_action = self.null_action.unsqueeze(0).expand_as(actions)  # [B, D]
                starts = null_start
                actions = null_action

        # [B, D], [B, D], [B, D] -> [B, 3 * D]
        cond = torch.cat([t, starts, actions], dim=-1)

        # [B, 3 * D] -> [B, D]
        cond = self.cond_proj(cond)

        # [B, N, D], [B, D] -> [B, N, D]
        for block in self.attn_blocks:
            x = block(x, cond)

        # [B, N, D] -> [B, N, C * P * P]
        x = self.unproj(x)

        # [B, N, C * P * P] -> [B, C, H, W]
        x = self.unpatchify(
            x, self.patch_size, self.num_patches_per_side, self.img_size, self.channels
        )

        return x
