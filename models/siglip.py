import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"


class SigLIPEncoder(nn.Module):
    def __init__(self, dim, model_name="google/siglip2-base-patch16-224", freeze=True):
        super().__init__()
        base_model = AutoModel.from_pretrained(model_name)
        self.vision = base_model.vision_model
        self.freeze = freeze
        if self.freeze:
            for p in self.vision.parameters():
                p.requires_grad = False

        siglip_dim = self.vision.config.hidden_size
        # [B, 768] -> [B, D]
        self.proj = nn.Linear(siglip_dim, dim)

        mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        B, C, H, W = x.shape

        # Resize to SigLIP resolution (224x224 for this model)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Normalize like CLIP/SigLIP: mean=0.5, std=0.5 -> [-1, 1]
        x = (x - self.mean) / self.std

        if self.freeze:
            with torch.no_grad():
                # [B, C, H, W] -> [B, N, siglipD]
                feats = self.vision(x)
        else:
            # [B, C, H, W] -> [B, N, siglipD]
            feats = self.vision(x)

        # [B, N, D] -> [B, siglipD]
        pooled = feats.pooler_output

        # [B, siglipD] -> [B, D]
        return self.proj(pooled)
