import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.inference_mode()
def get_inception_features(x, inception, device="cuda"):
    """
    x: [B, 3, H, W] in [0, 1] or [0, 255]
    returns: [B, D] feature vectors
    """
    x = x.to(device)

    # Scale to [0, 1]
    if x.max() > 1.0:
        x = x / 255.0

    # Inception expects 299x299
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

    # Inception V3 expects training=False, aux_logits=False for inference
    feats = inception(x)  # [B, 2048] because we set fc = Identity()
    return feats


def load_inception(device="cuda"):
    # For newer torchvision:
    from torchvision.models import inception_v3, Inception_V3_Weights

    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

    # Replace final FC with identity so output = penultimate features
    inception.fc = nn.Identity()

    inception.to(device)
    inception.eval()
    for p in inception.parameters():
        p.requires_grad = False
    return inception


@torch.inference_mode()
def polynomial_mmd_ksd(x, y, degree=3, gamma=None, coef0=1.0):
    """
    x: [N, D]
    y: [M, D]
    Polynomial kernel k(x, y) = (gamma * x·y + coef0)^degree
    Returns unbiased MMD^2 estimate (Kernel Inception Distance core).
    """
    N, D = x.shape
    M, _ = y.shape

    if gamma is None:
        gamma = 1.0 / D

    # Compute Gram matrices
    K_xx = (gamma * x @ x.T + coef0) ** degree
    K_yy = (gamma * y @ y.T + coef0) ** degree
    K_xy = (gamma * x @ y.T + coef0) ** degree

    # Remove diagonal for unbiased estimate
    # (sum of off-diagonal elements only)
    sum_xx = K_xx.sum() - K_xx.diag().sum()
    sum_yy = K_yy.sum() - K_yy.diag().sum()

    mmd_xx = sum_xx / (N * (N - 1))
    mmd_yy = sum_yy / (M * (M - 1))
    mmd_xy = K_xy.mean()

    mmd2 = mmd_xx + mmd_yy - 2 * mmd_xy
    return mmd2


@torch.inference_mode()
def compute_kid_from_tensors(real_imgs, fake_imgs, inception=None, device="cuda"):
    """
    real_imgs: [N, 3, H, W]
    fake_imgs: [N, 3, H, W]  (ideally N the same, but they don't *have* to be)
    Returns: scalar KID (MMD^2 in feature space)
    """
    assert (
        real_imgs.dim() == 4 and fake_imgs.dim() == 4
    ), f"Expect [N, 3, H, W] tensors, got {real_imgs.shape} and {fake_imgs.shape}"

    if inception is None:
        inception = load_inception(device=device)

    real_feats = []
    fake_feats = []

    B = 64  # batch size for feature extraction
    for i in range(0, real_imgs.size(0), B):
        real_batch = real_imgs[i : i + B]
        real_feats.append(get_inception_features(real_batch, inception, device))

    for i in range(0, fake_imgs.size(0), B):
        fake_batch = fake_imgs[i : i + B]
        fake_feats.append(get_inception_features(fake_batch, inception, device))

    real_feats = torch.cat(real_feats, dim=0)
    fake_feats = torch.cat(fake_feats, dim=0)

    kid = polynomial_mmd_ksd(real_feats, fake_feats)
    return kid.item()
