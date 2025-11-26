import torch
from tqdm import tqdm, trange
from typing import Optional
import os


def cosine_alpha_cumprod(T, s=0.008):
    t = torch.linspace(0, T, T + 1, device=device)
    f = torch.cos(((t / T) + s) / (1 + s) * torch.pi / 2) ** 2
    alpha_bar = f / f[0]
    return alpha_bar


def get_noise_schedule(T: int = 1000, device: str = "cpu"):
    alphas_bar = cosine_alpha_cumprod(T).to(device)
    one_minus_alphas_bar = 1 - alphas_bar
    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(one_minus_alphas_bar)

    noise_dict = {
        "T": T,
        "alphas_bar": alphas_bar,
        "one_minus_alphas_bar": one_minus_alphas_bar,
        "sqrt_alphas_bar": sqrt_alphas_bar,
        "sqrt_one_minus_alphas_bar": sqrt_one_minus_alphas_bar,
    }

    return noise_dict


device = "cuda" if torch.cuda.is_available() else "cpu"
T = 400
noise_schedule = get_noise_schedule(T=T, device=device)


def q_sample(x0, t):
    device = x0.device
    eps = torch.randn(x0.shape).to(device)
    sqrt_alphas_bar_t = noise_schedule["sqrt_alphas_bar"][t]
    # [B] -> [B, 1, 1, 1]
    sqrt_alphas_bar_t = sqrt_alphas_bar_t.view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_bar_t = noise_schedule["sqrt_one_minus_alphas_bar"][t]
    # [B] -> [B, 1, 1, 1]
    sqrt_one_minus_alphas_bar_t = sqrt_one_minus_alphas_bar_t.view(-1, 1, 1, 1)
    return sqrt_alphas_bar_t * x0 + sqrt_one_minus_alphas_bar_t * eps, eps


def train_diffusion(
    model,
    criterion,
    optimizer,
    dataloader,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ema: Optional[torch.nn.Module] = None,
    device: str = "cpu",
    num_epochs: int = 10,
    save_dir: str = "./checkpoints",
    save_freq: int = 10,
):
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    for i in trange(num_epochs):
        model.train()

        epoch_loss = 0.0

        for starts, _, _ in tqdm(dataloader):
            starts = starts.to(device).float()
            starts = (starts / 127.5) - 1

            t = torch.randint(0, T, (starts.shape[0],)).to(device)

            optimizer.zero_grad()

            xt, eps = q_sample(starts, t)

            t_float = t.float() / (T - 1)
            eps_pred = model(xt, t_float)

            loss = criterion(eps_pred, eps)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            if ema is not None:
                ema.update(model)

        if scheduler is not None:
            scheduler.step()

        epoch_loss /= len(dataloader)

        if (i + 1) % save_freq == 0:
            print(f"Epoch {i+1}/{num_epochs}, Loss: {epoch_loss}")
            torch.save(model.state_dict(), f"{save_dir}/model_{i+1}.pth")
