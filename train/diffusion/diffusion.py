import torch
from tqdm import tqdm, trange
from typing import Optional, Dict
import os

from .diffusion_sampler import ddim_sample
import aim
from aim import Run
import numpy as np

from ..metrics import compute_kid_from_tensors, load_inception
from ..utils import images_dict_to_grid

inception = load_inception()


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


def generate_snapshots(
    model, noise_schedule, steps_to_show=[300, 200, 100, 0]
) -> Dict[int, np.ndarray]:
    num_steps = 100

    # [B, C, H, W]
    xt = torch.randn(1, 3, 64, 64).to(device)

    ddim_snapshots = ddim_sample(xt, model, noise_schedule, num_steps, steps_to_show)
    return ddim_snapshots


def eval_diffusion(model, loader, noise_schedule, t_val=None):
    model.eval()
    with torch.inference_mode():
        imgs, _, _ = loader.dataset[0]
        x0 = (imgs.to(device) / 127.5) - 1
        x0 = x0.unsqueeze(0)

        B = x0.shape[0]
        if t_val is None:
            t = torch.randint(0, T, (B,), device=device)
        else:
            t = torch.full((B,), t_val, device=device)

        sqrt_alpha_cumprod = noise_schedule["sqrt_alphas_bar"]
        sqrt_one_minus_alpha_cumprod = noise_schedule["sqrt_one_minus_alphas_bar"]

        xt, eps = q_sample(x0, t)
        t_float = t.float() / (T - 1)
        with torch.autocast(device, torch.float16):
            eps_pred = model(xt, t_float)
        x0_hat = (
            xt - sqrt_one_minus_alpha_cumprod[t].view(B, 1, 1, 1) * eps_pred
        ) / sqrt_alpha_cumprod[t].view(B, 1, 1, 1)

        # unnormalize for visualization
        def denorm(x):
            x = x.clamp(-1, 1)
            x = (x + 1) / 2
            return x.squeeze(0)

        t_int = t.item()
        eval_dict = {
            "x0": denorm(x0),
            f"xt_{t_int}": denorm(xt),
            "x0_hat": denorm(x0_hat),
        }

    return eval_dict, eps_pred, eps


def sample_diffusion_batch(
    model,
    noise_schedule,
    num_samples: int,
    num_steps: int = 100,
    batch_size: int = 64,
    device: str = "cuda",
):
    """
    Generate num_samples images from the diffusion model using DDIM.
    Returns [N, 3, H, W] in [-1, 1].
    """
    model.eval()
    all_samples = []

    with torch.inference_mode():
        remaining = num_samples
        while remaining > 0:
            b = min(batch_size, remaining)
            xt = torch.randn(b, 3, 64, 64, device=device)

            # Only care about final x_0
            snapshots = ddim_sample(
                xt,
                model,
                noise_schedule,
                num_steps,
                steps_to_show=[0],
            )
            x0_hat = snapshots[0]  # assume dict {step: tensor}

            all_samples.append(x0_hat)
            remaining -= b

    return torch.cat(all_samples, dim=0)  # [N, 3, H, W]


def eval_kid(
    model,
    dataloader,
    noise_schedule,
    inception,
    num_real_samples: int = 512,
    num_steps: int = 100,
    device: str = "cuda",
):
    model.eval()
    real_imgs = []

    num_real_samples = min(num_real_samples, len(dataloader.dataset))

    with torch.inference_mode():
        for starts, _, _ in dataloader:
            starts = starts.to(device).float()
            starts = (starts / 127.5) - 1  # [-1, 1]
            real_imgs.append(starts)

            if sum(x.size(0) for x in real_imgs) >= num_real_samples:
                break

    real_imgs = torch.cat(real_imgs, dim=0)[:num_real_samples]  # [N, 3, H, W]

    fake_imgs = sample_diffusion_batch(
        model=model,
        noise_schedule=noise_schedule,
        num_samples=num_real_samples,
        num_steps=num_steps,
        batch_size=64,
        device=device,
    )  # [N, 3, H, W] in [-1, 1]

    def denorm(x):
        x = x.clamp(-1, 1)
        x = (x + 1) / 2.0  # [-1, 1] -> [0, 1]
        return x

    real_imgs = denorm(real_imgs)
    fake_imgs = denorm(fake_imgs)

    kid_value = compute_kid_from_tensors(real_imgs, fake_imgs, inception)
    return kid_value


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
    run_name: str = "run",
    save_freq: int = 10,
):
    run = Run(experiment="diffusion")
    run.name = run_name
    run["hparams"] = {
        "num_epochs": num_epochs,
        "save_freq": save_freq,
        "device": device,
        "T": T,
        "batch_size": dataloader.batch_size,
        "lr": optimizer.defaults["lr"],
        "dim": model.dim,
    }

    save_dir = f"{save_dir}/{run_name}"
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    scaler = torch.amp.GradScaler(device)

    for i in trange(num_epochs):
        model.train()

        epoch_loss = 0.0

        for j, (starts, _, _) in enumerate(tqdm(dataloader)):
            starts = starts.to(device).float()
            starts = (starts / 127.5) - 1

            t = torch.randint(0, T, (starts.shape[0],)).to(device)

            optimizer.zero_grad()

            xt, eps = q_sample(starts, t)

            t_float = t.float() / (T - 1)

            with torch.autocast(device, torch.float16):
                eps_pred = model(xt, t_float)
                loss = criterion(eps_pred, eps)

            global_step = i * len(dataloader) + j
            run.track(loss.item(), name="training_loss", step=global_step)
            epoch_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            if ema is not None:
                ema.update(model)

        if scheduler is not None:
            scheduler.step()

        epoch_loss /= len(dataloader)

        if (i + 1) % save_freq == 0:
            print(f"Epoch {i+1}/{num_epochs}, Loss: {epoch_loss}")
            torch.save(model.state_dict(), f"{save_dir}/model_{i+1}.pth")

            ddim_snapshots = generate_snapshots(model, noise_schedule)
            steps_str = "_".join(map(str, ddim_snapshots.keys()))
            imgs = images_dict_to_grid(ddim_snapshots)
            run.track(
                aim.Image(imgs, caption=f"ddim_snapshot_{steps_str}"),
                name="ddim_snapshot",
                step=global_step,
            )

            eval_dict, eps_pred, eps = eval_diffusion(
                model, dataloader, noise_schedule, t_val=250
            )
            steps_str = "_".join(map(str, eval_dict.keys()))
            eval_imgs = images_dict_to_grid(eval_dict)
            run.track(
                aim.Image(eval_imgs, caption=f"eval_{steps_str}"),
                name="eval",
                step=global_step,
            )

            loss = criterion(eps_pred, eps)
            run.track(loss.item(), name="eval_loss", step=global_step)

            kid_value = eval_kid(model, dataloader, noise_schedule, inception)
            run.track(kid_value, name="kid", step=global_step)
