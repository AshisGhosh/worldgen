import torch
from tqdm import tqdm, trange
from typing import Optional, Dict
import os
from itertools import islice

from torch.utils.data import DataLoader

from .world_sampler import world_ddim_sample
import aim
from aim import Run
import numpy as np

from ..metrics import compute_kid_from_tensors, load_inception
from ..diffusion import get_noise_schedule, q_sample
from ..utils import images_dict_to_grid

inception = load_inception()

device = "cuda" if torch.cuda.is_available() else "cpu"
T = 400
noise_schedule = get_noise_schedule(T=T, device=device)


def generate_world_snapshots(
    model, dataset, noise_schedule, steps_to_show=[300, 200, 100, 0]
) -> Dict[int, np.ndarray]:
    num_steps = 100

    # [B, C, H, W]
    xt = torch.randn(1, 3, 64, 64).to(device)
    # [1, C, H, W]
    start = dataset[0][0].to(device).unsqueeze(0)
    # [B]
    action = torch.full((xt.size(0),), dataset[0][1], device=device)

    ddim_snapshots = world_ddim_sample(
        xt, model, start, action, noise_schedule, num_steps, steps_to_show
    )
    return ddim_snapshots, start.squeeze(0), action.float().mean().int().item()


def eval_world_diffusion(model, loader, noise_schedule, t_val=None):
    model.eval()
    with torch.inference_mode():
        starts, actions, imgs = loader.dataset[0]

        # [1, C, H, W]
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

        # [1, C, H, W]
        starts = (starts.to(device) / 127.5) - 1
        starts = starts.unsqueeze(0)
        # [1]
        actions_float = torch.tensor([actions]).to(device).float() / 4
        with torch.autocast(device, torch.float16):
            eps_pred = model(xt, t_float, starts, actions_float)
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
            "starts": denorm(starts),
            "x0": denorm(x0),
            f"xt_{t_int}": denorm(xt),
            "x0_hat": denorm(x0_hat),
        }

    return eval_dict, eps_pred, eps, actions


def sample_world_batch(
    model,
    dataloader,
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

    assert num_samples % dataloader.batch_size == 0
    max_batches = num_samples // dataloader.batch_size

    with torch.inference_mode():
        for _, (starts, actions, ends) in enumerate(
            tqdm(islice(dataloader, max_batches))
        ):
            xt = torch.randn(starts.size(0), 3, 64, 64, device=device)
            # Only care about final x_0
            snapshots = world_ddim_sample(
                xt,
                model,
                starts.to(device),
                actions.to(device),
                noise_schedule,
                num_steps,
                steps_to_show=[0],
            )
            x0_hat = snapshots[0]  # assume dict {step: tensor}

            all_samples.append(x0_hat)

    return torch.cat(all_samples, dim=0)  # [N, 3, H, W]


def eval_world_kid(
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

    with torch.inference_mode():
        for _, _, ends in dataloader:
            ends = ends.to(device).float()
            ends = (ends / 127.5) - 1  # [-1, 1]
            real_imgs.append(ends)

            if sum(x.size(0) for x in real_imgs) >= num_real_samples:
                break

    real_imgs = torch.cat(real_imgs, dim=0)[:num_real_samples]  # [N, 3, H, W]

    fake_imgs = sample_world_batch(
        model=model,
        dataloader=dataloader,
        noise_schedule=noise_schedule,
        num_samples=num_real_samples,
        num_steps=num_steps,
        batch_size=dataloader.batch_size,
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


def train_world_model(
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
    run = Run(experiment="world_model")
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

    eval_loader = DataLoader(dataloader.dataset, batch_size=512, shuffle=False)

    scaler = torch.amp.GradScaler(device)

    for i in trange(num_epochs):
        model.train()

        epoch_loss = 0.0

        for j, (starts, actions, ends) in enumerate(tqdm(dataloader)):
            starts, actions, ends = (
                starts.to(device).float(),
                actions.to(device).float(),
                ends.to(device).float(),
            )
            starts = (starts / 127.5) - 1
            ends = (ends / 127.5) - 1

            t = torch.randint(0, T, (starts.shape[0],)).to(device)

            optimizer.zero_grad()

            xt, eps = q_sample(ends, t)

            t_float = t.float() / (T - 1)
            actions_float = actions.float() / 4

            with torch.autocast(device, torch.float16):
                eps_pred = model(xt, t_float, starts, actions_float)
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

            ddim_snapshots, start, action = generate_world_snapshots(
                model, dataloader.dataset, noise_schedule
            )
            steps_str = "_".join(map(str, ddim_snapshots.keys()))
            ddim_snapshots["start"] = start
            steps_str += f"_start_{action}"
            imgs = images_dict_to_grid(ddim_snapshots)
            run.track(
                aim.Image(imgs, caption=f"ddim_snapshot_{steps_str}"),
                name="ddim_snapshot",
                step=global_step,
            )

            eval_dict, eps_pred, eps, actions = eval_world_diffusion(
                model, eval_loader, noise_schedule, t_val=250
            )
            steps_str = "_".join(map(str, eval_dict.keys()))
            eval_imgs = images_dict_to_grid(eval_dict)
            run.track(
                aim.Image(eval_imgs, caption=f"eval_{steps_str}_{actions}"),
                name="eval",
                step=global_step,
            )

            loss = criterion(eps_pred, eps)
            run.track(loss.item(), name="eval_loss", step=global_step)
            kid_value = eval_world_kid(model, eval_loader, noise_schedule, inception)
            run.track(kid_value, name="kid", step=global_step)
