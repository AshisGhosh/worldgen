from diffusion import train_diffusion
from dit import DiT
from dataset import WorldDataset, SingleSampleDataset  # noqa: F401
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import random
from datetime import datetime

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = WorldDataset("data/world_map.png")
    # dataset = SingleSampleDataset(dataset)

    g = torch.Generator()
    g.manual_seed(42)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = DiT().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    run_name = f"diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    train_diffusion(
        model,
        criterion,
        optimizer,
        dataloader,
        device=device,
        num_epochs=2000,
        save_freq=10,
        run_name=run_name,
        save_dir="./checkpoints",
    )


if __name__ == "__main__":
    train()
