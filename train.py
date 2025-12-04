from train.diffusion import train_diffusion
from train.world import train_world_model
from train.world_flow import train_world_flow
from models import DiT, WorldDiT
from dataset import WorldDataset, SelectSampleDataset  # noqa: F401
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import random
from datetime import datetime

import argparse

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(
    run_name=None, experiment="diffusion", pretrained=False, pretrained_path=None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = WorldDataset("data/world_map.png")
    # dataset = SelectSampleDataset(dataset, num_samples=64)

    g = torch.Generator()
    g.manual_seed(42)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    run_name_str = ""
    if experiment is not None:
        run_name_str += f"{experiment}_"
    if run_name is not None:
        run_name_str += run_name

    run_name_str += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_name = run_name_str

    match experiment:
        case "diffusion":
            model = DiT().to(device)

            if pretrained:
                model.load_state_dict(torch.load(pretrained_path))

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
        case "world":
            model = WorldDiT().to(device)

            if pretrained:
                model.load_state_dict(torch.load(pretrained_path))

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            train_world_model(
                model,
                criterion,
                optimizer,
                dataloader,
                device=device,
                num_epochs=3000,
                save_freq=10,
                run_name=run_name,
                save_dir="./checkpoints",
            )
        case "world_flow":
            model = WorldDiT(enable_cfg=True).to(device)

            if pretrained:
                model.load_state_dict(torch.load(pretrained_path))

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            train_world_flow(
                model,
                criterion,
                optimizer,
                dataloader,
                device=device,
                num_epochs=3000,
                save_freq=10,
                run_name=run_name,
                save_dir="./checkpoints",
            )
        case _:
            raise ValueError(f"Unknown experiment: {experiment}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="world_flow")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="checkpoints/world_filmcond_pre_20251129_203128/model_3000.pth",
    )
    args = parser.parse_args()

    train(
        run_name=args.run_name,
        experiment=args.experiment,
        pretrained=args.pretrained,
        pretrained_path=args.pretrained_path,
    )
