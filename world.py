import torch


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
    save_freq: int = 10,
):
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    for i in trange(num_epochs):
        model.train()

        epoch_loss = 0.0

        for starts, actions, ends in tqdm(dataloader):
            starts, actions, ends = (
                starts.to(device).float(),
                actions.to(device).float(),
                ends.to(device).float(),
            )
            starts = (starts / 127.5) - 1
            ends = (ends / 127.5) - 1

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
