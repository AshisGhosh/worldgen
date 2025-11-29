import torch


def ddim_sample(xt, model, noise_schedule, ddim_steps, steps_to_show):
    device = xt.device
    T = noise_schedule["T"]
    ddim_steps = torch.linspace(0, T - 1, ddim_steps, device=device, dtype=torch.long)
    timesteps = list(reversed(ddim_steps.tolist()))  # e.g. [392, 384, ..., 0]

    # find the next closest timestep in steps_to_show
    # For each step in steps_to_show, find the closest timestep in ddim_steps
    steps_to_show = [
        int(min(ddim_steps, key=lambda t: abs(t - x)).item()) for x in steps_to_show
    ]
    snapshots = {}

    alpha_cumprod = noise_schedule["alphas_bar"]
    sqrt_alphas_bar = noise_schedule["sqrt_alphas_bar"]
    sqrt_one_minus_alphas_bar = noise_schedule["sqrt_one_minus_alphas_bar"]

    model.eval()
    with torch.inference_mode():
        for i, t in enumerate(timesteps):
            t_int = int(t)
            t_long = torch.full((xt.size(0),), t_int, device=device, dtype=torch.long)
            t_float = t_long.float() / (T - 1)

            # predict eps at this timestep
            with torch.autocast(device.type, torch.float16):
                eps = model(xt, t_float)

            sqrt_alpha_bar_t = sqrt_alphas_bar[t_int].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_bar[t_int].view(
                -1, 1, 1, 1
            )

            # estimate x0
            x0_hat = (xt - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t
            x0_hat = x0_hat.clamp(-1, 1)

            # if this is the last step (t == 0 in our schedule), we just take x0_hat
            if i == len(timesteps) - 1:
                xt = x0_hat
            else:
                # DDIM deterministic update to the *next* timestep in our chosen schedule
                t_next = int(timesteps[i + 1])
                alpha_bar_next = alpha_cumprod[t_next]
                sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next).view(1, 1, 1, 1)
                sqrt_one_minus_alpha_bar_next = torch.sqrt(1 - alpha_bar_next).view(
                    1, 1, 1, 1
                )

                # η = 0 DDIM (no extra noise term)
                xt = sqrt_alpha_bar_next * x0_hat + sqrt_one_minus_alpha_bar_next * eps

            if t_int in steps_to_show:
                img = xt.clamp(-1, 1)
                img = (img + 1) / 2
                snapshots[t_int] = img.squeeze(0)

    return snapshots
