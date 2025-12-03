import torch
from tqdm import tqdm
from typing import Dict


def world_flow_sample(
    x0, model, start, action, flow_steps, steps_to_show, cfg_scale=2.0
) -> Dict[int, torch.Tensor]:
    device = x0.device

    dt = 1 / flow_steps
    timesteps = torch.arange(0, 1, dt, device=device)

    snapshots = {}

    model.eval()
    with torch.inference_mode():
        start = (start / 127.5) - 1
        action_float = action / 4
        for i, t in enumerate(tqdm(timesteps)):
            # predict v at this timestep
            with torch.autocast(device.type, torch.float16):
                t_expand = torch.full((x0.size(0),), t, device=device)
                v = model(x0, t_expand, start, action_float)
                if model.enable_cfg and cfg_scale > 0.0:
                    v_null = model(x0, t_expand, start, action_float, cfg=True)
                    v = v_null + (v - v_null) * cfg_scale

            x0 = x0 + v * dt

            if i in steps_to_show:
                img = x0.clamp(-1, 1)
                img = (img + 1) / 2
                snapshots[i] = img.squeeze(0)

    return snapshots
