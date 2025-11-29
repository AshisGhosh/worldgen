from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
from typing import Tuple


def load_world_map(world_map_file: str) -> torch.Tensor:
    np_img = np.array(Image.open(world_map_file))
    assert np_img.shape[-1] == 3, f"Image must have 3 channels, got {np_img.shape[-1]}"
    img_tensor = torch.from_numpy(np_img).permute(2, 0, 1)
    return img_tensor


class WorldDataset(Dataset):
    def __init__(
        self, world_map_file, crop_size: int = 64, stride: int = 8, epoch_length=10_000
    ):
        self._world_map_file = world_map_file
        self._world_map = load_world_map(world_map_file)
        self.mapH, self.mapW = self._world_map.shape[-2], self._world_map.shape[-1]
        self.crop_size = crop_size
        self.stride = stride
        self.epoch_length = epoch_length

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Returns a tuple of:
        (start_crop, action, end_crop)
        where action is an integer from 0 to 3
        0: up
        1: down
        2: left
        3: right
        """
        start_center_x = torch.randint(
            self.crop_size // 2, self.mapW - self.crop_size // 2, (1,)
        ).item()
        start_center_y = torch.randint(
            self.crop_size // 2, self.mapH - self.crop_size // 2, (1,)
        ).item()

        # get start crop
        start_crop = self._world_map[
            :,
            start_center_y - self.crop_size // 2 : start_center_y + self.crop_size // 2,
            start_center_x - self.crop_size // 2 : start_center_x + self.crop_size // 2,
        ]

        # get action
        action = torch.randint(0, 4, (1,)).item()

        end_center_x = start_center_x
        end_center_y = start_center_y

        if action == 0:  # up
            end_center_y -= self.stride
        elif action == 1:  # down
            end_center_y += self.stride
        elif action == 2:  # left
            end_center_x -= self.stride
        elif action == 3:  # right
            end_center_x += self.stride

        # Clamp to ensure end_crop stays within bounds
        end_center_x = max(
            self.crop_size // 2, min(end_center_x, self.mapW - self.crop_size // 2)
        )
        end_center_y = max(
            self.crop_size // 2, min(end_center_y, self.mapH - self.crop_size // 2)
        )

        # get end crop
        end_crop = self._world_map[
            :,
            end_center_y - self.crop_size // 2 : end_center_y + self.crop_size // 2,
            end_center_x - self.crop_size // 2 : end_center_x + self.crop_size // 2,
        ]

        return start_crop, action, end_crop


class SelectSampleDataset(Dataset):
    """Wrapper dataset that always returns the same cached sample.
    Useful for overfitting experiments."""

    def __init__(
        self, base_dataset: Dataset, num_samples=10_000, epoch_length: int = 10_000
    ):
        self.cached_samples = [base_dataset[i] for i in range(num_samples)]
        self.epoch_length = epoch_length

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, index: int):
        return self.cached_samples[index % len(self.cached_samples)]
