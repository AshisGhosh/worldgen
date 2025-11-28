import torch
import torchvision.utils as vutils


def images_dict_to_grid(images_dict, nrow=4, normalize=True):
    images = list(images_dict.values())
    batch = torch.stack(images)

    grid = vutils.make_grid(batch, nrow=nrow, normalize=normalize, scale_each=True)

    return grid
