from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from typing import Optional, Literal, Tuple
from tqdm import tqdm
import numpy as np
import cv2


@torch.no_grad()
def generate_masks(
    num_masks: int,
    H: int,
    W: int,
    p_keep: float = 0.5,
    cell: int = 8,
    seed: Optional[int] = 0,
) -> torch.Tensor:
    # RNG
    g = torch.Generator(device="cpu") if seed is not None else None
    if g is not None:
        g.manual_seed(seed)

    # coarse grid size that covers the image
    grid_h = math.ceil(H / cell)
    grid_w = math.ceil(W / cell)

    # sample coarse Bernoulli grid
    coarse = (torch.rand((num_masks, 1, grid_h, grid_w), generator=g) < p_keep).float()

    # upsample large enough to avoid padding:
    # add +cell in each dim so any shift in [0, cell-1] stays fully inside
    upH = grid_h * cell + cell
    upW = grid_w * cell + cell
    up = F.interpolate(coarse, size=(upH, upW), mode="nearest")  # keep block masks

    # random shifts in [0, cell)
    shift_y = torch.randint(0, cell, (num_masks,), generator=g)
    shift_x = torch.randint(0, cell, (num_masks,), generator=g)

    masks = torch.empty((num_masks, 1, H, W), dtype=up.dtype)
    for i in range(num_masks):
        y0 = int(shift_y[i])
        x0 = int(shift_x[i])
        masks[i, 0] = up[i, 0, y0 : y0 + H, x0 : x0 + W]

    return masks.clamp_(0, 1)


def generate_mask_from_grid(
    image_size: Tuple[int, int], grid_size: Tuple[int, int], prob_thresh: float
) -> torch.Tensor:
    image_h, image_w = image_size
    grid_h, grid_w = grid_size
    cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
    up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

    mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) < prob_thresh).astype(
        np.float32
    )
    mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = np.random.randint(0, cell_w)
    offset_h = np.random.randint(0, cell_h)
    mask = mask[offset_h : offset_h + image_h, offset_w : offset_w + image_w]
    return torch.from_numpy(mask).float()


# def generate_reference_mask(image_size: Tuple[int, int], grid_size: Tuple[int, int], prob_thresh: float) -> torch.Tensor:
