from __future__ import annotations

import torch
from torch import Tensor


def poisson_encode(images: Tensor, time_steps: int) -> Tensor:
    """Convert normalized images [B, C, H, W] in [0, 1] to spikes [B, T, F]."""
    flat = images.flatten(start_dim=1)
    rand = torch.rand(
        flat.shape[0],
        time_steps,
        flat.shape[1],
        device=flat.device,
        dtype=flat.dtype,
    )
    return (rand < flat.unsqueeze(1)).float()
