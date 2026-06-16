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


def deterministic_rate_encode(images: Tensor, time_steps: int) -> Tensor:
    """Rate code with evenly spaced deterministic spikes instead of Bernoulli noise."""
    flat = images.flatten(start_dim=1).clamp(0.0, 1.0)
    steps = torch.arange(
        time_steps + 1,
        device=flat.device,
        dtype=flat.dtype,
    ).view(1, time_steps + 1, 1)
    cumulative = torch.floor(flat.unsqueeze(1) * steps)
    return (cumulative[:, 1:] > cumulative[:, :-1]).float()


def latency_encode(images: Tensor, time_steps: int) -> Tensor:
    """Time-to-first-spike code: brighter pixels fire earlier and at most once."""
    flat = images.flatten(start_dim=1).clamp(0.0, 1.0)
    spike_time = torch.round((1.0 - flat) * (time_steps - 1)).long()
    time_index = torch.arange(time_steps, device=flat.device).view(1, time_steps, 1)
    active = flat > 0
    return ((time_index == spike_time.unsqueeze(1)) & active.unsqueeze(1)).float()


def encode_spikes(images: Tensor, time_steps: int, encoding: str = "poisson") -> Tensor:
    if encoding == "poisson":
        return poisson_encode(images, time_steps)
    if encoding == "deterministic-rate":
        return deterministic_rate_encode(images, time_steps)
    if encoding == "latency":
        return latency_encode(images, time_steps)
    raise ValueError(f"Unknown spike encoding: {encoding}")
