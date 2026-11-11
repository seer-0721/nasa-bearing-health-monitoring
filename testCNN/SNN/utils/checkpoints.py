from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer

from model import LifModelSTDP


def save_checkpoint(
    path: str | Path,
    model: LifModelSTDP,
    optimizer: Optimizer | None,
    epoch: int,
    args: Namespace | dict[str, Any],
    metrics: dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "args": vars(args) if isinstance(args, Namespace) else args,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)
