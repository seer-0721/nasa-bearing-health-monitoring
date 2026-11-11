from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_stdp_params(path: str | Path = "stdp_params.npz") -> dict[str, float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"STDP parameter file not found: {path}")
    params = np.load(path)
    return {
        "A_plus": float(params["A_plus"]),
        "tau_plus": float(params["tau_plus"]),
        "A_minus": float(params["A_minus"]),
        "tau_minus": float(params["tau_minus"]),
    }
