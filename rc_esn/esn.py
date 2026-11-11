from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ESNConfig:
    input_dim: int = 1
    reservoir_size: int = 800
    spectral_radius: float = 0.9
    leaking_rate: float = 0.3
    input_scale: float = 0.5
    density: float = 0.05
    include_raw: bool = True
    seed: int = 42


class ESNFeatureMap:
    def __init__(self, config: ESNConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.w_in = rng.uniform(
            low=-config.input_scale,
            high=config.input_scale,
            size=(config.reservoir_size, config.input_dim + 1),
        ).astype(np.float32)

        mask = rng.random((config.reservoir_size, config.reservoir_size)) < config.density
        w_res = rng.uniform(-1.0, 1.0, size=mask.shape).astype(np.float32) * mask
        eigvals = np.linalg.eigvals(w_res)
        radius = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        if radius > 1e-8:
            w_res *= np.float32(config.spectral_radius / radius)
        self.w_res = w_res.astype(np.float32)

    def _sequence(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        input_dim = self.config.input_dim
        if input_dim == 1:
            return x.reshape(-1, 1)
        pad = (-len(x)) % input_dim
        if pad:
            x = np.pad(x, (0, pad), mode="constant")
        return x.reshape(-1, input_dim)

    def transform(self, features: np.ndarray) -> np.ndarray:
        states = []
        leak = np.float32(self.config.leaking_rate)
        for row in features.astype(np.float32, copy=False):
            state = np.zeros(self.config.reservoir_size, dtype=np.float32)
            state_sum = np.zeros_like(state)
            state_sq_sum = np.zeros_like(state)
            seq = self._sequence(row)
            for step in seq:
                with_bias = np.concatenate([np.ones(1, dtype=np.float32), step])
                candidate = np.tanh(self.w_in @ with_bias + self.w_res @ state)
                state = (1.0 - leak) * state + leak * candidate
                state_sum += state
                state_sq_sum += state * state
            mean_state = state_sum / max(len(seq), 1)
            var_state = np.maximum(state_sq_sum / max(len(seq), 1) - mean_state * mean_state, 0.0)
            mapped = np.concatenate([state, mean_state, np.sqrt(var_state)], axis=0)
            if self.config.include_raw:
                mapped = np.concatenate([mapped, row], axis=0)
            states.append(mapped.astype(np.float32, copy=False))
        return np.stack(states, axis=0)

