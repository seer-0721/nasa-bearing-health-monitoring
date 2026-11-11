from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimpleSNNConfig:
    input_size: int
    hidden_size: int = 512
    time_steps: int = 8
    beta: float = 0.85
    threshold: float = 0.65
    input_scale: float = 1.0
    recurrent_scale: float = 0.9
    recurrent_density: float = 0.05
    include_raw: bool = True
    seed: int = 42


class SimpleRateSNNFeatureMap:
    """Fixed one-layer rate/LIF SNN feature map.

    The model is intentionally small: deterministic rate spikes drive a single
    LIF layer, then balanced ridge readouts are trained on spike and membrane
    statistics. The reservoir is fixed, so training stays fast and stable.
    """

    def __init__(self, config: SimpleSNNConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        scale = config.input_scale / max(np.sqrt(config.input_size), 1.0)
        self.w_in = rng.normal(
            loc=0.0,
            scale=scale,
            size=(config.input_size, config.hidden_size),
        ).astype(np.float32)
        mask = rng.random((config.hidden_size, config.hidden_size)) < config.recurrent_density
        w_rec = rng.normal(
            loc=0.0,
            scale=1.0 / max(np.sqrt(config.hidden_size), 1.0),
            size=(config.hidden_size, config.hidden_size),
        ).astype(np.float32)
        w_rec *= mask.astype(np.float32)
        eigvals = np.linalg.eigvals(w_rec)
        radius = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        if radius > 1e-8:
            w_rec *= np.float32(config.recurrent_scale / radius)
        self.w_rec = w_rec.astype(np.float32)
        self.bias = rng.normal(
            loc=0.0,
            scale=0.05,
            size=(config.hidden_size,),
        ).astype(np.float32)

    def _rate_spikes(self, x: np.ndarray, step: int) -> np.ndarray:
        left = np.floor(x * step)
        right = np.floor(x * (step + 1))
        return (right > left).astype(np.float32, copy=False)

    def transform(self, features: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        outputs = []
        for start in range(0, len(features), chunk_size):
            x = features[start : start + chunk_size].astype(np.float32, copy=False)
            x = np.clip(x, 0.0, 1.0)
            batch = len(x)
            mem = np.zeros((batch, self.config.hidden_size), dtype=np.float32)
            spike_sum = np.zeros_like(mem)
            mem_sum = np.zeros_like(mem)
            mem_abs_max = np.zeros_like(mem)

            for step in range(self.config.time_steps):
                spikes = self._rate_spikes(x, step)
                current = spikes @ self.w_in + mem @ self.w_rec + self.bias
                mem = self.config.beta * mem + current
                out_spike = (mem >= self.config.threshold).astype(np.float32)
                mem = mem * (1.0 - out_spike)
                spike_sum += out_spike
                mem_sum += mem
                mem_abs_max = np.maximum(mem_abs_max, np.abs(mem))

            mapped = np.concatenate(
                [
                    spike_sum / max(self.config.time_steps, 1),
                    mem_sum / max(self.config.time_steps, 1),
                    mem_abs_max,
                ],
                axis=1,
            )
            if self.config.include_raw:
                mapped = np.concatenate([mapped, x], axis=1)
            outputs.append(mapped.astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0)
