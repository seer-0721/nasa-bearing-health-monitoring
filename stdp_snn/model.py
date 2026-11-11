from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class STDPSNNConfig:
    input_size: int
    hidden_size: int = 512
    time_steps: int = 8
    beta: float = 0.85
    threshold: float = 0.65
    input_scale: float = 1.0
    stdp_lr: float = 2e-5
    stdp_ltd_scale: float = 0.4
    w_min: float = 0.0
    w_max: float = 1.0
    include_raw: bool = True
    seed: int = 42


class STDPSNNFeatureMap:
    """Single-layer STDP-LIF feature extractor.

    The feature extractor keeps the main SNN mechanism explicit:
    deterministic rate spikes drive a trainable STDP synapse matrix, LIF neurons
    produce postsynaptic spikes, and downstream readouts are trained separately.
    """

    def __init__(self, config: STDPSNNConfig, stdp_params: dict[str, float]) -> None:
        self.config = config
        self.stdp_params = dict(stdp_params)
        rng = np.random.default_rng(config.seed)
        self.weight = rng.uniform(
            low=config.w_min,
            high=config.w_max,
            size=(config.input_size, config.hidden_size),
        ).astype(np.float32)
        self.bias = rng.normal(0.0, 0.02, size=(config.hidden_size,)).astype(np.float32)
        self.decay_pre = self._decay("tau_plus")
        self.decay_post = self._decay("tau_minus")
        self.a_plus = abs(float(stdp_params["A_plus"]))
        self.a_minus = abs(float(stdp_params["A_minus"]))

    def _decay(self, key: str) -> float:
        tau = float(self.stdp_params[key])
        return float(np.exp(-1.0 / tau)) if tau > 0 else 0.0

    def _rate_spikes(self, x: np.ndarray, step: int) -> np.ndarray:
        left = np.floor(x * step)
        right = np.floor(x * (step + 1))
        return (right > left).astype(np.float32, copy=False)

    def pretrain(self, features: np.ndarray, epochs: int = 1, batch_size: int = 512) -> None:
        x_all = np.clip(features.astype(np.float32, copy=False), 0.0, 1.0)
        rng = np.random.default_rng(self.config.seed + 1009)
        for _ in range(max(epochs, 0)):
            order = rng.permutation(len(x_all))
            for start in range(0, len(order), batch_size):
                batch = x_all[order[start : start + batch_size]]
                self._pretrain_batch(batch)

    def _pretrain_batch(self, x: np.ndarray) -> None:
        batch = len(x)
        if batch == 0:
            return
        mem = np.zeros((batch, self.config.hidden_size), dtype=np.float32)
        pre_trace = np.zeros((batch, self.config.input_size), dtype=np.float32)
        post_trace = np.zeros((batch, self.config.hidden_size), dtype=np.float32)

        for step in range(self.config.time_steps):
            pre = self._rate_spikes(x, step)
            current = (pre @ self.weight) * (self.config.input_scale / max(self.config.input_size, 1))
            current += self.bias
            mem = self.config.beta * mem + current
            post = (mem >= self.config.threshold).astype(np.float32)
            mem = mem * (1.0 - post)

            pre_trace = pre_trace * self.decay_pre + pre
            post_trace = post_trace * self.decay_post + post
            ltp = pre_trace.T @ post
            ltd = pre.T @ post_trace
            delta = (self.a_plus * ltp - self.a_minus * self.config.stdp_ltd_scale * ltd)
            delta /= max(batch, 1)
            self.weight += np.float32(self.config.stdp_lr) * delta.astype(np.float32)
            np.clip(self.weight, self.config.w_min, self.config.w_max, out=self.weight)

    def transform(self, features: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        outputs = []
        for start in range(0, len(features), chunk_size):
            x = np.clip(features[start : start + chunk_size].astype(np.float32, copy=False), 0.0, 1.0)
            batch = len(x)
            mem = np.zeros((batch, self.config.hidden_size), dtype=np.float32)
            spike_sum = np.zeros_like(mem)
            mem_sum = np.zeros_like(mem)
            mem_abs_max = np.zeros_like(mem)

            for step in range(self.config.time_steps):
                pre = self._rate_spikes(x, step)
                current = (pre @ self.weight) * (self.config.input_scale / max(self.config.input_size, 1))
                current += self.bias
                mem = self.config.beta * mem + current
                post = (mem >= self.config.threshold).astype(np.float32)
                mem = mem * (1.0 - post)
                spike_sum += post
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

