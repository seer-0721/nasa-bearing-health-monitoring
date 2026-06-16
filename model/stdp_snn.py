from __future__ import annotations

import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class STDPLayer(nn.Module):
    """Linear synapse layer with trace-based STDP updates."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        A_plus: float,
        tau_plus: float,
        A_minus: float,
        tau_minus: float,
        dt_ms: float = 1.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
        lr_stdp: float = 1e-6,
        current_scale: float | None = None,
        ltd_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A_plus = float(A_plus)
        self.A_minus = abs(float(A_minus))
        self.tau_plus = float(tau_plus)
        self.tau_minus = float(tau_minus)
        self.dt = float(dt_ms)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.lr_stdp = float(lr_stdp)
        self.ltd_scale = float(ltd_scale)
        self.current_scale = 1.0 / in_features if current_scale is None else current_scale

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.weight, w_min, w_max)

        self.decay_pre = np.exp(-dt_ms / tau_plus) if tau_plus > 0 else 0.0
        self.decay_post = np.exp(-dt_ms / tau_minus) if tau_minus > 0 else 0.0

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.weight.T) * self.current_scale

    @torch.no_grad()
    def update_stdp(
        self,
        pre_spike: Tensor,
        post_spike: Tensor,
        pre_trace: Tensor,
        post_trace: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pre_spike = pre_spike.float()
        post_spike = post_spike.float()

        pre_trace = pre_trace * self.decay_pre + pre_spike
        post_trace = post_trace * self.decay_post + post_spike

        batch_size = max(pre_spike.shape[0], 1)
        ltp = post_spike.T @ pre_trace
        ltd = post_trace.T @ pre_spike
        delta_w = (self.A_plus * ltp - self.A_minus * self.ltd_scale * ltd) / batch_size

        self.weight.add_(delta_w * self.lr_stdp * self.dt)
        self.weight.clamp_(self.w_min, self.w_max)
        return pre_trace, post_trace


class LifModelSTDP(nn.Module):
    """Two-layer LIF classifier with a first STDP synapse layer."""

    def __init__(
        self,
        input_size: int = 3 * 32 * 32,
        hidden_size1: int = 256,
        hidden_size2: int = 128,
        output_size: int = 10,
        beta: float = 0.9,
        threshold: float = 1.0,
        stdp_params: dict[str, float] | None = None,
        dt_ms: float = 1.0,
        stdp_lr: float = 1e-6,
        stdp_ltd_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if stdp_params is None:
            stdp_params = {
                "A_plus": 80.0,
                "tau_plus": 20.0,
                "A_minus": -60.0,
                "tau_minus": 20.0,
            }

        self.stdp1 = STDPLayer(
            input_size,
            hidden_size1,
            A_plus=stdp_params["A_plus"],
            tau_plus=stdp_params["tau_plus"],
            A_minus=stdp_params["A_minus"],
            tau_minus=stdp_params["tau_minus"],
            dt_ms=dt_ms,
            lr_stdp=stdp_lr,
            ltd_scale=stdp_ltd_scale,
        )
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.stdp_enabled = False

    def set_stdp(self, enabled: bool, lr_stdp: float | None = None) -> None:
        self.stdp_enabled = enabled
        if lr_stdp is not None:
            self.stdp1.lr_stdp = float(lr_stdp)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, time_steps, _ = x.shape
        device = x.device

        mem1 = torch.zeros(batch_size, self.hidden_size1, device=device)
        mem2 = torch.zeros(batch_size, self.hidden_size2, device=device)
        pre_trace = torch.zeros(batch_size, self.stdp1.in_features, device=device)
        post_trace = torch.zeros(batch_size, self.stdp1.out_features, device=device)

        outputs = []
        for t in range(time_steps):
            xt = x[:, t, :]
            i1 = self.stdp1(xt)
            spk1, mem1 = self.lif1(i1, mem1)

            if self.stdp_enabled:
                pre_trace, post_trace = self.stdp1.update_stdp(
                    pre_spike=xt > 0,
                    post_spike=spk1,
                    pre_trace=pre_trace,
                    post_trace=post_trace,
                )

            i2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(i2, mem2)
            outputs.append(self.fc3(spk2))

        return torch.stack(outputs, dim=1).mean(dim=1)


class ConvLifModelSTDP(nn.Module):
    """Convolutional CIFAR-10 SNN with a device-parameterized STDP synapse."""

    def __init__(
        self,
        hidden_size1: int = 512,
        hidden_size2: int = 256,
        output_size: int = 10,
        beta: float = 0.9,
        threshold: float = 1.0,
        stdp_params: dict[str, float] | None = None,
        dt_ms: float = 1.0,
        stdp_lr: float = 1e-6,
        conv_channels: int = 96,
        stdp_current_scale: float | None = None,
        stdp_ltd_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if stdp_params is None:
            stdp_params = {
                "A_plus": 80.0,
                "tau_plus": 20.0,
                "A_minus": -60.0,
                "tau_minus": 20.0,
            }

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        feature_size = conv_channels * 4 * 4

        self.stdp1 = STDPLayer(
            feature_size,
            hidden_size1,
            A_plus=stdp_params["A_plus"],
            tau_plus=stdp_params["tau_plus"],
            A_minus=stdp_params["A_minus"],
            tau_minus=stdp_params["tau_minus"],
            dt_ms=dt_ms,
            lr_stdp=stdp_lr,
            current_scale=stdp_current_scale,
            ltd_scale=stdp_ltd_scale,
        )
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.stdp_enabled = False

    def set_stdp(self, enabled: bool, lr_stdp: float | None = None) -> None:
        self.stdp_enabled = enabled
        if lr_stdp is not None:
            self.stdp1.lr_stdp = float(lr_stdp)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, time_steps, _ = x.shape
        device = x.device

        mem1 = torch.zeros(batch_size, self.hidden_size1, device=device)
        mem2 = torch.zeros(batch_size, self.hidden_size2, device=device)
        pre_trace = torch.zeros(batch_size, self.stdp1.in_features, device=device)
        post_trace = torch.zeros(batch_size, self.stdp1.out_features, device=device)

        outputs = []
        for t in range(time_steps):
            xt = x[:, t, :].view(batch_size, 3, 32, 32)
            features = self.encoder(xt)
            i1 = self.stdp1(features)
            spk1, mem1 = self.lif1(i1, mem1)

            if self.stdp_enabled:
                pre_trace, post_trace = self.stdp1.update_stdp(
                    pre_spike=features > 0,
                    post_spike=spk1,
                    pre_trace=pre_trace,
                    post_trace=post_trace,
                )

            i2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(i2, mem2)
            outputs.append(self.fc3(spk2))

        return torch.stack(outputs, dim=1).mean(dim=1)


class SpikingConvBlock(nn.Module):
    """Conv-BN-LIF block that keeps membrane state across timesteps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        beta: float,
        threshold: float,
        pool: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = snn.Leaky(beta=beta, threshold=threshold)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        self.out_channels = out_channels

    def forward(self, x: Tensor, mem: Tensor) -> tuple[Tensor, Tensor]:
        cur = self.bn(self.conv(x))
        spk, mem = self.lif(cur, mem)
        return self.pool(spk), mem


class SpikingConvLifModelSTDP(nn.Module):
    """Poisson-input convolutional SNN with a device-parameterized STDP synapse."""

    def __init__(
        self,
        hidden_size1: int = 768,
        hidden_size2: int = 384,
        output_size: int = 10,
        beta: float = 0.9,
        threshold: float = 1.0,
        stdp_params: dict[str, float] | None = None,
        dt_ms: float = 1.0,
        stdp_lr: float = 1e-6,
        conv_channels: int = 128,
        conv_blocks_per_stage: int = 1,
        stdp_current_scale: float | None = None,
        stdp_ltd_scale: float = 1.0,
        dropout: float = 0.0,
        readout: str = "spike",
        wta_k: int = 0,
        adaptive_threshold_scale: float = 0.0,
        adaptive_threshold_decay: float = 0.9,
    ) -> None:
        super().__init__()
        if readout not in {"spike", "membrane", "hybrid"}:
            raise ValueError(f"Unknown readout mode: {readout}")
        if conv_blocks_per_stage < 1:
            raise ValueError("conv_blocks_per_stage must be >= 1")
        if wta_k < 0:
            raise ValueError("wta_k must be >= 0")
        if stdp_params is None:
            stdp_params = {
                "A_plus": 80.0,
                "tau_plus": 20.0,
                "A_minus": -60.0,
                "tau_minus": 20.0,
            }

        stage_defs = [(3, 32, 32), (32, 64, 16), (64, conv_channels, 8)]
        self.conv_stages = nn.ModuleList()
        self.mem_shapes: list[tuple[int, int]] = []
        for in_channels, out_channels, spatial_size in stage_defs:
            blocks = nn.ModuleList()
            for block_idx in range(conv_blocks_per_stage):
                block_in = in_channels if block_idx == 0 else out_channels
                pool = block_idx == conv_blocks_per_stage - 1
                blocks.append(
                    SpikingConvBlock(
                        block_in,
                        out_channels,
                        beta=beta,
                        threshold=threshold,
                        pool=pool,
                    )
                )
                self.mem_shapes.append((out_channels, spatial_size))
            self.conv_stages.append(blocks)
        self.conv_channels = conv_channels
        self.conv_blocks_per_stage = conv_blocks_per_stage
        feature_size = conv_channels * 4 * 4
        if stdp_current_scale is None:
            stdp_current_scale = 1.0 / (feature_size**0.5)

        self.stdp1 = STDPLayer(
            feature_size,
            hidden_size1,
            A_plus=stdp_params["A_plus"],
            tau_plus=stdp_params["tau_plus"],
            A_minus=stdp_params["A_minus"],
            tau_minus=stdp_params["tau_minus"],
            dt_ms=dt_ms,
            lr_stdp=stdp_lr,
            current_scale=stdp_current_scale,
            ltd_scale=stdp_ltd_scale,
        )
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.readout = readout
        self.wta_k = int(wta_k)
        self.adaptive_threshold_scale = float(adaptive_threshold_scale)
        self.adaptive_threshold_decay = float(adaptive_threshold_decay)
        self.stdp_enabled = False

    def set_stdp(self, enabled: bool, lr_stdp: float | None = None) -> None:
        self.stdp_enabled = enabled
        if lr_stdp is not None:
            self.stdp1.lr_stdp = float(lr_stdp)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, time_steps, _ = x.shape
        device = x.device

        conv_mems = [
            torch.zeros(batch_size, channels, size, size, device=device)
            for channels, size in self.mem_shapes
        ]
        mem1 = torch.zeros(batch_size, self.hidden_size1, device=device)
        mem2 = torch.zeros(batch_size, self.hidden_size2, device=device)
        adaptive_threshold = torch.zeros(batch_size, self.hidden_size1, device=device)
        pre_trace = torch.zeros(batch_size, self.stdp1.in_features, device=device)
        post_trace = torch.zeros(batch_size, self.stdp1.out_features, device=device)

        outputs = []
        for t in range(time_steps):
            spk = x[:, t, :].view(batch_size, 3, 32, 32)
            mem_idx = 0
            for blocks in self.conv_stages:
                for block in blocks:
                    spk, conv_mems[mem_idx] = block(spk, conv_mems[mem_idx])
                    mem_idx += 1
            features = F.adaptive_max_pool2d(spk, (4, 4)).flatten(1)

            i1 = self.stdp1(features)
            spk1, mem1 = self.lif1(i1 - adaptive_threshold, mem1)
            if self.wta_k > 0:
                k = min(self.wta_k, spk1.shape[1])
                winner_idx = i1.topk(k, dim=1).indices
                winner_mask = torch.zeros_like(spk1).scatter(1, winner_idx, 1.0)
                spk1 = spk1 * winner_mask
            if self.adaptive_threshold_scale > 0:
                adaptive_threshold = (
                    adaptive_threshold * self.adaptive_threshold_decay
                    + spk1.detach() * self.adaptive_threshold_scale
                )

            if self.stdp_enabled:
                pre_trace, post_trace = self.stdp1.update_stdp(
                    pre_spike=features > 0,
                    post_spike=spk1,
                    pre_trace=pre_trace,
                    post_trace=post_trace,
                )

            spk1 = self.dropout(spk1)
            i2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(i2, mem2)
            if self.readout == "membrane":
                readout = mem2
            elif self.readout == "hybrid":
                readout = spk2 + mem2
            else:
                readout = spk2
            outputs.append(self.fc3(readout))

        return torch.stack(outputs, dim=1).mean(dim=1)
