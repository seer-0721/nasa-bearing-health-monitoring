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
        hidden_current_scale: float = 1.0,
        readout: str = "spike",
    ) -> None:
        super().__init__()
        if readout not in {"spike", "membrane", "hybrid"}:
            raise ValueError(f"Unknown readout mode: {readout}")
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
        self.hidden_current_scale = float(hidden_current_scale)
        self.readout = readout
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

            i2 = self.fc2(spk1) * self.hidden_current_scale
            spk2, mem2 = self.lif2(i2, mem2)
            if self.readout == "membrane":
                readout = mem2
            elif self.readout == "hybrid":
                readout = spk2 + mem2
            else:
                readout = spk2
            outputs.append(self.fc3(readout))

        return torch.stack(outputs, dim=1).mean(dim=1)

    @torch.no_grad()
    def spike_stats(self, x: Tensor) -> dict[str, float]:
        batch_size, time_steps, _ = x.shape
        device = x.device

        mem1 = torch.zeros(batch_size, self.hidden_size1, device=device)
        mem2 = torch.zeros(batch_size, self.hidden_size2, device=device)
        hidden1_total = 0.0
        hidden2_total = 0.0
        hidden1_any = torch.zeros(batch_size, self.hidden_size1, dtype=torch.bool, device=device)
        hidden2_any = torch.zeros(batch_size, self.hidden_size2, dtype=torch.bool, device=device)
        mem1_abs_sum = 0.0
        mem2_abs_sum = 0.0
        mem1_max = 0.0
        mem2_max = 0.0

        for t in range(time_steps):
            xt = x[:, t, :]
            i1 = self.stdp1(xt)
            spk1, mem1 = self.lif1(i1, mem1)
            i2 = self.fc2(spk1) * self.hidden_current_scale
            spk2, mem2 = self.lif2(i2, mem2)

            hidden1_total += float(spk1.sum().item())
            hidden2_total += float(spk2.sum().item())
            hidden1_any |= spk1.bool()
            hidden2_any |= spk2.bool()
            mem1_abs_sum += float(mem1.abs().sum().item())
            mem2_abs_sum += float(mem2.abs().sum().item())
            mem1_max = max(mem1_max, float(mem1.abs().max().item()))
            mem2_max = max(mem2_max, float(mem2.abs().max().item()))

        input_total = float((x > 0).sum().item())
        input_rate = input_total / max(x.numel(), 1)
        hidden1_count = batch_size * time_steps * self.hidden_size1
        hidden2_count = batch_size * time_steps * self.hidden_size2
        return {
            "input_spike_rate": input_rate,
            "input_spikes_per_sample": input_total / max(batch_size, 1),
            "hidden1_spike_rate": hidden1_total / max(hidden1_count, 1),
            "hidden1_spikes_per_sample": hidden1_total / max(batch_size, 1),
            "hidden1_active_fraction": float(hidden1_any.float().mean().item()),
            "hidden2_spike_rate": hidden2_total / max(hidden2_count, 1),
            "hidden2_spikes_per_sample": hidden2_total / max(batch_size, 1),
            "hidden2_active_fraction": float(hidden2_any.float().mean().item()),
            "mem1_abs_mean": mem1_abs_sum / max(hidden1_count, 1),
            "mem1_abs_max": mem1_max,
            "mem2_abs_mean": mem2_abs_sum / max(hidden2_count, 1),
            "mem2_abs_max": mem2_max,
        }


class RecurrentLifModelSTDP(nn.Module):
    """Recurrent LIF classifier with a first STDP synapse layer."""

    def __init__(
        self,
        input_size: int = 700,
        hidden_size1: int = 512,
        hidden_size2: int = 256,
        output_size: int = 20,
        beta: float = 0.9,
        threshold: float = 1.0,
        stdp_params: dict[str, float] | None = None,
        dt_ms: float = 1.0,
        stdp_lr: float = 1e-6,
        stdp_ltd_scale: float = 1.0,
        hidden_current_scale: float = 1.0,
        recurrent_current_scale: float = 0.05,
        readout: str = "membrane",
    ) -> None:
        super().__init__()
        if readout not in {"spike", "membrane", "hybrid"}:
            raise ValueError(f"Unknown readout mode: {readout}")
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
        self.rec1 = nn.Linear(hidden_size1, hidden_size1, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.rec2 = nn.Linear(hidden_size2, hidden_size2, bias=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        nn.init.normal_(self.rec1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.rec2.weight, mean=0.0, std=0.02)

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.hidden_current_scale = float(hidden_current_scale)
        self.recurrent_current_scale = float(recurrent_current_scale)
        self.readout = readout
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
        prev_spk1 = torch.zeros(batch_size, self.hidden_size1, device=device)
        prev_spk2 = torch.zeros(batch_size, self.hidden_size2, device=device)
        pre_trace = torch.zeros(batch_size, self.stdp1.in_features, device=device)
        post_trace = torch.zeros(batch_size, self.stdp1.out_features, device=device)

        outputs = []
        for t in range(time_steps):
            xt = x[:, t, :]
            i1 = self.stdp1(xt) + self.rec1(prev_spk1) * self.recurrent_current_scale
            spk1, mem1 = self.lif1(i1, mem1)

            if self.stdp_enabled:
                pre_trace, post_trace = self.stdp1.update_stdp(
                    pre_spike=xt > 0,
                    post_spike=spk1,
                    pre_trace=pre_trace,
                    post_trace=post_trace,
                )

            i2 = (
                self.fc2(spk1) * self.hidden_current_scale
                + self.rec2(prev_spk2) * self.recurrent_current_scale
            )
            spk2, mem2 = self.lif2(i2, mem2)
            if self.readout == "membrane":
                readout = mem2
            elif self.readout == "hybrid":
                readout = spk2 + mem2
            else:
                readout = spk2
            outputs.append(self.fc3(readout))

            prev_spk1 = spk1
            prev_spk2 = spk2

        return torch.stack(outputs, dim=1).mean(dim=1)

    @torch.no_grad()
    def spike_stats(self, x: Tensor) -> dict[str, float]:
        batch_size, time_steps, _ = x.shape
        device = x.device

        mem1 = torch.zeros(batch_size, self.hidden_size1, device=device)
        mem2 = torch.zeros(batch_size, self.hidden_size2, device=device)
        prev_spk1 = torch.zeros(batch_size, self.hidden_size1, device=device)
        prev_spk2 = torch.zeros(batch_size, self.hidden_size2, device=device)
        hidden1_total = 0.0
        hidden2_total = 0.0
        hidden1_any = torch.zeros(batch_size, self.hidden_size1, dtype=torch.bool, device=device)
        hidden2_any = torch.zeros(batch_size, self.hidden_size2, dtype=torch.bool, device=device)
        mem1_abs_sum = 0.0
        mem2_abs_sum = 0.0
        mem1_max = 0.0
        mem2_max = 0.0

        for t in range(time_steps):
            xt = x[:, t, :]
            i1 = self.stdp1(xt) + self.rec1(prev_spk1) * self.recurrent_current_scale
            spk1, mem1 = self.lif1(i1, mem1)
            i2 = (
                self.fc2(spk1) * self.hidden_current_scale
                + self.rec2(prev_spk2) * self.recurrent_current_scale
            )
            spk2, mem2 = self.lif2(i2, mem2)

            hidden1_total += float(spk1.sum().item())
            hidden2_total += float(spk2.sum().item())
            hidden1_any |= spk1.bool()
            hidden2_any |= spk2.bool()
            mem1_abs_sum += float(mem1.abs().sum().item())
            mem2_abs_sum += float(mem2.abs().sum().item())
            mem1_max = max(mem1_max, float(mem1.abs().max().item()))
            mem2_max = max(mem2_max, float(mem2.abs().max().item()))

            prev_spk1 = spk1
            prev_spk2 = spk2

        input_total = float((x > 0).sum().item())
        input_rate = input_total / max(x.numel(), 1)
        hidden1_count = batch_size * time_steps * self.hidden_size1
        hidden2_count = batch_size * time_steps * self.hidden_size2
        return {
            "input_spike_rate": input_rate,
            "input_spikes_per_sample": input_total / max(batch_size, 1),
            "hidden1_spike_rate": hidden1_total / max(hidden1_count, 1),
            "hidden1_spikes_per_sample": hidden1_total / max(batch_size, 1),
            "hidden1_active_fraction": float(hidden1_any.float().mean().item()),
            "hidden2_spike_rate": hidden2_total / max(hidden2_count, 1),
            "hidden2_spikes_per_sample": hidden2_total / max(batch_size, 1),
            "hidden2_active_fraction": float(hidden2_any.float().mean().item()),
            "mem1_abs_mean": mem1_abs_sum / max(hidden1_count, 1),
            "mem1_abs_max": mem1_max,
            "mem2_abs_mean": mem2_abs_sum / max(hidden2_count, 1),
            "mem2_abs_max": mem2_max,
        }


class TemporalConvLifModelSTDP(nn.Module):
    """Temporal-convolution LIF classifier with a first STDP synapse layer."""

    def __init__(
        self,
        input_size: int = 700,
        hidden_size1: int = 512,
        hidden_size2: int = 256,
        output_size: int = 20,
        beta: float = 0.9,
        threshold: float = 1.0,
        stdp_params: dict[str, float] | None = None,
        dt_ms: float = 1.0,
        stdp_lr: float = 1e-6,
        stdp_ltd_scale: float = 1.0,
        hidden_current_scale: float = 1.0,
        readout: str = "membrane",
        temporal_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if readout not in {"spike", "membrane", "hybrid"}:
            raise ValueError(f"Unknown readout mode: {readout}")
        if temporal_kernel_size < 1 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer")
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
        self.temporal = nn.Conv1d(
            hidden_size1,
            hidden_size1,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2,
            groups=hidden_size1,
            bias=False,
        )
        self.temporal_mix = nn.Linear(hidden_size1, hidden_size2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.hidden_current_scale = float(hidden_current_scale)
        self.readout = readout
        self.stdp_enabled = False

    def set_stdp(self, enabled: bool, lr_stdp: float | None = None) -> None:
        self.stdp_enabled = enabled
        if lr_stdp is not None:
            self.stdp1.lr_stdp = float(lr_stdp)

    def encode_first_layer(self, x: Tensor) -> Tensor:
        batch_size, time_steps, _ = x.shape
        device = x.device
        mem1 = torch.zeros(batch_size, self.hidden_size1, device=device)
        pre_trace = torch.zeros(batch_size, self.stdp1.in_features, device=device)
        post_trace = torch.zeros(batch_size, self.stdp1.out_features, device=device)
        spikes = []

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
            spikes.append(spk1)

        return torch.stack(spikes, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        spk1_seq = self.encode_first_layer(x)
        temporal_features = self.temporal(spk1_seq.transpose(1, 2)).transpose(1, 2)
        batch_size, time_steps, _ = temporal_features.shape
        device = x.device
        mem2 = torch.zeros(batch_size, self.hidden_size2, device=device)

        outputs = []
        for t in range(time_steps):
            i2 = self.temporal_mix(temporal_features[:, t, :]) * self.hidden_current_scale
            spk2, mem2 = self.lif2(i2, mem2)
            if self.readout == "membrane":
                readout = mem2
            elif self.readout == "hybrid":
                readout = spk2 + mem2
            else:
                readout = spk2
            outputs.append(self.fc3(readout))

        return torch.stack(outputs, dim=1).mean(dim=1)

    @torch.no_grad()
    def spike_stats(self, x: Tensor) -> dict[str, float]:
        spk1_seq = self.encode_first_layer(x)
        temporal_features = self.temporal(spk1_seq.transpose(1, 2)).transpose(1, 2)
        batch_size, time_steps, _ = temporal_features.shape
        device = x.device
        mem2 = torch.zeros(batch_size, self.hidden_size2, device=device)
        hidden2_total = 0.0
        hidden2_any = torch.zeros(batch_size, self.hidden_size2, dtype=torch.bool, device=device)
        mem2_abs_sum = 0.0
        mem2_max = 0.0

        for t in range(time_steps):
            i2 = self.temporal_mix(temporal_features[:, t, :]) * self.hidden_current_scale
            spk2, mem2 = self.lif2(i2, mem2)
            hidden2_total += float(spk2.sum().item())
            hidden2_any |= spk2.bool()
            mem2_abs_sum += float(mem2.abs().sum().item())
            mem2_max = max(mem2_max, float(mem2.abs().max().item()))

        input_total = float((x > 0).sum().item())
        hidden1_total = float(spk1_seq.sum().item())
        hidden1_any = spk1_seq.bool().any(dim=1)
        input_rate = input_total / max(x.numel(), 1)
        hidden1_count = batch_size * time_steps * self.hidden_size1
        hidden2_count = batch_size * time_steps * self.hidden_size2
        return {
            "input_spike_rate": input_rate,
            "input_spikes_per_sample": input_total / max(batch_size, 1),
            "hidden1_spike_rate": hidden1_total / max(hidden1_count, 1),
            "hidden1_spikes_per_sample": hidden1_total / max(batch_size, 1),
            "hidden1_active_fraction": float(hidden1_any.float().mean().item()),
            "hidden2_spike_rate": hidden2_total / max(hidden2_count, 1),
            "hidden2_spikes_per_sample": hidden2_total / max(batch_size, 1),
            "hidden2_active_fraction": float(hidden2_any.float().mean().item()),
            "mem2_abs_mean": mem2_abs_sum / max(hidden2_count, 1),
            "mem2_abs_max": mem2_max,
        }


class SpikingTemporalConvBlock(nn.Module):
    """Temporal convolution current followed by LIF dynamics."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        beta: float,
        threshold: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.lif = snn.Leaky(beta=beta, threshold=threshold)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        current = self.bn(self.conv(x.transpose(1, 2)))
        residual = self.residual(x.transpose(1, 2))
        current = (current + residual).transpose(1, 2)
        batch_size, time_steps, _ = current.shape
        mem = torch.zeros(batch_size, self.out_channels, device=current.device)
        spikes = []
        mems = []
        for t in range(time_steps):
            spk, mem = self.lif(current[:, t, :], mem)
            spikes.append(spk)
            mems.append(mem)
        return self.dropout(torch.stack(spikes, dim=1)), torch.stack(mems, dim=1)


class SpikingTemporalConvLifModelSTDP(nn.Module):
    """Spiking temporal-convolution classifier with a first STDP synapse layer."""

    def __init__(
        self,
        input_size: int = 700,
        hidden_size1: int = 512,
        hidden_size2: int = 256,
        output_size: int = 20,
        beta: float = 0.9,
        threshold: float = 1.0,
        stdp_params: dict[str, float] | None = None,
        dt_ms: float = 1.0,
        stdp_lr: float = 1e-6,
        stdp_ltd_scale: float = 1.0,
        readout: str = "membrane",
        temporal_kernel_size: int = 5,
        temporal_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if readout not in {"spike", "membrane", "hybrid"}:
            raise ValueError(f"Unknown readout mode: {readout}")
        if temporal_kernel_size < 1 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer")
        if temporal_blocks < 1:
            raise ValueError("temporal_blocks must be >= 1")
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
        channels = [hidden_size1] + [hidden_size2] * temporal_blocks
        self.temporal_blocks = nn.ModuleList(
            [
                SpikingTemporalConvBlock(
                    in_channels=channels[idx],
                    out_channels=channels[idx + 1],
                    kernel_size=temporal_kernel_size,
                    dilation=2**idx,
                    beta=beta,
                    threshold=threshold,
                    dropout=dropout,
                )
                for idx in range(temporal_blocks)
            ]
        )
        self.fc = nn.Linear(hidden_size2, output_size)

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.readout = readout
        self.stdp_enabled = False

    def set_stdp(self, enabled: bool, lr_stdp: float | None = None) -> None:
        self.stdp_enabled = enabled
        if lr_stdp is not None:
            self.stdp1.lr_stdp = float(lr_stdp)

    def encode_first_layer(self, x: Tensor) -> Tensor:
        batch_size, time_steps, _ = x.shape
        device = x.device
        mem1 = torch.zeros(batch_size, self.hidden_size1, device=device)
        pre_trace = torch.zeros(batch_size, self.stdp1.in_features, device=device)
        post_trace = torch.zeros(batch_size, self.stdp1.out_features, device=device)
        spikes = []

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
            spikes.append(spk1)

        return torch.stack(spikes, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        spk_seq = self.encode_first_layer(x)
        mem_seq = None
        for block in self.temporal_blocks:
            spk_seq, mem_seq = block(spk_seq)
        if self.readout == "membrane":
            readout_seq = mem_seq
        elif self.readout == "hybrid":
            readout_seq = spk_seq + mem_seq
        else:
            readout_seq = spk_seq
        return self.fc(readout_seq.mean(dim=1))

    @torch.no_grad()
    def spike_stats(self, x: Tensor) -> dict[str, float]:
        spk1_seq = self.encode_first_layer(x)
        spk_seq = spk1_seq
        mem_seq = None
        for block in self.temporal_blocks:
            spk_seq, mem_seq = block(spk_seq)

        batch_size, time_steps, _ = x.shape
        input_total = float((x > 0).sum().item())
        hidden1_total = float(spk1_seq.sum().item())
        hidden2_total = float(spk_seq.sum().item())
        hidden1_any = spk1_seq.bool().any(dim=1)
        hidden2_any = spk_seq.bool().any(dim=1)
        hidden1_count = batch_size * time_steps * self.hidden_size1
        hidden2_count = batch_size * time_steps * self.hidden_size2
        stats = {
            "input_spike_rate": input_total / max(x.numel(), 1),
            "input_spikes_per_sample": input_total / max(batch_size, 1),
            "hidden1_spike_rate": hidden1_total / max(hidden1_count, 1),
            "hidden1_spikes_per_sample": hidden1_total / max(batch_size, 1),
            "hidden1_active_fraction": float(hidden1_any.float().mean().item()),
            "hidden2_spike_rate": hidden2_total / max(hidden2_count, 1),
            "hidden2_spikes_per_sample": hidden2_total / max(batch_size, 1),
            "hidden2_active_fraction": float(hidden2_any.float().mean().item()),
        }
        if mem_seq is not None:
            stats["mem2_abs_mean"] = float(mem_seq.abs().mean().item())
            stats["mem2_abs_max"] = float(mem_seq.abs().max().item())
        return stats


class SpikingTemporalConvLifModelSTDPV2(SpikingTemporalConvLifModelSTDP):
    """Spiking temporal-convolution STDP model with richer temporal pooling."""

    def __init__(
        self,
        input_size: int = 700,
        hidden_size1: int = 512,
        hidden_size2: int = 256,
        output_size: int = 20,
        beta: float = 0.9,
        threshold: float = 1.0,
        stdp_params: dict[str, float] | None = None,
        dt_ms: float = 1.0,
        stdp_lr: float = 1e-6,
        stdp_ltd_scale: float = 1.0,
        readout: str = "membrane",
        temporal_kernel_size: int = 5,
        temporal_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            output_size=output_size,
            beta=beta,
            threshold=threshold,
            stdp_params=stdp_params,
            dt_ms=dt_ms,
            stdp_lr=stdp_lr,
            stdp_ltd_scale=stdp_ltd_scale,
            readout=readout,
            temporal_kernel_size=temporal_kernel_size,
            temporal_blocks=temporal_blocks,
            dropout=dropout,
        )
        self.attention_score = nn.Linear(hidden_size2, 1)
        self.pool_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_size2 * 4, output_size)

    def forward(self, x: Tensor) -> Tensor:
        spk_seq = self.encode_first_layer(x)
        mem_seq = None
        for block in self.temporal_blocks:
            spk_seq, mem_seq = block(spk_seq)
        if self.readout == "membrane":
            readout_seq = mem_seq
        elif self.readout == "hybrid":
            readout_seq = spk_seq + mem_seq
        else:
            readout_seq = spk_seq

        mean_pool = readout_seq.mean(dim=1)
        max_pool = readout_seq.amax(dim=1)
        last_pool = readout_seq[:, -1, :]
        attention = torch.softmax(self.attention_score(readout_seq), dim=1)
        attention_pool = (readout_seq * attention).sum(dim=1)
        pooled = torch.cat([mean_pool, max_pool, last_pool, attention_pool], dim=1)
        return self.fc(self.pool_dropout(pooled))


class DualPathSpikingTemporalConvLifModelSTDP(nn.Module):
    """Dual-path SNN with an STDP-LIF path and a raw-spike temporal path."""

    def __init__(
        self,
        input_size: int = 700,
        hidden_size1: int = 512,
        hidden_size2: int = 256,
        output_size: int = 20,
        beta: float = 0.9,
        threshold: float = 1.0,
        stdp_params: dict[str, float] | None = None,
        dt_ms: float = 1.0,
        stdp_lr: float = 1e-6,
        stdp_ltd_scale: float = 1.0,
        readout: str = "membrane",
        temporal_kernel_size: int = 5,
        temporal_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if readout not in {"spike", "membrane", "hybrid"}:
            raise ValueError(f"Unknown readout mode: {readout}")
        if temporal_kernel_size < 1 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer")
        if temporal_blocks < 1:
            raise ValueError("temporal_blocks must be >= 1")
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

        stdp_channels = [hidden_size1] + [hidden_size2] * temporal_blocks
        raw_channels = [input_size] + [hidden_size2] * temporal_blocks
        self.stdp_temporal_blocks = nn.ModuleList(
            [
                SpikingTemporalConvBlock(
                    in_channels=stdp_channels[idx],
                    out_channels=stdp_channels[idx + 1],
                    kernel_size=temporal_kernel_size,
                    dilation=2**idx,
                    beta=beta,
                    threshold=threshold,
                    dropout=dropout,
                )
                for idx in range(temporal_blocks)
            ]
        )
        self.raw_temporal_blocks = nn.ModuleList(
            [
                SpikingTemporalConvBlock(
                    in_channels=raw_channels[idx],
                    out_channels=raw_channels[idx + 1],
                    kernel_size=temporal_kernel_size,
                    dilation=2**idx,
                    beta=beta,
                    threshold=threshold,
                    dropout=dropout,
                )
                for idx in range(temporal_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_size2 * 2, output_size)

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.readout = readout
        self.stdp_enabled = False

    def set_stdp(self, enabled: bool, lr_stdp: float | None = None) -> None:
        self.stdp_enabled = enabled
        if lr_stdp is not None:
            self.stdp1.lr_stdp = float(lr_stdp)

    def encode_stdp_path(self, x: Tensor) -> Tensor:
        batch_size, time_steps, _ = x.shape
        device = x.device
        mem1 = torch.zeros(batch_size, self.hidden_size1, device=device)
        pre_trace = torch.zeros(batch_size, self.stdp1.in_features, device=device)
        post_trace = torch.zeros(batch_size, self.stdp1.out_features, device=device)
        spikes = []

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
            spikes.append(spk1)
        return torch.stack(spikes, dim=1)

    def run_temporal_path(
        self,
        sequence: Tensor,
        blocks: nn.ModuleList,
    ) -> tuple[Tensor, Tensor]:
        spk_seq = sequence
        mem_seq = None
        for block in blocks:
            spk_seq, mem_seq = block(spk_seq)
        if mem_seq is None:
            raise RuntimeError("Temporal path has no blocks")
        return spk_seq, mem_seq

    def choose_readout(self, spk_seq: Tensor, mem_seq: Tensor) -> Tensor:
        if self.readout == "membrane":
            return mem_seq
        if self.readout == "hybrid":
            return spk_seq + mem_seq
        return spk_seq

    def forward(self, x: Tensor) -> Tensor:
        stdp_spk = self.encode_stdp_path(x)
        stdp_spk, stdp_mem = self.run_temporal_path(stdp_spk, self.stdp_temporal_blocks)
        raw_spk, raw_mem = self.run_temporal_path(x, self.raw_temporal_blocks)

        stdp_readout = self.choose_readout(stdp_spk, stdp_mem)
        raw_readout = self.choose_readout(raw_spk, raw_mem)
        fused = torch.cat([stdp_readout, raw_readout], dim=2)
        return self.fc(self.dropout(fused.mean(dim=1)))

    @torch.no_grad()
    def spike_stats(self, x: Tensor) -> dict[str, float]:
        stdp_first = self.encode_stdp_path(x)
        stdp_spk, stdp_mem = self.run_temporal_path(stdp_first, self.stdp_temporal_blocks)
        raw_spk, raw_mem = self.run_temporal_path(x, self.raw_temporal_blocks)

        batch_size, time_steps, _ = x.shape
        input_total = float((x > 0).sum().item())
        hidden1_total = float(stdp_first.sum().item())
        stdp_total = float(stdp_spk.sum().item())
        raw_total = float(raw_spk.sum().item())
        hidden1_count = batch_size * time_steps * self.hidden_size1
        hidden2_count = batch_size * time_steps * self.hidden_size2
        return {
            "input_spike_rate": input_total / max(x.numel(), 1),
            "input_spikes_per_sample": input_total / max(batch_size, 1),
            "hidden1_spike_rate": hidden1_total / max(hidden1_count, 1),
            "hidden1_spikes_per_sample": hidden1_total / max(batch_size, 1),
            "stdp_path_spike_rate": stdp_total / max(hidden2_count, 1),
            "raw_path_spike_rate": raw_total / max(hidden2_count, 1),
            "stdp_mem_abs_mean": float(stdp_mem.abs().mean().item()),
            "stdp_mem_abs_max": float(stdp_mem.abs().max().item()),
            "raw_mem_abs_mean": float(raw_mem.abs().mean().item()),
            "raw_mem_abs_max": float(raw_mem.abs().max().item()),
        }


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
