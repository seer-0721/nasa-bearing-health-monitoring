from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from torch import Tensor

from .stdp_snn import STDPLayer


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + self.net(x))


class SHDTemporalCNN(nn.Module):
    """Temporal CNN baseline for SHD spike bins.

    This is a non-SNN upper-bound model. It is useful for validating SHD loading,
    binning, and the supervised training setup before attributing low accuracy to
    SNN/STDP constraints.
    """

    def __init__(
        self,
        input_size: int = 700,
        width: int = 256,
        num_classes: int = 20,
        blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.stem = nn.Sequential(
            nn.Conv1d(input_size, width, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        dilations = [2**idx for idx in range(blocks)]
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(
                    channels=width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(width * 2, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # SHD loader returns [batch, time, input_size]; Conv1d expects channels first.
        x = x.transpose(1, 2)
        features = self.blocks(self.stem(x))
        avg_pool = features.mean(dim=-1)
        max_pool = features.amax(dim=-1)
        return self.fc(self.dropout(torch.cat([avg_pool, max_pool], dim=1)))


class STDPTemporalCNN(nn.Module):
    """Hybrid SHD model with an STDP-LIF front-end and non-spiking temporal CNN head."""

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
        blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
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
        self.stem = nn.Sequential(
            nn.Conv1d(hidden_size1, hidden_size2, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(
                    channels=hidden_size2,
                    kernel_size=kernel_size,
                    dilation=2**idx,
                    dropout=dropout,
                )
                for idx in range(blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_size2 * 2, output_size)

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
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
        spk1 = self.encode_first_layer(x).transpose(1, 2)
        features = self.blocks(self.stem(spk1))
        avg_pool = features.mean(dim=-1)
        max_pool = features.amax(dim=-1)
        return self.fc(self.dropout(torch.cat([avg_pool, max_pool], dim=1)))

    @torch.no_grad()
    def spike_stats(self, x: Tensor) -> dict[str, float]:
        spk1 = self.encode_first_layer(x)
        batch_size, time_steps, _ = x.shape
        input_total = float((x > 0).sum().item())
        hidden1_total = float(spk1.sum().item())
        hidden1_any = spk1.bool().any(dim=1)
        hidden1_count = batch_size * time_steps * self.hidden_size1
        return {
            "input_spike_rate": input_total / max(x.numel(), 1),
            "input_spikes_per_sample": input_total / max(batch_size, 1),
            "hidden1_spike_rate": hidden1_total / max(hidden1_count, 1),
            "hidden1_spikes_per_sample": hidden1_total / max(batch_size, 1),
            "hidden1_active_fraction": float(hidden1_any.float().mean().item()),
        }
