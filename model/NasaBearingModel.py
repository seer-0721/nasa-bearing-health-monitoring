import torch.nn as nn
import snntorch as snn
import torch
from torch import Tensor

from .stdp_snn import STDPLayer


class NasaBearingSNN(nn.Module):
    """STDP/LIF model for NASA bearing feature windows.

    The model has two supervised outputs:
    - health score: continuous value from 1.0 healthy to 0.0 failed
    - fault part: normal / component fault classes

    Inputs are normalized feature vectors in [0, 1]. They are encoded as
    deterministic rate spikes inside the model.
    """

    def __init__(
        self,
        input_size: int = 80,
        hidden_size1: int = 256,
        hidden_size2: int = 128,
        fault_classes: int = 6,
        time_steps: int = 16,
        beta: float = 0.9,
        threshold: float = 1.0,
        stdp_params: dict[str, float] | None = None,
        stdp_lr: float = 1e-6,
        stdp_ltd_scale: float = 1.0,
        readout: str = "hybrid",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if fault_classes < 2:
            raise ValueError("fault_classes must be >= 2")
        if time_steps < 1:
            raise ValueError("time_steps must be >= 1")
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
            lr_stdp=stdp_lr,
            ltd_scale=stdp_ltd_scale,
        )
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.health_score_head = nn.Linear(hidden_size2, 1)
        self.fault_head = nn.Linear(hidden_size2, fault_classes)

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.fault_classes = fault_classes
        self.time_steps = time_steps
        self.readout = readout
        self.stdp_enabled = False

    def set_stdp(self, enabled: bool, lr_stdp: float | None = None) -> None:
        self.stdp_enabled = enabled
        if lr_stdp is not None:
            self.stdp1.lr_stdp = float(lr_stdp)

    def encode(self, x: Tensor) -> Tensor:
        x = x.clamp(0.0, 1.0)
        steps = torch.arange(
            self.time_steps + 1,
            device=x.device,
            dtype=x.dtype,
        ).view(1, self.time_steps + 1, 1)
        cumulative = torch.floor(x.unsqueeze(1) * steps)
        return (cumulative[:, 1:] > cumulative[:, :-1]).float()

    def forward_features(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            spikes = self.encode(x)
        elif x.dim() == 3:
            spikes = x.float()
        else:
            raise ValueError(f"Expected [batch, features] or [batch, time, features], got {x.shape}")

        batch_size, time_steps, _ = spikes.shape
        device = spikes.device
        mem1 = torch.zeros(batch_size, self.hidden_size1, device=device)
        mem2 = torch.zeros(batch_size, self.hidden_size2, device=device)
        pre_trace = torch.zeros(batch_size, self.stdp1.in_features, device=device)
        post_trace = torch.zeros(batch_size, self.stdp1.out_features, device=device)

        readouts = []
        for t in range(time_steps):
            xt = spikes[:, t, :]
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
            if self.readout == "membrane":
                readout = mem2
            elif self.readout == "hybrid":
                readout = spk2 + mem2
            else:
                readout = spk2
            readouts.append(readout)

        return self.dropout(torch.stack(readouts, dim=1).mean(dim=1))

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        features = self.forward_features(x)
        return {
            "health_score_pred": self.health_score_head(features).sigmoid().squeeze(1),
            "fault_logits": self.fault_head(features),
        }

    @torch.no_grad()
    def predict(self, x: Tensor) -> dict[str, Tensor]:
        outputs = self.forward(x)
        fault_probs = outputs["fault_logits"].softmax(dim=1)
        health_score = outputs["health_score_pred"]
        return {
            **outputs,
            "health_score": health_score,
            "fault_class": fault_probs.argmax(dim=1),
        }


NasaBearing = NasaBearingSNN
