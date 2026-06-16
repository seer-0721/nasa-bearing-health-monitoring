from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from data import CIFAR10_CLASSES, CIFAR10_INPUT_SIZE, build_cifar10_loaders, encode_spikes
from model import CifarResNet, ConvLifModelSTDP, LifModelSTDP, SpikingConvLifModelSTDP
from utils import load_checkpoint, load_stdp_params, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 STDP/SNN checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--data-root", type=str, default="data/cifar10")
    parser.add_argument("--stdp-params-file", type=str, default="stdp_params.npz")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        choices=["poisson", "deterministic-rate", "latency"],
    )
    parser.add_argument("--eval-repeats", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model_from_checkpoint(
    checkpoint: dict,
    stdp_params_file: str,
    device: torch.device,
) -> nn.Module:
    train_args = checkpoint.get("args", {})
    stdp_params = load_stdp_params(stdp_params_file)
    model_type = train_args.get("model_type", "mlp-stdp")
    if model_type == "cifar-resnet":
        model = CifarResNet(
            num_classes=CIFAR10_CLASSES,
            width=train_args.get("resnet_width", 64),
            blocks_per_stage=train_args.get("resnet_blocks", 3),
            dropout=train_args.get("dropout", 0.0),
        ).to(device)
    elif model_type == "conv-stdp":
        model = ConvLifModelSTDP(
            hidden_size1=train_args.get("hidden_size1", 256),
            hidden_size2=train_args.get("hidden_size2", 128),
            output_size=CIFAR10_CLASSES,
            beta=train_args.get("beta", 0.9),
            threshold=train_args.get("threshold", 1.0),
            stdp_params=stdp_params,
            stdp_lr=train_args.get("stdp_lr", 1e-6),
            conv_channels=train_args.get("conv_channels", 96),
            stdp_current_scale=train_args.get("stdp_current_scale"),
        ).to(device)
    elif model_type == "spiking-conv-stdp":
        model = SpikingConvLifModelSTDP(
            hidden_size1=train_args.get("hidden_size1", 768),
            hidden_size2=train_args.get("hidden_size2", 384),
            output_size=CIFAR10_CLASSES,
            beta=train_args.get("beta", 0.9),
            threshold=train_args.get("threshold", 1.0),
            stdp_params=stdp_params,
            stdp_lr=train_args.get("stdp_lr", 1e-6),
            conv_channels=train_args.get("conv_channels", 128),
            stdp_current_scale=train_args.get("stdp_current_scale"),
            dropout=train_args.get("dropout", 0.0),
        ).to(device)
    else:
        model = LifModelSTDP(
            input_size=CIFAR10_INPUT_SIZE,
            hidden_size1=train_args.get("hidden_size1", 256),
            hidden_size2=train_args.get("hidden_size2", 128),
            output_size=CIFAR10_CLASSES,
            beta=train_args.get("beta", 0.9),
            threshold=train_args.get("threshold", 1.0),
            stdp_params=stdp_params,
            stdp_lr=train_args.get("stdp_lr", 1e-6),
        ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def is_snn_model(train_args: dict) -> bool:
    return train_args.get("model_type", "mlp-stdp") in {
        "mlp-stdp",
        "conv-stdp",
        "spiking-conv-stdp",
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_root: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    time_steps: int,
    encoding: str,
    eval_repeats: int,
    train_args: dict,
) -> tuple[float, float]:
    _, test_loader = build_cifar10_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        download=False,
        normalize=not is_snn_model(train_args),
    )
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_snn_model(train_args):
            logits = model(encode_spikes(images, time_steps, encoding))
            for _ in range(1, eval_repeats):
                logits = logits + model(encode_spikes(images, time_steps, encoding))
            logits = logits / eval_repeats
        else:
            logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    time_steps = args.time_steps or train_args.get("time_steps", 10)
    encoding = args.encoding or train_args.get("encoding", "poisson")
    eval_repeats = args.eval_repeats or train_args.get("eval_repeats", 1)

    model = build_model_from_checkpoint(checkpoint, args.stdp_params_file, device)
    test_loss, test_acc = evaluate(
        model=model,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        time_steps=time_steps,
        encoding=encoding,
        eval_repeats=eval_repeats,
        train_args=train_args,
    )

    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_epoch={checkpoint.get('epoch', 'unknown')}")
    print(f"time_steps={time_steps}")
    print(f"encoding={encoding if is_snn_model(train_args) else 'none'}")
    print(f"eval_repeats={eval_repeats}")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc={test_acc:.4f}")
    if "metrics" in checkpoint:
        metrics = checkpoint["metrics"]
        print(f"saved_test_loss={metrics.get('test_loss', float('nan')):.4f}")
        print(f"saved_test_acc={metrics.get('test_acc', float('nan')):.4f}")


if __name__ == "__main__":
    main()
