from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data import (
    CIFAR10_CLASSES,
    CIFAR10_INPUT_SIZE,
    CIFAR10_MEAN,
    CIFAR10_STD,
    encode_spikes,
)
from model import CifarResNet, ConvLifModelSTDP, LifModelSTDP, SpikingConvLifModelSTDP
from utils import load_checkpoint, load_stdp_params, resolve_device


CIFAR10_LABELS = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR-10 inference from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--data-root", type=str, default="data/cifar10")
    parser.add_argument("--stdp-params-file", type=str, default="stdp_params.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        choices=["poisson", "deterministic-rate", "latency"],
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    time_steps = args.time_steps or train_args.get("time_steps", 10)
    encoding = args.encoding or train_args.get("encoding", "poisson")

    stdp_params = load_stdp_params(args.stdp_params_file)
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
            conv_blocks_per_stage=train_args.get("conv_blocks_per_stage", 1),
            stdp_current_scale=train_args.get("stdp_current_scale"),
            dropout=train_args.get("dropout", 0.0),
            readout=train_args.get("readout", "spike"),
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

    transform_steps: list[transforms.Transform] = [transforms.ToTensor()]
    if model_type == "cifar-resnet":
        transform_steps.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    dataset = datasets.CIFAR10(
        root=args.data_root,
        train=False,
        transform=transforms.Compose(transform_steps),
        download=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    seen = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if model_type in {"mlp-stdp", "conv-stdp", "spiking-conv-stdp"}:
                logits = model(encode_spikes(images, time_steps, encoding))
            else:
                logits = model(images)
            preds = logits.argmax(dim=1)

            for pred, target in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                print(f"pred={CIFAR10_LABELS[pred]} target={CIFAR10_LABELS[target]}")
                correct += int(pred == target)
                seen += 1
                if seen >= args.limit:
                    print(f"shown_acc={correct / seen:.4f}")
                    return


if __name__ == "__main__":
    main()
