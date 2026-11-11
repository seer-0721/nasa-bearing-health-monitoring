from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS_CSV = ROOT / "docs" / "assets" / "nasa_bearing_training_metrics.csv"
OUTPUT_DIR = ROOT / "docs" / "assets"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "Times"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 1200,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")


def plot_accuracy_curves(df: pd.DataFrame) -> None:
    colors = {
        "train_health": "#1f77b4",
        "val_health": "#d62728",
        "train_fault": "#2ca02c",
        "val_fault": "#9467bd",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharex=True, sharey=True)

    axes[0].plot(
        df["epoch"],
        df["train_health_acc"],
        color=colors["train_health"],
        linewidth=1.7,
        label="Train health",
    )
    axes[0].plot(
        df["epoch"],
        df["val_health_acc"],
        color=colors["val_health"],
        linewidth=1.7,
        label="Validation health",
    )
    axes[0].plot(
        df["epoch"],
        df["train_fault_acc"],
        color=colors["train_fault"],
        linewidth=1.7,
        linestyle="--",
        label="Train fault",
    )
    axes[0].plot(
        df["epoch"],
        df["val_fault_acc"],
        color=colors["val_fault"],
        linewidth=1.7,
        linestyle="--",
        label="Validation fault",
    )
    axes[0].set_title("Overall accuracy")

    axes[1].plot(
        df["epoch"],
        df["train_health_macro_acc"],
        color=colors["train_health"],
        linewidth=1.7,
        label="Train health",
    )
    axes[1].plot(
        df["epoch"],
        df["val_health_macro_acc"],
        color=colors["val_health"],
        linewidth=1.7,
        label="Validation health",
    )
    axes[1].plot(
        df["epoch"],
        df["train_fault_macro_acc"],
        color=colors["train_fault"],
        linewidth=1.7,
        linestyle="--",
        label="Train fault",
    )
    axes[1].plot(
        df["epoch"],
        df["val_fault_macro_acc"],
        color=colors["val_fault"],
        linewidth=1.7,
        linestyle="--",
        label="Validation fault",
    )
    axes[1].set_title("Macro accuracy")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Accuracy")
        axis.set_xlim(1, 50)
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.75)
        axis.legend(loc="lower right", fontsize=8)

    save_figure(fig, "nasa_bearing_accuracy_curves")
    plt.close(fig)


def plot_loss_curve(df: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(5.0, 3.4))
    axis.plot(
        df["epoch"],
        df["train_loss"],
        color="#1f77b4",
        linewidth=1.8,
        label="Train loss",
    )
    axis.plot(
        df["epoch"],
        df["val_loss"],
        color="#d62728",
        linewidth=1.8,
        label="Validation loss",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_xlim(1, 50)
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.75)
    axis.legend(loc="upper right", fontsize=8)

    save_figure(fig, "nasa_bearing_loss_curves")
    plt.close(fig)


def plot_learning_rate_curve(df: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(5.0, 3.4))
    axis.plot(
        df["epoch"],
        df["lr"],
        color="#4d4d4d",
        linewidth=1.8,
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Learning rate")
    axis.set_xlim(1, 50)
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.75)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    save_figure(fig, "nasa_bearing_learning_rate_curve")
    plt.close(fig)


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS_CSV)
    plot_accuracy_curves(df)
    plot_loss_curve(df)
    plot_learning_rate_curve(df)


if __name__ == "__main__":
    main()
