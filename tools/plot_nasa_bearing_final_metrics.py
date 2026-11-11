from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS_CSV = ROOT / "docs" / "assets" / "nasa_bearing_final_metrics.csv"
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
            "figure.dpi": 180,
            "savefig.dpi": 1200,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")


def plot_final_metrics(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True)

    axes[0].plot(
        df["epoch"],
        df["val_health_mae"],
        color="#1f77b4",
        linewidth=1.9,
        marker="o",
        markersize=2.8,
        label="Validation",
    )
    axes[0].plot(
        df["epoch"],
        df["train_health_mae"],
        color="#8c8c8c",
        linewidth=1.4,
        linestyle="--",
        label="Train",
    )
    axes[0].set_title("Health MAE")
    axes[0].set_ylabel("MAE")

    axes[1].plot(
        df["epoch"],
        df["val_health_acc_at_0_1"],
        color="#2ca02c",
        linewidth=1.9,
        marker="o",
        markersize=2.8,
        label="Validation",
    )
    axes[1].plot(
        df["epoch"],
        df["train_health_acc_at_0_1"],
        color="#8c8c8c",
        linewidth=1.4,
        linestyle="--",
        label="Train",
    )
    axes[1].set_title("Health Acc@0.1")
    axes[1].set_ylabel("Accuracy")

    axes[2].plot(
        df["epoch"],
        df["val_fault_macro_acc"],
        color="#d62728",
        linewidth=1.9,
        marker="o",
        markersize=2.8,
        label="Validation",
    )
    axes[2].plot(
        df["epoch"],
        df["train_fault_macro_acc"],
        color="#8c8c8c",
        linewidth=1.4,
        linestyle="--",
        label="Train",
    )
    axes[2].set_title("Fault Macro Accuracy")
    axes[2].set_ylabel("Accuracy")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_xlim(float(df["epoch"].min()), float(df["epoch"].max()))
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.75)
        axis.legend(loc="best", fontsize=8)

    axes[0].set_ylim(0.035, 0.125)
    axes[1].set_ylim(0.55, 0.90)
    axes[2].set_ylim(0.995, 1.0)
    save_figure(fig, "nasa_bearing_final_metrics")
    plt.close(fig)


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_final_metrics(pd.read_csv(METRICS_CSV))


if __name__ == "__main__":
    main()
