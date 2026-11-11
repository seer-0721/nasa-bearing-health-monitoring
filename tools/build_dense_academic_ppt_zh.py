from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nasa-bearing")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import DEFAULT_FAULT_MAP, NASA_BEARING_FAULT_PARTS, build_nasa_bearing_loaders
from eval import build_model
from utils import load_checkpoint, resolve_device, set_seed


DATA_ROOT = ROOT / "data" / "NasaBearingDataset"
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
PPT_ASSETS = DOCS / "ppt_assets_dense_zh"
OUTPUT = DOCS / "nasa_bearing_snn_academic_report_dense_zh.pptx"
CHECKPOINT = ROOT / "checkpoints" / "nasa_bearing_snn_health_reg_v3" / "best.pt"

FONT = "Microsoft YaHei"
FALLBACK_FONT = "Noto Sans CJK SC"
INK = RGBColor(25, 28, 32)
TEXT = RGBColor(55, 65, 75)
MUTED = RGBColor(100, 110, 120)
GRID = RGBColor(218, 224, 230)
WHITE = RGBColor(255, 255, 255)
PALE = RGBColor(247, 248, 249)
BLUE = RGBColor(76, 106, 132)
GREEN = RGBColor(99, 128, 111)
RED = RGBColor(148, 91, 84)
GOLD = RGBColor(158, 132, 91)

P_BLUE = "#4c6a84"
P_GREEN = "#63806f"
P_RED = "#945b54"
P_GOLD = "#9e845b"
P_GREY = "#7b8490"
P_LIGHT = "#f2f4f6"


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, FALLBACK_FONT, "Noto Sans", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 700,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#5b6470",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
        }
    )


def save_fig(fig: plt.Figure, name: str) -> Path:
    PPT_ASSETS.mkdir(parents=True, exist_ok=True)
    path = PPT_ASSETS / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def line_count(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8")) - 1


def sample_csv(path: Path, n: int, seed: int) -> pd.DataFrame:
    rows = line_count(path)
    if rows <= n:
        return pd.read_csv(path)
    rng = np.random.default_rng(seed)
    selected = set(rng.choice(np.arange(1, rows + 1), size=n, replace=False).tolist())
    return pd.read_csv(path, skiprows=lambda idx: idx != 0 and idx not in selected)


def clean_axis(ax) -> None:
    ax.grid(True, linestyle=":", linewidth=0.55, color="#cbd5df", alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_dataset_overview(good_rows: int, fault_rows: int) -> Path:
    labels = ["Good windows", "Fault windows", "Split normal", "Split inner", "Split rolling"]
    values = [good_rows, fault_rows, good_rows * 4 + fault_rows * 2, fault_rows, fault_rows]
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.bar(labels, values, color=[P_BLUE, P_RED, P_GREY, P_GOLD, P_GREEN], edgecolor="#4b5563", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_title("Dataset source and label distribution")
    ax.set_ylabel("Sample count, log scale")
    ax.tick_params(axis="x", rotation=16)
    for idx, value in enumerate(values):
        ax.text(idx, value * 1.25, f"{value:,}", ha="center", fontsize=8, color="#334155")
    clean_axis(ax)
    return save_fig(fig, "dataset_overview")


def make_processing_and_health(good_rows: int, fault_rows: int) -> Path:
    normal_scores = np.ones(good_rows * 4 + fault_rows * 2, dtype=np.float32)
    fault_scores = 1.0 - np.arange(fault_rows, dtype=np.float32) / max(fault_rows - 1, 1)
    scores = np.concatenate([normal_scores, fault_scores, fault_scores])

    fig = plt.figure(figsize=(12.0, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.75, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    steps = [
        ("Raw\nvibration", "B1-B4"),
        ("Feature\nextraction", "20 features/bearing"),
        ("Per-bearing\nsplit", "4 samples/row"),
        ("Labeling", "health_score\n+ fault_part"),
        ("STDP-SNN", "regression\n+ classification"),
    ]
    xs = np.linspace(0.09, 0.91, len(steps))
    box_w = 0.135
    for i, (title, sub) in enumerate(steps):
        ax0.add_patch(plt.Rectangle((xs[i] - box_w / 2, 0.37), box_w, 0.30, facecolor=P_LIGHT, edgecolor="#7b8490", lw=0.9))
        ax0.text(xs[i], 0.56, title, ha="center", va="center", fontsize=7.0, fontweight="bold", color="#1f2937", linespacing=1.0)
        ax0.text(xs[i], 0.43, sub, ha="center", va="center", fontsize=5.8, color="#4b5563", linespacing=1.0)
        if i < len(steps) - 1:
            ax0.annotate("", xy=(xs[i + 1] - box_w / 2 - 0.012, 0.52), xytext=(xs[i] + box_w / 2 + 0.012, 0.52), arrowprops=dict(arrowstyle="->", lw=1.0, color="#6b7280"))
    ax0.set_title("Dataset processing pipeline")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(scores, bins=np.linspace(0, 1, 12), color=P_BLUE, edgecolor="white")
    ax1.set_yscale("log")
    ax1.set_title("Health-score distribution")
    ax1.set_xlabel("health_score")
    ax1.set_ylabel("Count, log scale")
    clean_axis(ax1)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(np.arange(fault_rows), fault_scores, color=P_RED, linewidth=1.8)
    ax2.set_title("Time-derived target")
    ax2.set_xlabel("Fault-window position")
    ax2.set_ylabel("health_score")
    ax2.set_ylim(-0.04, 1.04)
    clean_axis(ax2)
    return save_fig(fig, "processing_and_health")


def make_raw_waveform() -> Path:
    df = pd.read_csv(DATA_ROOT / "test_final.csv", nrows=2200)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.5, 2.7))
    offsets = [0.0, 1.05, 2.10, 3.15]
    for idx, (col, color) in enumerate(zip(["Bearing1", "Bearing2", "Bearing3", "Bearing4"], [P_BLUE, P_GREEN, P_GOLD, P_RED])):
        y = df[col].to_numpy(dtype=float)
        y = y / max(float(np.std(y)), 1e-6) * 0.18 + offsets[idx]
        ax.plot(x, y, color=color, linewidth=0.7, label=f"B{idx + 1}")
    ax.set_title("Raw vibration segment")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Normalized amplitude + offset")
    ax.set_yticks(offsets)
    ax.set_yticklabels(["B1", "B2", "B3", "B4"])
    ax.legend(ncol=4, fontsize=7.5, loc="upper right")
    clean_axis(ax)
    return save_fig(fig, "raw_waveform")


def make_feature_evidence(good: pd.DataFrame, fault: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(11.0, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.45, 1.05])

    ax0 = fig.add_subplot(gs[0, 0])
    box = ax0.boxplot(
        [good["B3__rms"].dropna().to_numpy(), fault["B3__rms"].dropna().to_numpy(), good["B4__p2p"].dropna().to_numpy(), fault["B4__p2p"].dropna().to_numpy()],
        tick_labels=["B3 RMS\nGood", "B3 RMS\nFault", "B4 P2P\nGood", "B4 P2P\nFault"],
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(box["boxes"], ["#e2e8ee", "#eadbd2", "#e2e8ee", "#eadbd2"]):
        patch.set_facecolor(color)
        patch.set_edgecolor("#5b6470")
    ax0.set_title("Feature separation")
    ax0.set_ylabel("Feature value")
    ax0.tick_params(axis="x", labelsize=7)
    clean_axis(ax0)

    ax1 = fig.add_subplot(gs[0, 1])
    x = np.linspace(0, 1, len(fault))
    for bearing, color in [("B3", P_GOLD), ("B4", P_GREEN)]:
        for feature, style in [("rms", "-"), ("p2p", "--")]:
            y = fault[f"{bearing}__{feature}"].rolling(31, center=True, min_periods=1).mean()
            y = (y - y.min()) / max(float(y.max() - y.min()), 1e-9)
            ax1.plot(x, y, linestyle=style, color=color, linewidth=1.25, label=f"{bearing} {feature}")
    ax1.plot(x, 1 - x, color="#1f2937", linewidth=1.35, label="Target health_score")
    ax1.set_title("Feature evolution along fault timeline")
    ax1.set_xlabel("Normalized fault timeline")
    ax1.set_ylabel("Normalized value")
    ax1.legend(fontsize=6.8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), borderaxespad=0.0)
    clean_axis(ax1)

    ax2 = fig.add_subplot(gs[0, 2])
    cols = [c for c in fault.columns if c.startswith("B3__")]
    labels = [c.split("__", 1)[1] for c in cols]
    corr = fault[cols].corr().to_numpy()
    im = ax2.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.82)
    ax2.set_title("B3 feature correlation")
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels, rotation=60, ha="right", fontsize=5.5)
    ax2.set_yticklabels(labels, fontsize=5.5)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    return save_fig(fig, "feature_evidence")


def make_model_and_stdp() -> Path:
    params = np.load(ROOT / "stdp_params.npz")
    tau_plus = float(params["tau_plus"])
    tau_minus = float(params["tau_minus"])
    fig = plt.figure(figsize=(12.0, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    blocks = [("Input\nfeatures", "#f2f4f6"), ("Spike\ncoding", "#eef3f6"), ("STDP\nlayer", "#f4efe4"), ("LIF\ndynamics", "#eef4ed"), ("Shared\nstate", "#f7f7f7")]
    xs = np.linspace(0.08, 0.64, len(blocks))
    box_w = 0.11
    for i, (label, color) in enumerate(blocks):
        ax0.add_patch(plt.Rectangle((xs[i] - box_w / 2, 0.55), box_w, 0.18, facecolor=color, edgecolor="#7b8490", lw=0.9))
        ax0.text(xs[i], 0.64, label, ha="center", va="center", fontsize=7.4, fontweight="bold", linespacing=1.0)
        if i < len(blocks) - 1:
            ax0.annotate("", xy=(xs[i + 1] - box_w / 2 - 0.012, 0.64), xytext=(xs[i] + box_w / 2 + 0.012, 0.64), arrowprops=dict(arrowstyle="->", lw=1.0, color="#6b7280"))
    heads = [("health_score", "Regression", "#eef4ed", 0.84, 0.78), ("health_stage", "Auxiliary", "#eef3f6", 0.84, 0.56), ("fault_part", "Classification", "#f4e8e5", 0.84, 0.34)]
    for title, sub, color, xh, yh in heads:
        ax0.add_patch(plt.Rectangle((xh - 0.085, yh - 0.055), 0.17, 0.11, facecolor=color, edgecolor="#7b8490", lw=0.9))
        ax0.text(xh, yh + 0.018, title, ha="center", va="center", fontsize=8.0, fontweight="bold")
        ax0.text(xh, yh - 0.027, sub, ha="center", va="center", fontsize=7.0, color="#4b5563")
        ax0.annotate("", xy=(xh - 0.093, yh), xytext=(0.72, 0.64), arrowprops=dict(arrowstyle="->", lw=0.85, color="#7b8490"))
    ax0.set_title("STDP-SNN data flow")

    ax1 = fig.add_subplot(gs[0, 1])
    window = max(tau_plus, tau_minus)
    dt = np.linspace(-window, window, 700)
    dw = np.where(dt >= 0, np.exp(-dt / tau_plus), -np.exp(dt / tau_minus))
    ax1.plot(dt, dw, color=P_BLUE, linewidth=1.6)
    ax1.axhline(0, color="#4b5563", lw=0.8)
    ax1.axvline(0, color="#4b5563", lw=0.8)
    ax1.text(0.03, 0.88, f"tau+={tau_plus:.0f}, tau-={tau_minus:.0f}", transform=ax1.transAxes, fontsize=8, color="#4b5563")
    ax1.set_title("Normalized STDP learning window")
    ax1.set_xlabel("Spike timing difference Delta t")
    ax1.set_ylabel("Normalized update")
    ax1.set_ylim(-1.08, 1.08)
    clean_axis(ax1)
    return save_fig(fig, "model_and_stdp")


def make_training_strategy() -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    labels = ["Health score\nL1 regression", "Health stage\nauxiliary CE", "Fault part\nclassification CE"]
    values = [2.0, 0.05, 0.2]
    ax.bar(labels, values, color=[P_BLUE, P_GREY, P_RED], edgecolor="#4b5563", linewidth=0.4)
    ax.set_title("Task-level training weights")
    ax.set_ylabel("Loss weight")
    for i, value in enumerate(values):
        ax.text(i, value + 0.04, f"{value:g}", ha="center", fontsize=9)
    clean_axis(ax)
    return save_fig(fig, "training_strategy")


def make_final_results(final_df: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(11.0, 3.6))
    gs = fig.add_gridspec(1, 3)
    specs = [
        ("val_health_mae", "train_health_mae", "Health MAE", "MAE", (0.035, 0.125), P_BLUE),
        ("val_health_acc_at_0_1", "train_health_acc_at_0_1", "Health Acc@0.1", "Accuracy", (0.55, 0.90), P_GREEN),
        ("val_fault_macro_acc", "train_fault_macro_acc", "Fault Macro Accuracy", "Accuracy", (0.995, 1.0), P_RED),
    ]
    for i, (val_col, train_col, title, ylabel, ylim, color) in enumerate(specs):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(final_df["epoch"], final_df[train_col], color=P_GREY, lw=1.1, ls="--", label="Train")
        ax.plot(final_df["epoch"], final_df[val_col], color=color, lw=1.5, marker="o", ms=2.3, label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=7.5)
        clean_axis(ax)
    return save_fig(fig, "final_results")


def make_comparison_bars() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    labels = ["Stage-cls.", "Reg.-oriented"]
    x = np.arange(2)
    mae = [0.0960, 0.0419]
    acc = [0.5755, 0.8709]
    axes[0].bar(x, mae, color=[P_GREY, P_BLUE], edgecolor="#4b5563", linewidth=0.4)
    axes[0].set_title("Health MAE")
    axes[0].set_ylabel("Lower is better")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[1].bar(x, acc, color=[P_GREY, P_GREEN], edgecolor="#4b5563", linewidth=0.4)
    axes[1].set_title("Health Acc@0.1")
    axes[1].set_ylabel("Higher is better")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    for ax, values, fmt in [(axes[0], mae, "{:.4f}"), (axes[1], acc, "{:.1%}")]:
        for i, value in enumerate(values):
            ax.text(i, value + (0.004 if max(values) < 0.2 else 0.03), fmt.format(value), ha="center", fontsize=8.5)
        clean_axis(ax)
    return save_fig(fig, "comparison_bars")


@torch.no_grad()
def collect_predictions(max_scatter: int = 5000):
    set_seed(42)
    device = resolve_device("cpu")
    checkpoint = load_checkpoint(CHECKPOINT, map_location=device)
    args = checkpoint.get("args", {})
    _, val_loader, input_size = build_nasa_bearing_loaders(
        data_root=DATA_ROOT,
        batch_size=1024,
        num_workers=0,
        health_bins=args.get("health_bins", 11),
        health_label_mode=args.get("health_label_mode", "time"),
        fault_map=args.get("fault_map", DEFAULT_FAULT_MAP),
        include_bearing_id=args.get("include_bearing_id", True),
        val_fraction=args.get("val_fraction", 0.2),
        limit_good=args.get("limit_good"),
        limit_fault=args.get("limit_fault"),
        seed=args.get("seed", 42),
    )
    model = build_model(checkpoint, input_size, "stdp_params.npz", device)
    pred_h, target_h, pred_f, target_f = [], [], [], []
    for batch in val_loader:
        outputs = model(batch["features"].to(device))
        pred_h.append(outputs["health_score_pred"].cpu().numpy())
        target_h.append(batch["health_score"].numpy())
        pred_f.append(outputs["fault_logits"].argmax(dim=1).cpu().numpy())
        target_f.append(batch["fault_part_label"].numpy())
    pred_h = np.concatenate(pred_h)
    target_h = np.concatenate(target_h)
    pred_f = np.concatenate(pred_f)
    target_f = np.concatenate(target_f)
    rng = np.random.default_rng(42)
    idx = rng.choice(np.arange(len(pred_h)), size=min(max_scatter, len(pred_h)), replace=False)
    return pred_h[idx], target_h[idx], np.abs(pred_h[idx] - target_h[idx]), pred_f, target_f


def make_prediction_quality(pred_h, target_h, errors, pred_f, target_f) -> Path:
    fig = plt.figure(figsize=(11.0, 3.7))
    gs = fig.add_gridspec(1, 3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(target_h, pred_h, s=6, alpha=0.28, color=P_BLUE, edgecolor="none")
    ax0.plot([0, 1], [0, 1], color=P_RED, lw=1.2, ls="--", label="Ideal")
    ax0.set_title("Health-score regression")
    ax0.set_xlabel("Target health_score")
    ax0.set_ylabel("Predicted health_score")
    ax0.set_xlim(-0.02, 1.02)
    ax0.set_ylim(-0.02, 1.02)
    ax0.legend(fontsize=7.5)
    clean_axis(ax0)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(errors, bins=np.linspace(0, 0.35, 36), color=P_GREEN, edgecolor="white")
    ax1.axvline(0.1, color=P_RED, lw=1.2, ls="--", label="0.1 tolerance")
    ax1.set_title("Absolute health-score error")
    ax1.set_xlabel("|prediction - target|")
    ax1.set_ylabel("Validation samples")
    ax1.legend(fontsize=7.5)
    clean_axis(ax1)

    all_names = list(NASA_BEARING_FAULT_PARTS)
    present = sorted(set(target_f.astype(int).tolist()) | set(pred_f.astype(int).tolist()))
    names = [all_names[i] for i in present]
    remap = {old: new for new, old in enumerate(present)}
    cm = np.zeros((len(names), len(names)), dtype=float)
    for t, p in zip(target_f, pred_f):
        cm[remap[int(t)], remap[int(p)]] += 1
    support = cm.sum(axis=1, keepdims=True)
    cm = np.divide(cm, support, out=np.zeros_like(cm), where=support > 0)
    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.imshow(cm, cmap="Blues", vmin=0, vmax=1, alpha=0.78)
    ax2.set_title("Fault-part confusion matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Target")
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_yticks(np.arange(len(names)))
    ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax2.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            if support[i, 0] > 0 and cm[i, j] > 0.01:
                ax2.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    return save_fig(fig, "prediction_quality")


def make_application_flow() -> Path:
    fig, ax = plt.subplots(figsize=(11.8, 2.7))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [
        ("High-value rotor", "turbopump / bearing"),
        ("Vibration sensing", "B1-B4 channels"),
        ("Feature and coding", "features -> spikes"),
        ("STDP-SNN", "health + fault part"),
        ("Maintenance decision", "warning / locating / repair"),
    ]
    xs = np.linspace(0.10, 0.90, len(steps))
    box_w = 0.15
    for i, (title, sub) in enumerate(steps):
        ax.add_patch(plt.Rectangle((xs[i] - box_w / 2, 0.36), box_w, 0.30, facecolor=P_LIGHT, edgecolor="#4b5563", lw=0.9))
        ax.text(xs[i], 0.56, title, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#1f2937")
        ax.text(xs[i], 0.43, sub, ha="center", va="center", fontsize=7.5, color="#4b5563")
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - box_w / 2 - 0.012, 0.51), xytext=(xs[i] + box_w / 2 + 0.012, 0.51), arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.1))
    ax.set_title("Application scenario: bearing health monitoring")
    return save_fig(fig, "application_flow")


def make_figures() -> dict[str, Path]:
    configure_plot_style()
    PPT_ASSETS.mkdir(parents=True, exist_ok=True)
    good_rows = line_count(DATA_ROOT / "features_good_named.csv")
    fault_rows = line_count(DATA_ROOT / "features_fault_named.csv")
    good = sample_csv(DATA_ROOT / "features_good_named.csv", 18000, seed=42)
    fault = pd.read_csv(DATA_ROOT / "features_fault_named.csv")
    final_df = pd.read_csv(ASSETS / "nasa_bearing_final_metrics.csv")
    pred_h, target_h, errors, pred_f, target_f = collect_predictions()
    return {
        "application": make_application_flow(),
        "dataset": make_dataset_overview(good_rows, fault_rows),
        "processing": make_processing_and_health(good_rows, fault_rows),
        "raw": make_raw_waveform(),
        "features": make_feature_evidence(good, fault),
        "model": make_model_and_stdp(),
        "training": make_training_strategy(),
        "results": make_final_results(final_df),
        "comparison": make_comparison_bars(),
        "quality": make_prediction_quality(pred_h, target_h, errors, pred_f, target_f),
    }


def set_font(run, size: float, color: RGBColor = TEXT, bold: bool = False) -> None:
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold


def textbox(slide, text: str, x: float, y: float, w: float, h: float, size: float = 11, color: RGBColor = TEXT, bold: bool = False, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    set_font(r, size, color, bold)
    return shape


def title(slide, text: str, subtitle: str | None = None) -> None:
    textbox(slide, text, 0.42, 0.18, 12.1, 0.32, 18, INK, True)
    if subtitle:
        textbox(slide, subtitle, 0.44, 0.55, 12.0, 0.22, 8.5, MUTED)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.42), Inches(0.82), Inches(12.45), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = GRID
    line.line.fill.background()


def bullet_box(slide, items: list[str], x: float, y: float, w: float, h: float, size: float = 9.5) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.margin_left = Inches(0.04)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {item}"
        p.font.name = FONT
        p.font.size = Pt(size)
        p.font.color.rgb = TEXT
        p.space_after = Pt(2.5)


def add_picture_fit(slide, path: Path, x: float, y: float, w: float, h: float):
    with Image.open(path) as image:
        aspect = image.width / image.height
    box_aspect = w / h
    if aspect >= box_aspect:
        draw_w = w
        draw_h = w / aspect
        draw_x = x
        draw_y = y + (h - draw_h) / 2
    else:
        draw_h = h
        draw_w = h * aspect
        draw_x = x + (w - draw_w) / 2
        draw_y = y
    return slide.shapes.add_picture(str(path), Inches(draw_x), Inches(draw_y), width=Inches(draw_w), height=Inches(draw_h))


def caption(slide, text: str, x: float, y: float, w: float, h: float = 0.28) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = PALE
    shape.line.color.rgb = GRID
    textbox(slide, f"图注：{text}", x + 0.06, y + 0.055, w - 0.12, h - 0.08, 7.8, TEXT)


def table(slide, rows: list[list[str]], x: float, y: float, w: float, h: float, size: float = 8) -> None:
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = shape.table
    for ri, row in enumerate(rows):
        for ci, value in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(242, 244, 246) if ri == 0 else WHITE
            for p in cell.text_frame.paragraphs:
                p.font.name = FONT
                p.font.size = Pt(size)
                p.font.bold = ri == 0
                p.font.color.rgb = INK if ri == 0 else TEXT


def metric(slide, label: str, value: str, x: float, y: float, color: RGBColor) -> None:
    textbox(slide, value, x, y, 1.7, 0.26, 17, color, True)
    textbox(slide, label, x, y + 0.31, 2.0, 0.18, 7.5, TEXT)


def footer(slide, n: int) -> None:
    textbox(slide, f"NASA Bearing STDP-SNN | {n}/8", 11.35, 7.18, 1.45, 0.15, 7, MUTED, align=PP_ALIGN.RIGHT)


def add_image_with_caption(slide, path: Path, x: float, y: float, w: float, h: float, cap: str) -> None:
    add_picture_fit(slide, path, x, y, w, h)
    caption(slide, cap, x, y + h + 0.04, w)


def build_ppt(fig: dict[str, Path]) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # 1
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "研究目标与应用场景", "目标：用 STDP-SNN 同时完成轴承健康度预测与故障部位识别")
    table(slide, [
        ["任务", "定义", "主要指标"],
        ["Task 1", "连续健康度预测：health_score ∈ [0, 1]", "MAE, Acc@0.1"],
        ["Task 2", "故障部位分类：normal / inner_ring / rolling_element", "Macro Accuracy"],
        ["应用场景", "高价值旋转机械、航天涡轮泵轴承在线监测", "提前预警 + 部位定位"],
    ], 0.55, 1.08, 5.8, 1.75, 8.5)
    add_image_with_caption(slide, fig["application"], 6.65, 1.02, 5.95, 1.82, "应用流程图说明从振动传感到健康度预测、故障部位定位和维护决策的完整链路。")
    bullet_box(slide, [
        "健康度不再作为主要分类任务，而是作为连续回归值进行预测。",
        "故障部位仍是分类任务，因为输出具有明确离散物理类别。",
        "最终汇报只使用 Task 1/Task 2 对应的核心指标，不使用辅助 stage accuracy 作为主结果。",
    ], 0.65, 3.45, 11.4, 1.0, 9.2)
    metric(slide, "验证健康度 MAE", "0.0419", 1.0, 5.45, BLUE)
    metric(slide, "健康度 Acc@0.1", "87.1%", 3.6, 5.45, GREEN)
    metric(slide, "故障部位 Macro Acc", "99.75%", 6.2, 5.45, RED)
    footer(slide, 1)

    # 2
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "数据集与标签构造", "NASA Bearing 数据经过单轴承拆分后用于两个监督任务")
    table(slide, [
        ["项目", "数值/定义"],
        ["正常特征窗口", "222,811 行"],
        ["故障特征窗口", "1,703 行"],
        ["轴承通道", "B1, B2, B3, B4"],
        ["输入样本构造", "每行拆为 4 条单轴承样本；加入轴承编号信息"],
        ["默认故障映射", "B3=inner_ring, B4=rolling_element；B1/B2=normal"],
    ], 0.55, 1.05, 5.65, 2.45, 8.2)
    add_image_with_caption(slide, fig["dataset"], 6.55, 0.98, 5.9, 3.05, "柱状图证明原始 good/fault 数据和拆分后的类别分布高度不均衡，因此训练需要类别权重和平衡采样。")
    bullet_box(slide, [
        "当前验证中实际出现的故障类别为 normal、inner_ring、rolling_element。",
        "outer_ring、cage、unknown_fault 在当前 run 中没有验证样本，因此后续混淆矩阵不画空类别。",
    ], 0.65, 4.38, 11.2, 0.7, 9.0)
    footer(slide, 2)

    # 3
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "数据处理流程与健康度标签", "从四通道振动数据到连续 health_score 监督信号")
    add_image_with_caption(slide, fig["processing"], 0.55, 1.00, 11.95, 3.55, "流程图与健康度分布说明：模型输入来自单轴承样本，health_score 按故障时间顺序从 1.0 递减到 0.0。")
    table(slide, [
        ["标签", "生成方式", "用途"],
        ["health_score", "1 - fault_position / fault_length", "Task 1 主回归目标"],
        ["health_stage", "由 health_score 离散得到", "辅助约束，不作为主指标"],
        ["fault_part", "由 fault_map 指定部位", "Task 2 分类目标"],
    ], 0.8, 5.25, 11.3, 1.05, 8.2)
    footer(slide, 3)

    # 4
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "数据可视化与特征有效性", "原始信号、工程特征和故障时间线均包含可学习信息")
    add_image_with_caption(slide, fig["raw"], 0.55, 1.00, 5.35, 2.15, "原始振动片段显示四个轴承通道存在连续振动响应，是后续特征提取的信号来源。")
    add_image_with_caption(slide, fig["features"], 6.15, 1.00, 6.25, 4.10, "箱线图、时间线曲线和相关性热图共同说明工程特征能区分正常/故障，并随退化进程变化。")
    bullet_box(slide, [
        "特征分布差异支持故障部位分类。",
        "故障时间线上的特征变化支持连续健康度回归。",
        "相关性图说明特征存在冗余和互补信息，需要模型学习共享表征。",
    ], 0.7, 4.15, 5.1, 1.35, 9.0)
    footer(slide, 4)

    # 5
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "模型结构与数据流", "不展示模型超参数，只说明任务相关的数据流和输出头")
    add_image_with_caption(slide, fig["model"], 0.55, 1.00, 11.95, 3.55, "模型数据流图说明同一 STDP-SNN 主干同时输出连续健康度、辅助健康阶段和故障部位分类。")
    bullet_box(slide, [
        "STDP 层体现脉冲时序可塑性；图中使用归一化 learning window 展示其学习规则形状。",
        "health_score 是主要输出；health_stage 仅用于辅助约束退化顺序。",
        "fault_part 是分类输出，对应具体故障部位。",
    ], 0.75, 5.18, 11.2, 1.1, 9.0)
    footer(slide, 5)

    # 6
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "训练策略", "最终训练目标更重视健康度回归，同时保留故障分类稳定性")
    table(slide, [
        ["训练项", "设置", "理由"],
        ["健康度", "L1 回归，权重 2.0", "直接降低 MAE，是 Task 1 主目标"],
        ["健康阶段", "辅助 CE，权重 0.05", "保留退化顺序约束，但不主导训练"],
        ["故障部位", "CE 分类，权重 0.2", "分类已较稳定，降低权重避免压制回归"],
        ["采样", "故障类别平衡", "缓解 normal 多数类影响"],
    ], 0.55, 1.05, 6.1, 2.25, 8.2)
    add_image_with_caption(slide, fig["training"], 7.0, 1.00, 4.9, 2.35, "训练权重图证明最终版本把优化重心放在连续健康度回归，而不是已经接近饱和的故障分类。")
    add_image_with_caption(slide, fig["comparison"], 0.8, 4.25, 5.9, 2.0, "对比柱状图证明回归导向训练将健康度 MAE 从约 0.096 降到 0.0419，同时 Acc@0.1 明显提升。")
    footer(slide, 6)

    # 7
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "核心结果", "健康度误差明显下降，故障部位分类保持稳定")
    add_image_with_caption(slide, fig["results"], 0.55, 1.00, 11.95, 3.45, "最终训练曲线证明验证 MAE 下降到 0.0419，Acc@0.1 达到 0.8709，同时故障部位 Macro Accuracy 约 0.9975。")
    table(slide, [
        ["指标", "结果", "解释"],
        ["Validation Health MAE", "0.0419", "0-1 健康度范围内平均误差约 4.2%"],
        ["Validation Health Acc@0.1", "0.8709", "87.1% 样本误差不超过 0.1"],
        ["Validation Fault Macro Acc", "0.9975", "按类别平均后的故障部位准确率"],
    ], 0.75, 5.15, 11.2, 1.05, 8.3)
    footer(slide, 7)

    # 8
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "预测质量与结论", "连续健康度预测和故障部位识别均达到可汇报水平")
    add_image_with_caption(slide, fig["quality"], 0.55, 1.00, 11.95, 3.6, "散点图、误差直方图和混淆矩阵共同证明：健康度预测误差集中在 0.1 内，实际故障类别识别稳定。")
    table(slide, [
        ["结论", "说明"],
        ["Task 1", "连续健康度预测平均误差约 4.2%，不再依赖阶段分类准确率。"],
        ["Task 2", "故障部位分类 Macro Accuracy 约 99.75%，normal/inner_ring/rolling_element 区分稳定。"],
        ["限制", "当前验证为随机切分，可能比严格时间外推更乐观，汇报时需要注明。"],
    ], 0.75, 5.05, 11.2, 1.28, 8.2)
    footer(slide, 8)

    prs.save(OUTPUT)


def main() -> None:
    figures = make_figures()
    build_ppt(figures)
    prs = Presentation(OUTPUT)
    print(f"saved={OUTPUT}")
    print(f"slides={len(prs.slides)}")
    print(f"assets={PPT_ASSETS}")


if __name__ == "__main__":
    main()
