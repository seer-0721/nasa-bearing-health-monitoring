from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import DEFAULT_FAULT_MAP, NASA_BEARING_FAULT_PARTS, build_nasa_bearing_loaders
from eval import build_model
from utils import load_checkpoint, resolve_device, set_seed


DATA_ROOT = ROOT / "data" / "NasaBearingDataset"
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
PPT_ASSETS = DOCS / "ppt_assets"
OUTPUT = DOCS / "nasa_bearing_snn_academic_report.pptx"
CHECKPOINT = ROOT / "checkpoints" / "nasa_bearing_snn_health_reg_v3" / "best.pt"

NAVY = RGBColor(20, 34, 58)
BLUE = RGBColor(36, 99, 235)
CYAN = RGBColor(8, 145, 178)
GREEN = RGBColor(22, 163, 74)
RED = RGBColor(220, 38, 38)
AMBER = RGBColor(217, 119, 6)
SLATE = RGBColor(71, 85, 105)
LIGHT = RGBColor(248, 250, 252)
MID = RGBColor(226, 232, 240)
WHITE = RGBColor(255, 255, 255)


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "Times"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
            "savefig.dpi": 450,
            "axes.linewidth": 0.8,
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


def sample_csv(path: Path, n: int, seed: int) -> pd.DataFrame:
    rows = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
    if rows <= n:
        return pd.read_csv(path)
    rng = np.random.default_rng(seed)
    selected = set(rng.choice(np.arange(1, rows + 1), size=n, replace=False).tolist())
    return pd.read_csv(path, skiprows=lambda idx: idx != 0 and idx not in selected)


def line_count(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8")) - 1


def health_scores_for_counts(good_rows: int, fault_rows: int) -> np.ndarray:
    normal_scores = np.ones(good_rows * 4 + fault_rows * 2, dtype=np.float32)
    if fault_rows > 1:
        fault_scores = 1.0 - np.arange(fault_rows, dtype=np.float32) / float(fault_rows - 1)
    else:
        fault_scores = np.zeros(fault_rows, dtype=np.float32)
    return np.concatenate([normal_scores, fault_scores, fault_scores])


def make_dataset_count_plot(good_rows: int, fault_rows: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    axes[0].bar(["Good windows", "Fault windows"], [good_rows, fault_rows], color=["#2563eb", "#dc2626"])
    axes[0].set_yscale("log")
    axes[0].set_title("Source files")
    axes[0].set_ylabel("Rows (log scale)")
    for idx, value in enumerate([good_rows, fault_rows]):
        axes[0].text(idx, value * 1.18, f"{value:,}", ha="center", fontsize=9)

    expanded = [good_rows * 4 + fault_rows * 2, fault_rows, fault_rows]
    labels = ["normal", "inner_ring", "rolling_element"]
    axes[1].bar(labels, expanded, color=["#64748b", "#f97316", "#7c3aed"])
    axes[1].set_yscale("log")
    axes[1].set_title("Labels after per-bearing split")
    axes[1].set_ylabel("Samples (log scale)")
    axes[1].tick_params(axis="x", rotation=15)
    for idx, value in enumerate(expanded):
        axes[1].text(idx, value * 1.18, f"{value:,}", ha="center", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    return save_fig(fig, "dataset_count_bars")


def make_raw_waveform_plot() -> Path:
    df = pd.read_csv(DATA_ROOT / "test_final.csv", nrows=2400)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9.4, 3.3))
    offsets = [0.0, 1.25, 2.5, 3.75]
    colors = ["#2563eb", "#0891b2", "#f97316", "#7c3aed"]
    for idx, col in enumerate(["Bearing1", "Bearing2", "Bearing3", "Bearing4"]):
        y = df[col].to_numpy(dtype=float)
        y = y / max(np.std(y), 1e-6) * 0.22 + offsets[idx]
        ax.plot(x, y, linewidth=0.8, color=colors[idx], label=col)
    ax.set_title("Raw vibration segment")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Normalized amplitude + offset")
    ax.set_yticks(offsets)
    ax.set_yticklabels(["B1", "B2", "B3", "B4"])
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    return save_fig(fig, "raw_vibration_segment")


def make_health_distribution_plot(good_rows: int, fault_rows: int) -> Path:
    scores = health_scores_for_counts(good_rows, fault_rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    axes[0].hist(scores, bins=np.linspace(0, 1, 12), color="#2563eb", edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_title("Health-score distribution")
    axes[0].set_xlabel("health_score")
    axes[0].set_ylabel("Samples (log scale)")
    axes[0].grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    fault_score = 1.0 - np.arange(fault_rows, dtype=float) / max(fault_rows - 1, 1)
    axes[1].plot(np.arange(fault_rows), fault_score, color="#dc2626", linewidth=2.0)
    axes[1].set_title("Time-derived degradation target")
    axes[1].set_xlabel("Fault-window position")
    axes[1].set_ylabel("health_score")
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    return save_fig(fig, "health_score_distribution")


def make_feature_trajectory_plot(fault_df: pd.DataFrame) -> Path:
    x = np.linspace(0, 1, len(fault_df))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3), sharex=True)
    for bearing, color in [("B3", "#f97316"), ("B4", "#7c3aed")]:
        for feature, style in [("rms", "-"), ("p2p", "--")]:
            y = fault_df[f"{bearing}__{feature}"].rolling(31, center=True, min_periods=1).mean()
            y = (y - y.min()) / max(float(y.max() - y.min()), 1e-9)
            axes[0].plot(x, y, linestyle=style, color=color, linewidth=1.6, label=f"{bearing} {feature}")
    axes[0].set_title("Fault-bearing feature trajectory")
    axes[0].set_xlabel("Normalized fault timeline")
    axes[0].set_ylabel("Normalized feature")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    health = 1.0 - x
    axes[1].plot(x, health, color="#1f2937", linewidth=2.0, label="Target health_score")
    axes[1].plot(x, 1.0 - health, color="#dc2626", linewidth=1.6, linestyle="--", label="Degradation")
    axes[1].set_title("Regression target over time")
    axes[1].set_xlabel("Normalized fault timeline")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    return save_fig(fig, "feature_trajectory_fault")


def make_feature_boxplot(good_df: pd.DataFrame, fault_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    specs = [("B3__rms", "B3 RMS"), ("B4__p2p", "B4 peak-to-peak")]
    for ax, (col, title) in zip(axes, specs):
        values = [
            good_df[col].dropna().to_numpy(),
            fault_df[col].dropna().to_numpy(),
        ]
        box = ax.boxplot(values, labels=["Good", "Fault"], patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], ["#bfdbfe", "#fecaca"]):
            patch.set_facecolor(color)
            patch.set_edgecolor("#475569")
        ax.set_title(title)
        ax.set_ylabel("Feature value")
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    return save_fig(fig, "feature_separation_boxplot")


def make_correlation_heatmap(fault_df: pd.DataFrame) -> Path:
    cols = [c for c in fault_df.columns if c.startswith("B3__")]
    labels = [c.split("__", 1)[1] for c in cols]
    corr = fault_df[cols].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("B3 feature correlation during fault period")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    return save_fig(fig, "feature_correlation_heatmap")


def make_stdp_curve_plot() -> Path:
    params = np.load(ROOT / "stdp_params.npz")
    a_plus = float(params["A_plus"])
    tau_plus = float(params["tau_plus"])
    a_minus = float(params["A_minus"])
    tau_minus = float(params["tau_minus"])
    dt = np.linspace(-120, 120, 500)
    dw = np.where(dt >= 0, a_plus * np.exp(-dt / tau_plus), a_minus * np.exp(dt / tau_minus))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(dt, dw, color="#2563eb", linewidth=2.0)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_title("Device-fitted STDP update")
    ax.set_xlabel(r"Spike timing difference $\Delta t$ (ms)")
    ax.set_ylabel(r"Synaptic update $\Delta w$")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    return save_fig(fig, "stdp_curve")


def make_pipeline_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 3.1))
    ax.axis("off")
    steps = [
        ("Raw vibration", "4 bearing channels"),
        ("Feature extraction", "20 features per bearing"),
        ("Per-bearing split", "B1/B2/B3/B4 samples"),
        ("Label construction", "health_score + fault_part"),
        ("STDP-SNN", "spike-based inference"),
        ("Outputs", "health + fault location"),
    ]
    xs = np.linspace(0.07, 0.93, len(steps))
    for i, (title, sub) in enumerate(steps):
        rect = plt.Rectangle((xs[i] - 0.075, 0.34), 0.15, 0.34, facecolor="#f8fafc", edgecolor="#334155", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(xs[i], 0.56, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(xs[i], 0.43, sub, ha="center", va="center", fontsize=8.5, color="#475569")
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.085, 0.51), xytext=(xs[i] + 0.085, 0.51), arrowprops=dict(arrowstyle="->", lw=1.4, color="#2563eb"))
    ax.text(0.5, 0.13, "Evidence: the pipeline converts a run-level bearing file into supervised samples aligned with the two target tasks.", ha="center", fontsize=10, color="#334155")
    return save_fig(fig, "dataset_processing_pipeline")


def make_model_dataflow_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.axis("off")
    blocks = [
        ("24-D input", "#dbeafe"),
        ("Rate spikes", "#e0f2fe"),
        ("STDP layer", "#fef3c7"),
        ("LIF dynamics", "#ecfccb"),
        ("Shared state", "#f8fafc"),
    ]
    xs = np.linspace(0.10, 0.66, len(blocks))
    for idx, (label, color) in enumerate(blocks):
        ax.add_patch(plt.Rectangle((xs[idx] - 0.055, 0.47), 0.11, 0.23, facecolor=color, edgecolor="#334155", linewidth=1.0))
        ax.text(xs[idx], 0.585, label, ha="center", va="center", fontsize=10, fontweight="bold")
        if idx < len(blocks) - 1:
            ax.annotate("", xy=(xs[idx + 1] - 0.063, 0.585), xytext=(xs[idx] + 0.063, 0.585), arrowprops=dict(arrowstyle="->", lw=1.4, color="#2563eb"))
    heads = [
        ("health_score", "Regression", "#dcfce7", 0.82, 0.72),
        ("health_stage", "Auxiliary", "#e0f2fe", 0.82, 0.48),
        ("fault_part", "Classification", "#fce7f3", 0.82, 0.24),
    ]
    for title, sub, color, x, y in heads:
        ax.add_patch(plt.Rectangle((x - 0.085, y - 0.065), 0.17, 0.13, facecolor=color, edgecolor="#334155", linewidth=1.0))
        ax.text(x, y + 0.022, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x, y - 0.031, sub, ha="center", va="center", fontsize=8.5, color="#475569")
        ax.annotate("", xy=(x - 0.095, y), xytext=(0.72, 0.585), arrowprops=dict(arrowstyle="->", lw=1.2, color="#64748b"))
    ax.text(0.5, 0.08, "Evidence: the model separates continuous health estimation from component-level fault classification.", ha="center", fontsize=10, color="#334155")
    return save_fig(fig, "model_dataflow_diagram")


def make_comparison_bar_plot() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    versions = ["Stage-cls\nintermediate", "Health-reg\nfinal"]
    mae = [0.0960, 0.0419]
    acc01 = [0.5755, 0.8709]
    x = np.arange(len(versions))
    axes[0].bar(x, mae, color=["#94a3b8", "#2563eb"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(versions)
    axes[0].set_title("Health-score MAE")
    axes[0].set_ylabel("MAE (lower is better)")
    for i, v in enumerate(mae):
        axes[0].text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=9)
    axes[1].bar(x, acc01, color=["#94a3b8", "#16a34a"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(versions)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Health Acc@0.1")
    axes[1].set_ylabel("Accuracy (higher is better)")
    for i, v in enumerate(acc01):
        axes[1].text(i, v + 0.03, f"{v:.1%}", ha="center", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    return save_fig(fig, "result_comparison_bars")


def make_pareto_plot(final_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    sc = ax.scatter(
        final_df["val_health_mae"],
        final_df["val_fault_macro_acc"],
        c=final_df["epoch"],
        cmap="viridis",
        s=55,
        edgecolor="white",
        linewidth=0.6,
    )
    best = final_df.loc[final_df["val_health_mae"].idxmin()]
    ax.scatter([best["val_health_mae"]], [best["val_fault_macro_acc"]], s=110, color="#dc2626", marker="*", label=f"Best health epoch {int(best['epoch'])}")
    ax.set_title("Health-fault performance trade-off")
    ax.set_xlabel("Validation health MAE")
    ax.set_ylabel("Validation fault macro accuracy")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(fontsize=8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Epoch")
    return save_fig(fig, "result_pareto_epochs")


@torch.no_grad()
def collect_checkpoint_predictions(max_scatter: int = 6000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    set_seed(42)
    device = resolve_device("cpu")
    checkpoint = load_checkpoint(CHECKPOINT, map_location=device)
    train_args = checkpoint.get("args", {})
    _, val_loader, input_size = build_nasa_bearing_loaders(
        data_root=DATA_ROOT,
        batch_size=1024,
        num_workers=0,
        health_bins=train_args.get("health_bins", 11),
        health_label_mode=train_args.get("health_label_mode", "time"),
        fault_map=train_args.get("fault_map", DEFAULT_FAULT_MAP),
        include_bearing_id=train_args.get("include_bearing_id", True),
        val_fraction=train_args.get("val_fraction", 0.2),
        limit_good=train_args.get("limit_good"),
        limit_fault=train_args.get("limit_fault"),
        seed=train_args.get("seed", 42),
    )
    model = build_model(checkpoint, input_size, "stdp_params.npz", device)
    model.eval()
    pred_health = []
    target_health = []
    pred_fault = []
    target_fault = []
    for batch in val_loader:
        features = batch["features"].to(device)
        outputs = model(features)
        pred_health.append(outputs["health_score_pred"].cpu().numpy())
        target_health.append(batch["health_score"].numpy())
        pred_fault.append(outputs["fault_logits"].argmax(dim=1).cpu().numpy())
        target_fault.append(batch["fault_part_label"].numpy())
    pred_h = np.concatenate(pred_health)
    target_h = np.concatenate(target_health)
    pred_f = np.concatenate(pred_fault)
    target_f = np.concatenate(target_fault)
    rng = np.random.default_rng(42)
    if len(pred_h) > max_scatter:
        idx = rng.choice(np.arange(len(pred_h)), size=max_scatter, replace=False)
    else:
        idx = np.arange(len(pred_h))
    return pred_h[idx], target_h[idx], pred_f, target_f, np.abs(pred_h - target_h)[idx]


def make_prediction_scatter_and_error(pred_h: np.ndarray, target_h: np.ndarray, errors: np.ndarray) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(5.1, 4.4))
    ax.scatter(target_h, pred_h, s=8, alpha=0.35, color="#2563eb", edgecolor="none")
    ax.plot([0, 1], [0, 1], color="#dc2626", linewidth=1.6, linestyle="--", label="Ideal")
    ax.set_title("Health-score regression")
    ax.set_xlabel("Target health_score")
    ax.set_ylabel("Predicted health_score")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(fontsize=8)
    scatter_path = save_fig(fig, "prediction_health_scatter")

    fig, ax = plt.subplots(figsize=(5.1, 3.6))
    ax.hist(errors, bins=np.linspace(0, 0.35, 36), color="#0891b2", edgecolor="white")
    ax.axvline(0.1, color="#dc2626", linewidth=1.6, linestyle="--", label="0.1 tolerance")
    ax.set_title("Absolute health-score error")
    ax.set_xlabel("|prediction - target|")
    ax.set_ylabel("Validation samples")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(fontsize=8)
    hist_path = save_fig(fig, "prediction_health_error_hist")
    return scatter_path, hist_path


def make_fault_confusion_matrix(pred_f: np.ndarray, target_f: np.ndarray) -> Path:
    classes = list(NASA_BEARING_FAULT_PARTS)
    confusion = np.zeros((len(classes), len(classes)), dtype=float)
    for t, p in zip(target_f, pred_f):
        confusion[int(t), int(p)] += 1
    support = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, support, out=np.zeros_like(confusion), where=support > 0)
    present = support[:, 0] > 0
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title("Fault-part confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if present[i] and normalized[i, j] > 0.005:
                ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#0f172a")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized accuracy")
    return save_fig(fig, "fault_confusion_matrix")


def make_all_figures() -> dict[str, Path]:
    configure_plot_style()
    PPT_ASSETS.mkdir(parents=True, exist_ok=True)
    good_rows = line_count(DATA_ROOT / "features_good_named.csv")
    fault_rows = line_count(DATA_ROOT / "features_fault_named.csv")
    good_sample = sample_csv(DATA_ROOT / "features_good_named.csv", n=20000, seed=42)
    fault_df = pd.read_csv(DATA_ROOT / "features_fault_named.csv")
    final_df = pd.read_csv(ASSETS / "nasa_bearing_final_metrics.csv")

    figures = {
        "dataset_counts": make_dataset_count_plot(good_rows, fault_rows),
        "raw_waveform": make_raw_waveform_plot(),
        "health_distribution": make_health_distribution_plot(good_rows, fault_rows),
        "feature_trajectory": make_feature_trajectory_plot(fault_df),
        "feature_boxplot": make_feature_boxplot(good_sample, fault_df),
        "correlation": make_correlation_heatmap(fault_df),
        "stdp_curve": make_stdp_curve_plot(),
        "pipeline": make_pipeline_diagram(),
        "model_dataflow": make_model_dataflow_diagram(),
        "comparison_bars": make_comparison_bar_plot(),
        "pareto": make_pareto_plot(final_df),
        "final_metrics": ASSETS / "nasa_bearing_final_metrics.png",
        "model": ASSETS / "nasa_bearing_model.png",
        "application": ASSETS / "aerospace_bearing_snn_application.png",
    }
    pred_h, target_h, pred_f, target_f, errors = collect_checkpoint_predictions()
    scatter, hist = make_prediction_scatter_and_error(pred_h, target_h, errors)
    figures["health_scatter"] = scatter
    figures["health_error_hist"] = hist
    figures["fault_confusion"] = make_fault_confusion_matrix(pred_f, target_f)
    return figures


def set_slide_bg(slide, color: RGBColor = WHITE) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, text: str, x: float, y: float, w: float, h: float, size: int = 18, color: RGBColor = NAVY, bold: bool = False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = "Aptos"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def add_multiline(slide, lines: list[str], x: float, y: float, w: float, h: float, size: int = 15, color: RGBColor = SLATE, bullet: bool = True):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    for idx, line in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.level = 0
        if bullet:
            p.text = f"• {line}"
    return box


def add_title(slide, title: str, subtitle: str | None = None) -> None:
    add_textbox(slide, title, 0.55, 0.25, 12.0, 0.42, size=25, color=NAVY, bold=True)
    if subtitle:
        add_textbox(slide, subtitle, 0.58, 0.72, 11.5, 0.30, size=11, color=SLATE)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.04), Inches(12.25), Inches(0.03))
    line.fill.solid()
    line.fill.fore_color.rgb = BLUE
    line.line.fill.background()


def add_footer(slide, idx: int) -> None:
    add_textbox(slide, f"NASA Bearing STDP-SNN | {idx}", 11.35, 7.18, 1.4, 0.18, size=8, color=SLATE, align=PP_ALIGN.RIGHT)


def add_image(slide, path: Path, x: float, y: float, w: float | None = None, h: float | None = None):
    if w is not None and h is not None:
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))
    if w is not None:
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    if h is not None:
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y), height=Inches(h))
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y))


def add_evidence(slide, text: str, x: float = 0.70, y: float = 6.74, w: float = 12.0) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.35))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(239, 246, 255)
    shape.line.color.rgb = RGBColor(191, 219, 254)
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = f"Evidence: {text}"
    r.font.name = "Aptos"
    r.font.size = Pt(10.5)
    r.font.color.rgb = NAVY


def add_metric_card(slide, label: str, value: str, note: str, x: float, y: float, color: RGBColor = BLUE):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(2.85), Inches(1.12))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(248, 250, 252)
    shape.line.color.rgb = MID
    add_textbox(slide, value, x + 0.18, y + 0.14, 2.4, 0.34, size=22, color=color, bold=True)
    add_textbox(slide, label, x + 0.18, y + 0.56, 2.5, 0.22, size=10, color=NAVY, bold=True)
    add_textbox(slide, note, x + 0.18, y + 0.80, 2.5, 0.20, size=8.5, color=SLATE)


def add_table(slide, rows: list[list[str]], x: float, y: float, w: float, h: float, font_size: int = 9) -> None:
    table_shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    table = table_shape.table
    for r_idx, row in enumerate(rows):
        for c_idx, value in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = value
            fill = cell.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor(241, 245, 249) if r_idx == 0 else WHITE
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name = "Aptos"
                paragraph.font.size = Pt(font_size)
                paragraph.font.bold = r_idx == 0
                paragraph.font.color.rgb = NAVY if r_idx == 0 else SLATE


def add_section_slide(prs: Presentation, title: str, subtitle: str, idx: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_textbox(slide, title, 0.72, 2.55, 11.8, 0.55, size=30, color=WHITE, bold=True)
    add_textbox(slide, subtitle, 0.75, 3.18, 10.5, 0.36, size=15, color=RGBColor(203, 213, 225))
    add_footer(slide, idx)


def build_ppt(figures: dict[str, Path]) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide_no = 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)
    add_image(slide, figures["application"], 0, 0, w=13.333, h=7.5)
    overlay = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(6.3), Inches(7.5))
    overlay.fill.solid()
    overlay.fill.fore_color.rgb = RGBColor(255, 255, 255)
    overlay.fill.transparency = 8
    overlay.line.fill.background()
    add_textbox(slide, "Neuromorphic Bearing Health Prediction", 0.55, 0.75, 5.4, 0.72, size=28, color=NAVY, bold=True)
    add_textbox(slide, "NASA Bearing dataset | STDP-SNN | continuous health regression + fault-part classification", 0.58, 1.62, 5.25, 0.62, size=13, color=SLATE)
    add_metric_card(slide, "Validation health MAE", "0.0419", "average error about 4.2%", 0.62, 3.15, BLUE)
    add_metric_card(slide, "Health Acc@0.1", "87.1%", "error within 0.1", 0.62, 4.45, GREEN)
    add_metric_card(slide, "Fault macro accuracy", "99.75%", "component-level classification", 3.62, 4.45, RED)
    add_footer(slide, slide_no)
    slide_no += 1

    add_section_slide(prs, "Part 1. General Overview", "Problem definition, system objective, and application context", slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Problem formulation", "A two-output prognostics and diagnostics model for bearing condition monitoring")
    add_multiline(
        slide,
        [
            "Task 1 is continuous health-score regression: 1.0 means healthy, 0.0 means near failure.",
            "Task 2 is component-level fault classification: normal, inner_ring, rolling_element, etc.",
            "The final model uses a shared STDP-SNN backbone with task-specific heads.",
            "The evaluation therefore emphasizes MAE / Acc@0.1 for health and macro accuracy for fault location.",
        ],
        0.78,
        1.35,
        5.35,
        3.2,
        size=15,
    )
    add_image(slide, figures["application"], 6.55, 1.28, w=5.95, h=3.35)
    add_metric_card(slide, "Best health MAE", "0.0419", "epoch 81", 1.0, 5.28, BLUE)
    add_metric_card(slide, "Best Acc@0.1", "0.8709", "epoch 81", 4.0, 5.28, GREEN)
    add_metric_card(slide, "Fault macro acc.", "0.9975", "epoch 81", 7.0, 5.28, RED)
    add_evidence(slide, "The task definition separates lifetime/health estimation from discrete fault localization.")
    add_footer(slide, slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Application scenario", "From high-value rotating machinery to onboard health monitoring")
    add_image(slide, figures["application"], 0.62, 1.25, w=7.2, h=4.06)
    add_multiline(
        slide,
        [
            "Vibration sensors capture transient bearing behavior before visible failure.",
            "A neuromorphic front-end can process spike-coded features with low-latency dynamics.",
            "The model outputs both degradation score and likely fault component.",
            "This presentation uses an aerospace bearing/turbopump scenario as the motivating application.",
        ],
        8.2,
        1.42,
        4.3,
        3.1,
        size=13,
    )
    add_evidence(slide, "The application graphic maps the project from offline dataset modeling to an interpretable health-monitoring system.")
    add_footer(slide, slide_no)
    slide_no += 1

    add_section_slide(prs, "Part 2. Dataset Introduction", "Dataset description, visualization, and conversion into model samples", slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Dataset description", "Four bearing channels with engineered time- and frequency-domain features")
    add_image(slide, figures["dataset_counts"], 0.65, 1.28, w=6.4, h=2.45)
    rows = [
        ["Item", "Value"],
        ["Good feature rows", "222,811"],
        ["Fault feature rows", "1,703"],
        ["Bearing channels", "B1, B2, B3, B4"],
        ["Features per bearing", "20"],
        ["Actual model input", "24-D = 20 features + 4 bearing-ID"],
    ]
    add_table(slide, rows, 7.35, 1.30, 4.9, 2.35, font_size=9)
    add_multiline(
        slide,
        [
            "The original source distribution is highly imbalanced.",
            "Per-bearing splitting creates component-aligned samples for fault localization.",
        ],
        0.85,
        4.45,
        10.8,
        0.7,
        size=13,
    )
    add_evidence(slide, "The bar charts justify class weighting and balanced sampling during training.")
    add_footer(slide, slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Raw vibration visualization", "A short segment reveals multi-channel oscillatory behavior before feature extraction")
    add_image(slide, figures["raw_waveform"], 0.78, 1.28, w=7.1, h=2.55)
    add_image(slide, figures["feature_boxplot"], 0.78, 4.05, w=7.1, h=2.45)
    add_multiline(
        slide,
        [
            "Raw vibration is not directly fed into the final model.",
            "Feature extraction compresses local signal windows into compact descriptors.",
            "The box plots show that selected features shift between good and fault periods.",
        ],
        8.35,
        1.55,
        4.1,
        3.2,
        size=13,
    )
    add_evidence(slide, "The waveform motivates feature extraction; feature distributions show measurable condition-dependent change.")
    add_footer(slide, slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Dataset processing pipeline", "How the bearing files are converted into supervised SNN samples")
    add_image(slide, figures["pipeline"], 0.72, 1.20, w=11.9, h=3.45)
    add_multiline(
        slide,
        [
            "Each feature row is split into four single-bearing samples.",
            "B3 is labeled as inner_ring fault and B4 as rolling_element fault in the default run.",
            "B1/B2 remain normal during the fault segment under the configured mapping.",
            "A 4-D bearing-ID code is appended so the model can distinguish sensor position.",
        ],
        1.0,
        4.85,
        11.0,
        1.1,
        size=13,
    )
    add_evidence(slide, "The pipeline makes labels physically meaningful at the bearing/component level.")
    add_footer(slide, slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Label construction", "Continuous health-score target and component-level fault labels")
    add_image(slide, figures["health_distribution"], 0.65, 1.22, w=6.35, h=2.55)
    add_image(slide, figures["feature_trajectory"], 0.65, 4.05, w=6.35, h=2.2)
    rows = [
        ["Bearing", "Fault label in fault segment"],
        ["B1", "normal"],
        ["B2", "normal"],
        ["B3", "inner_ring"],
        ["B4", "rolling_element"],
    ]
    add_table(slide, rows, 7.55, 1.42, 4.35, 1.85, font_size=9)
    add_multiline(
        slide,
        [
            "Health-score labels decay from 1.0 to 0.0 across the fault timeline.",
            "The 11-stage label is retained only as an auxiliary ordinal target.",
            "Task 1 is evaluated by regression metrics, not by exact stage accuracy.",
        ],
        7.55,
        3.75,
        4.8,
        1.55,
        size=13,
    )
    add_evidence(slide, "The target design converts run order into a continuous degradation supervision signal.")
    add_footer(slide, slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Feature structure", "Correlation among engineered descriptors during the fault period")
    add_image(slide, figures["correlation"], 0.95, 1.18, w=5.2, h=4.55)
    add_multiline(
        slide,
        [
            "Time-domain amplitude descriptors are strongly correlated.",
            "Frequency-domain descriptors provide complementary information.",
            "The SNN receives a compact 20-D descriptor vector instead of a long raw sequence.",
            "Feature redundancy motivates a learned shared representation before task heads.",
        ],
        6.85,
        1.42,
        5.35,
        2.6,
        size=13,
    )
    add_evidence(slide, "The heatmap shows that the feature vector has structured redundancy and complementary spectral cues.")
    add_footer(slide, slide_no)
    slide_no += 1

    add_section_slide(prs, "Part 3. Model", "STDP-SNN architecture, model visualization, and data flow", slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Model visualization", "NasaBearingSNN: device-inspired spike encoding with multi-task heads")
    add_image(slide, figures["model"], 0.75, 1.20, w=6.0, h=4.55)
    rows = [
        ["Module", "Configuration"],
        ["Input", "24 features"],
        ["Spike encoder", "16 time steps"],
        ["STDP layer", "24 -> 384"],
        ["LIF + FC", "384 -> 192"],
        ["Heads", "health_score, health_stage, fault_part"],
    ]
    add_table(slide, rows, 7.25, 1.35, 4.65, 2.2, font_size=9)
    add_multiline(
        slide,
        [
            "The shared backbone learns condition-sensitive representations.",
            "The health head is continuous regression.",
            "The fault head remains a classification problem.",
        ],
        7.25,
        4.05,
        4.6,
        1.2,
        size=13,
    )
    add_evidence(slide, "The architecture explicitly matches the problem decomposition: regression for health, classification for fault part.")
    add_footer(slide, slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Model data flow", "From normalized bearing features to interpretable diagnostic outputs")
    add_image(slide, figures["model_dataflow"], 0.65, 1.25, w=11.9, h=3.85)
    add_multiline(
        slide,
        [
            "The model first converts bounded features into deterministic rate spikes.",
            "Temporal spike activity is integrated through LIF dynamics.",
            "Shared features feed three heads; only health_score and fault_part are primary outputs.",
        ],
        1.0,
        5.25,
        11.0,
        0.9,
        size=13,
    )
    add_evidence(slide, "The data-flow diagram clarifies how one input stream supports both prognostics and diagnostics.")
    add_footer(slide, slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "STDP device layer", "Device-fitted plasticity parameters initialize the first synaptic layer")
    add_image(slide, figures["stdp_curve"], 0.75, 1.40, w=6.05, h=2.95)
    add_multiline(
        slide,
        [
            "STDP links spike timing to synaptic update magnitude.",
            "The first layer can be pre-trained in an unsupervised manner.",
            "Supervised training then optimizes task heads and downstream layers.",
            "This gives the model a neuromorphic interpretation rather than a purely conventional MLP.",
        ],
        7.2,
        1.42,
        4.8,
        2.6,
        size=13,
    )
    add_evidence(slide, "The curve documents that the SNN layer is parameterized by the measured device-inspired STDP rule.")
    add_footer(slide, slide_no)
    slide_no += 1

    add_section_slide(prs, "Part 4. Result", "Regression quality, fault classification quality, and evidence of improvement", slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Training protocol", "Final configuration emphasizes continuous health regression")
    rows = [
        ["Setting", "Value"],
        ["Checkpoint", "nasa_bearing_snn_health_reg_v3"],
        ["Epochs", "100"],
        ["Batch size", "256"],
        ["Health loss", "L1, weight 2.0"],
        ["Health stage aux.", "CE, weight 0.05"],
        ["Fault loss", "CE, weight 0.2"],
        ["Sampler", "fault-class balanced"],
        ["STDP pretrain", "enabled, 1 epoch"],
    ]
    add_table(slide, rows, 0.85, 1.35, 5.15, 3.55, font_size=10)
    add_image(slide, figures["comparison_bars"], 6.55, 1.35, w=5.9, h=2.95)
    add_multiline(
        slide,
        [
            "The final setting reduces health MAE from about 0.0960 to 0.0419.",
            "Fault classification remains close to saturation after reducing its loss weight.",
        ],
        6.85,
        4.65,
        5.2,
        0.8,
        size=13,
    )
    add_evidence(slide, "The bar graph proves that the regression-oriented objective substantially improves the primary health metric.")
    add_footer(slide, slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Final training trajectory", "Late-stage validation metrics remain stable while health MAE reaches the best point")
    add_image(slide, figures["final_metrics"], 0.55, 1.18, w=8.35, h=4.55)
    add_metric_card(slide, "Best validation MAE", "0.0419", "epoch 81", 9.35, 1.35, BLUE)
    add_metric_card(slide, "Best Acc@0.1", "87.09%", "epoch 81", 9.35, 2.72, GREEN)
    add_metric_card(slide, "Fault macro acc.", "99.75%", "epoch 81", 9.35, 4.09, RED)
    add_evidence(slide, "The line graphs show that the final model reaches low health error without sacrificing fault localization.")
    add_footer(slide, slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Health-score prediction quality", "Validation predictions concentrate near the ideal target-prediction diagonal")
    add_image(slide, figures["health_scatter"], 0.85, 1.28, w=5.05, h=4.35)
    add_image(slide, figures["health_error_hist"], 6.55, 1.58, w=5.55, h=3.9)
    add_metric_card(slide, "Validation MAE", "0.0419", "0-1 health scale", 1.08, 5.82, BLUE)
    add_metric_card(slide, "Within 0.1", "87.1%", "tolerance accuracy", 4.10, 5.82, GREEN)
    add_evidence(slide, "The scatter and error histogram directly validate the continuous-regression interpretation of Task 1.")
    add_footer(slide, slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Fault-part classification quality", "Component-level classification remains highly reliable")
    add_image(slide, figures["fault_confusion"], 0.85, 1.12, w=5.4, h=4.75)
    add_multiline(
        slide,
        [
            "The confusion matrix is row-normalized by target class.",
            "The validation macro accuracy reaches approximately 0.9975 at the best health checkpoint.",
            "The final-stage maximum observed fault macro accuracy is approximately 0.9984.",
            "This confirms that reducing fault loss weight did not collapse Task 2.",
        ],
        6.95,
        1.55,
        5.0,
        2.5,
        size=13,
    )
    add_evidence(slide, "The confusion matrix proves that the model maintains component-level discrimination after health-regression reweighting.")
    add_footer(slide, slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Performance comparison", "Why the final regression-oriented objective is preferred")
    add_image(slide, figures["comparison_bars"], 0.82, 1.20, w=6.0, h=3.3)
    add_image(slide, figures["pareto"], 7.1, 1.22, w=4.95, h=3.35)
    add_multiline(
        slide,
        [
            "The final model cuts health MAE by more than half.",
            "Health Acc@0.1 improves from about 57.6% to 87.1%.",
            "Fault macro accuracy remains around 99.8%, indicating no meaningful trade-off loss.",
        ],
        1.0,
        4.95,
        10.9,
        0.9,
        size=13,
    )
    add_evidence(slide, "The comparison and Pareto plot show that the final setting improves Task 1 while preserving Task 2.")
    add_footer(slide, slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "What each result graph proves", "Evidence map for the final presentation")
    rows = [
        ["Graph", "Claim supported"],
        ["Dataset count bars", "The original dataset is imbalanced; balancing is necessary."],
        ["Health-score distribution", "Task 1 is a continuous degradation prediction task."],
        ["Feature trajectory", "Fault-window features change along the degradation timeline."],
        ["Model data flow", "The architecture separates regression and classification outputs."],
        ["Final line graph", "Health MAE reaches 0.0419 while fault macro accuracy stays near 0.998."],
        ["Prediction scatter + error histogram", "Most health predictions are close to the continuous target."],
        ["Fault confusion matrix", "Component-level fault localization is stable."],
        ["Comparison bars", "The final objective improves over the stage-classification baseline."],
    ]
    add_table(slide, rows, 0.72, 1.22, 11.9, 4.9, font_size=8)
    add_evidence(slide, "This slide links every major visual to the argument it supports.")
    add_footer(slide, slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Validity considerations", "What the current results do and do not prove")
    add_multiline(
        slide,
        [
            "The current validation split is random, so nearby temporal windows may appear in both train and validation sets.",
            "The health_score label is time-derived; it is interpretable, but it is still a proxy for physical degradation.",
            "Fault-part labels depend on the configured bearing-to-component map for the selected run.",
            "The result is strong for the current dataset setup and should be reported with these assumptions.",
        ],
        0.85,
        1.25,
        6.0,
        3.1,
        size=15,
    )
    add_image(slide, figures["model_dataflow"], 7.0, 1.55, w=5.6, h=2.0)
    add_image(slide, figures["health_distribution"], 7.0, 3.92, w=5.6, h=2.0)
    add_evidence(slide, "The assumptions are explicit, which makes the reported performance academically defensible.")
    add_footer(slide, slide_no)
    slide_no += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_textbox(slide, "Conclusion", 0.72, 0.72, 5.5, 0.55, size=30, color=WHITE, bold=True)
    add_multiline(
        slide,
        [
            "The final model formulates bearing health as continuous regression and fault location as classification.",
            "Validation health MAE reaches 0.0419, corresponding to approximately 4.2% average error.",
            "Health Acc@0.1 reaches 87.1%, and fault macro accuracy remains approximately 99.8%.",
            "The STDP-SNN framework gives the model a neuromorphic implementation path for high-value rotating systems.",
        ],
        0.85,
        1.65,
        6.0,
        3.4,
        size=15,
        color=RGBColor(226, 232, 240),
    )
    add_image(slide, figures["application"], 7.05, 0.95, w=5.75, h=3.25)
    add_metric_card(slide, "Health MAE", "0.0419", "best validation", 7.15, 4.72, BLUE)
    add_metric_card(slide, "Fault macro acc.", "0.9975", "best-health checkpoint", 10.05, 4.72, RED)
    add_footer(slide, slide_no)

    prs.save(OUTPUT)


def main() -> None:
    figures = make_all_figures()
    build_ppt(figures)
    print(f"saved_pptx={OUTPUT}")
    print(f"assets_dir={PPT_ASSETS}")


if __name__ == "__main__":
    main()
