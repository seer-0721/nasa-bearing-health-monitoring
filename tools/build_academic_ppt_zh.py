from __future__ import annotations

import sys
from pathlib import Path

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
PPT_ASSETS = DOCS / "ppt_assets_zh"
OUT_ZH = DOCS / "nasa_bearing_snn_academic_report_zh.pptx"
OUT_MAIN = DOCS / "nasa_bearing_snn_academic_report.pptx"
CHECKPOINT = ROOT / "checkpoints" / "nasa_bearing_snn_health_reg_v3" / "best.pt"

INK = RGBColor(45, 55, 65)
TEXT = RGBColor(76, 86, 96)
MUTED_BLUE = RGBColor(93, 126, 154)
MUTED_GREEN = RGBColor(112, 144, 124)
MUTED_RED = RGBColor(167, 104, 98)
MUTED_GOLD = RGBColor(181, 151, 105)
GRID = RGBColor(224, 229, 233)
PALE = RGBColor(248, 249, 250)
WHITE = RGBColor(255, 255, 255)

PLOT_BLUE = "#5d7e9a"
PLOT_GREEN = "#70907c"
PLOT_RED = "#a76862"
PLOT_GOLD = "#b59769"
PLOT_GREY = "#8b949e"
PLOT_LIGHT = "#edf1f4"


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
            "savefig.dpi": 700,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#6b7280",
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


def add_clean_grid(ax) -> None:
    ax.grid(True, linestyle=":", linewidth=0.6, color="#cbd5df", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_dataset_count_plot(good_rows: int, fault_rows: int) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    labels = ["正常窗口", "故障窗口", "拆分后 normal", "拆分后 inner", "拆分后 rolling"]
    values = [good_rows, fault_rows, good_rows * 4 + fault_rows * 2, fault_rows, fault_rows]
    colors = [PLOT_BLUE, PLOT_RED, PLOT_GREY, PLOT_GOLD, PLOT_GREEN]
    ax.bar(labels, values, color=colors, edgecolor="#4b5563", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_ylabel("Sample count, log scale")
    ax.set_title("Dataset source and label distribution")
    ax.tick_params(axis="x", rotation=18)
    for i, value in enumerate(values):
        ax.text(i, value * 1.25, f"{value:,}", ha="center", fontsize=8, color="#334155")
    add_clean_grid(ax)
    return save_fig(fig, "zh_dataset_count")


def make_raw_waveform_plot() -> Path:
    df = pd.read_csv(DATA_ROOT / "test_final.csv", nrows=2400)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    offsets = [0.0, 1.15, 2.3, 3.45]
    colors = [PLOT_BLUE, PLOT_GREEN, PLOT_GOLD, PLOT_RED]
    for idx, col in enumerate(["Bearing1", "Bearing2", "Bearing3", "Bearing4"]):
        y = df[col].to_numpy(dtype=float)
        y = y / max(float(np.std(y)), 1e-6) * 0.20 + offsets[idx]
        ax.plot(x, y, linewidth=0.75, color=colors[idx], label=f"B{idx + 1}")
    ax.set_title("Raw vibration segment")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Normalized amplitude + offset")
    ax.set_yticks(offsets)
    ax.set_yticklabels(["B1", "B2", "B3", "B4"])
    ax.legend(ncol=4, loc="upper right", fontsize=8)
    add_clean_grid(ax)
    return save_fig(fig, "zh_raw_waveform")


def make_health_distribution_plot(good_rows: int, fault_rows: int) -> Path:
    normal_scores = np.ones(good_rows * 4 + fault_rows * 2, dtype=np.float32)
    fault_scores = 1.0 - np.arange(fault_rows, dtype=np.float32) / max(fault_rows - 1, 1)
    scores = np.concatenate([normal_scores, fault_scores, fault_scores])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    axes[0].hist(scores, bins=np.linspace(0, 1, 12), color=PLOT_BLUE, edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_title("Health-score label distribution")
    axes[0].set_xlabel("health_score")
    axes[0].set_ylabel("Sample count, log scale")
    add_clean_grid(axes[0])
    x = np.arange(fault_rows)
    axes[1].plot(x, fault_scores, color=PLOT_RED, linewidth=1.8)
    axes[1].set_title("Time-derived degradation target")
    axes[1].set_xlabel("Fault-window position")
    axes[1].set_ylabel("health_score")
    axes[1].set_ylim(-0.04, 1.04)
    add_clean_grid(axes[1])
    return save_fig(fig, "zh_health_distribution")


def make_feature_plots(good_sample: pd.DataFrame, fault_df: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    for ax, col, title in [
        (axes[0], "B3__rms", "B3 RMS feature"),
        (axes[1], "B4__p2p", "B4 peak-to-peak feature"),
    ]:
        box = ax.boxplot(
            [good_sample[col].dropna().to_numpy(), fault_df[col].dropna().to_numpy()],
            tick_labels=["Good", "Fault"],
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(box["boxes"], ["#d8e3ea", "#eadbd2"]):
            patch.set_facecolor(color)
            patch.set_edgecolor("#6b7280")
        ax.set_title(title)
        ax.set_ylabel("Feature value")
        add_clean_grid(ax)
    box_path = save_fig(fig, "zh_feature_boxplot")

    x = np.linspace(0, 1, len(fault_df))
    fig, ax = plt.subplots(figsize=(9.2, 3.35))
    for bearing, color in [("B3", PLOT_GOLD), ("B4", PLOT_GREEN)]:
        for feature, style in [("rms", "-"), ("p2p", "--")]:
            y = fault_df[f"{bearing}__{feature}"].rolling(31, center=True, min_periods=1).mean()
            y = (y - y.min()) / max(float(y.max() - y.min()), 1e-9)
            ax.plot(x, y, linestyle=style, color=color, linewidth=1.45, label=f"{bearing} {feature}")
    ax.plot(x, 1 - x, color="#2f3741", linewidth=1.5, label="Target health_score")
    ax.set_title("Feature evolution along the fault timeline")
    ax.set_xlabel("Normalized fault timeline")
    ax.set_ylabel("Normalized value")
    ax.legend(
        fontsize=8,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        borderaxespad=0.0,
        handlelength=2.4,
        columnspacing=1.2,
    )
    add_clean_grid(ax)
    traj_path = save_fig(fig, "zh_feature_trajectory")
    return box_path, traj_path


def make_correlation_heatmap(fault_df: pd.DataFrame) -> Path:
    cols = [c for c in fault_df.columns if c.startswith("B3__")]
    labels = [c.split("__", 1)[1] for c in cols]
    corr = fault_df[cols].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(5.1, 4.7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.82)
    ax.set_title("B3 feature correlation during fault period")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    return save_fig(fig, "zh_feature_correlation")


def make_stdp_curve_plot() -> Path:
    params = np.load(ROOT / "stdp_params.npz")
    tau_plus = float(params["tau_plus"])
    tau_minus = float(params["tau_minus"])
    window = max(tau_plus, tau_minus)
    dt = np.linspace(-window, window, 700)
    # Normalize amplitudes for presentation; raw fitted amplitudes are reported in text/config.
    dw = np.where(dt >= 0, np.exp(-dt / tau_plus), -np.exp(dt / tau_minus))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(dt, dw, color=PLOT_BLUE, linewidth=1.8)
    ax.axhline(0, color="#475569", linewidth=0.8)
    ax.axvline(0, color="#475569", linewidth=0.8)
    ax.set_title("Normalized STDP learning window")
    ax.set_xlabel("Spike timing difference Δt")
    ax.set_ylabel("Normalized synaptic update")
    ax.text(
        0.02,
        0.91,
        f"τ+={tau_plus:.0f}, τ-={tau_minus:.0f}",
        transform=ax.transAxes,
        fontsize=8,
        color="#5f6b76",
    )
    ax.set_ylim(-1.08, 1.08)
    add_clean_grid(ax)
    return save_fig(fig, "zh_stdp_curve")


def make_pipeline_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(10.0, 2.7))
    ax.axis("off")
    steps = [
        ("Raw vibration", "Four bearing channels"),
        ("Feature extraction", "20 features per bearing"),
        ("Per-bearing split", "B1/B2/B3/B4"),
        ("Label construction", "health + fault"),
        ("STDP-SNN", "spike inference"),
        ("Outputs", "health + fault part"),
    ]
    xs = np.linspace(0.07, 0.93, len(steps))
    for i, (title, sub) in enumerate(steps):
        rect = plt.Rectangle((xs[i] - 0.07, 0.36), 0.14, 0.30, facecolor="#f6f8fa", edgecolor="#8b949e", linewidth=0.9)
        ax.add_patch(rect)
        ax.text(xs[i], 0.56, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#2f3741")
        ax.text(xs[i], 0.44, sub, ha="center", va="center", fontsize=8, color="#5f6b76")
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.078, 0.51), xytext=(xs[i] + 0.078, 0.51), arrowprops=dict(arrowstyle="->", lw=1.0, color="#7d95a8"))
    return save_fig(fig, "zh_pipeline")


def make_model_dataflow_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(10.0, 3.0))
    ax.axis("off")
    blocks = [("24-D input", "#eef3f6"), ("Spike coding", "#edf4f5"), ("STDP layer", "#f4efe4"), ("LIF dynamics", "#eef4ed"), ("Shared state", "#f7f7f7")]
    xs = np.linspace(0.09, 0.65, len(blocks))
    for i, (label, color) in enumerate(blocks):
        ax.add_patch(plt.Rectangle((xs[i] - 0.055, 0.55), 0.11, 0.18, facecolor=color, edgecolor="#8b949e", linewidth=0.9))
        ax.text(xs[i], 0.64, label, ha="center", va="center", fontsize=9.5, color="#2f3741", fontweight="bold")
        if i < len(blocks) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.062, 0.64), xytext=(xs[i] + 0.062, 0.64), arrowprops=dict(arrowstyle="->", lw=1.0, color="#7d95a8"))
    heads = [("health_score", "Regression", "#eef4ed", 0.82, 0.78), ("health_stage", "Auxiliary", "#eef3f6", 0.82, 0.54), ("fault_part", "Classification", "#f4e8e5", 0.82, 0.30)]
    for title, sub, color, x, y in heads:
        ax.add_patch(plt.Rectangle((x - 0.085, y - 0.055), 0.17, 0.11, facecolor=color, edgecolor="#8b949e", linewidth=0.9))
        ax.text(x, y + 0.018, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#2f3741")
        ax.text(x, y - 0.027, sub, ha="center", va="center", fontsize=8, color="#5f6b76")
        ax.annotate("", xy=(x - 0.092, y), xytext=(0.71, 0.64), arrowprops=dict(arrowstyle="->", lw=0.9, color="#8b949e"))
    return save_fig(fig, "zh_model_dataflow")


def make_final_metrics_plot(final_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.55), sharex=True)
    plots = [
        ("val_health_mae", "train_health_mae", "Health MAE", "MAE", (0.035, 0.125), PLOT_BLUE),
        ("val_health_acc_at_0_1", "train_health_acc_at_0_1", "Health Acc@0.1", "Accuracy", (0.55, 0.90), PLOT_GREEN),
        ("val_fault_macro_acc", "train_fault_macro_acc", "Fault Macro Accuracy", "Accuracy", (0.995, 1.0), PLOT_RED),
    ]
    for ax, (val_col, train_col, title, ylabel, ylim, color) in zip(axes, plots):
        ax.plot(final_df["epoch"], final_df[train_col], color=PLOT_GREY, linewidth=1.2, linestyle="--", label="Train")
        ax.plot(final_df["epoch"], final_df[val_col], color=color, linewidth=1.7, marker="o", markersize=2.5, label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        add_clean_grid(ax)
    return save_fig(fig, "zh_final_metrics")


def make_comparison_bar_plot() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    labels = ["Stage-classification", "Continuous-regression"]
    x = np.arange(2)
    mae = [0.0960, 0.0419]
    acc = [0.5755, 0.8709]
    axes[0].bar(x, mae, color=[PLOT_GREY, PLOT_BLUE], edgecolor="#6b7280", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Health MAE")
    axes[0].set_ylabel("Lower is better")
    axes[1].bar(x, acc, color=[PLOT_GREY, PLOT_GREEN], edgecolor="#6b7280", linewidth=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Health Acc@0.1")
    axes[1].set_ylabel("Higher is better")
    for ax, values, fmt in [(axes[0], mae, "{:.4f}"), (axes[1], acc, "{:.1%}")]:
        for i, v in enumerate(values):
            ax.text(i, v + (0.004 if max(values) < 0.2 else 0.03), fmt.format(v), ha="center", fontsize=8.5, color="#334155")
        add_clean_grid(ax)
    return save_fig(fig, "zh_comparison_bars")


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


def make_prediction_plots(pred_h, target_h, errors) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.scatter(target_h, pred_h, s=7, alpha=0.28, color=PLOT_BLUE, edgecolor="none")
    ax.plot([0, 1], [0, 1], color=PLOT_RED, linewidth=1.3, linestyle="--", label="Ideal")
    ax.set_title("Health-score regression")
    ax.set_xlabel("Target health_score")
    ax.set_ylabel("Predicted health_score")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    add_clean_grid(ax)
    scatter = save_fig(fig, "zh_prediction_scatter")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.hist(errors, bins=np.linspace(0, 0.35, 36), color=PLOT_GREEN, edgecolor="white")
    ax.axvline(0.1, color=PLOT_RED, linewidth=1.3, linestyle="--", label="0.1 tolerance")
    ax.set_title("Absolute health-score error")
    ax.set_xlabel("|prediction - target|")
    ax.set_ylabel("Validation samples")
    ax.legend(fontsize=8)
    add_clean_grid(ax)
    hist = save_fig(fig, "zh_error_hist")
    return scatter, hist


def make_confusion_matrix(pred_f, target_f) -> Path:
    all_names = list(NASA_BEARING_FAULT_PARTS)
    present_indices = sorted(set(target_f.astype(int).tolist()) | set(pred_f.astype(int).tolist()))
    names = [all_names[i] for i in present_indices]
    index_map = {old: new for new, old in enumerate(present_indices)}
    cm = np.zeros((len(names), len(names)), dtype=float)
    for t, p in zip(target_f, pred_f):
        cm[index_map[int(t)], index_map[int(p)]] += 1
    support = cm.sum(axis=1, keepdims=True)
    cm = np.divide(cm, support, out=np.zeros_like(cm), where=support > 0)
    fig, ax = plt.subplots(figsize=(4.5, 3.9))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, alpha=0.78)
    ax.set_title("Fault-part confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            if support[i, 0] > 0 and cm[i, j] > 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#1f2937")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save_fig(fig, "zh_fault_confusion")


def make_all_figures() -> dict[str, Path]:
    configure_plot_style()
    PPT_ASSETS.mkdir(parents=True, exist_ok=True)
    good_rows = line_count(DATA_ROOT / "features_good_named.csv")
    fault_rows = line_count(DATA_ROOT / "features_fault_named.csv")
    good_sample = sample_csv(DATA_ROOT / "features_good_named.csv", 18000, 42)
    fault_df = pd.read_csv(DATA_ROOT / "features_fault_named.csv")
    final_df = pd.read_csv(ASSETS / "nasa_bearing_final_metrics.csv")
    pred_h, target_h, errors, pred_f, target_f = collect_predictions()
    scatter, hist = make_prediction_plots(pred_h, target_h, errors)
    box, traj = make_feature_plots(good_sample, fault_df)
    return {
        "application": ASSETS / "aerospace_bearing_snn_watercolor.png",
        "dataset_counts": make_dataset_count_plot(good_rows, fault_rows),
        "raw_waveform": make_raw_waveform_plot(),
        "health_distribution": make_health_distribution_plot(good_rows, fault_rows),
        "feature_box": box,
        "feature_traj": traj,
        "corr": make_correlation_heatmap(fault_df),
        "pipeline": make_pipeline_diagram(),
        "model_flow": make_model_dataflow_diagram(),
        "stdp": make_stdp_curve_plot(),
        "final_metrics": make_final_metrics_plot(final_df),
        "compare": make_comparison_bar_plot(),
        "scatter": scatter,
        "hist": hist,
        "confusion": make_confusion_matrix(pred_f, target_f),
    }


def font(run, size=14, color=TEXT, bold=False):
    run.font.name = "Noto Sans CJK SC"
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold


def set_bg(slide, color=WHITE):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def text(slide, s, x, y, w, h, size=14, color=TEXT, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = s
    font(r, size, color, bold)
    return box


def title(slide, s, sub=None):
    text(slide, s, 0.55, 0.28, 11.8, 0.35, size=22, color=INK, bold=True)
    if sub:
        text(slide, sub, 0.57, 0.70, 11.4, 0.26, size=10, color=TEXT)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.00), Inches(12.15), Inches(0.018))
    line.fill.solid()
    line.fill.fore_color.rgb = GRID
    line.line.fill.background()


def bullets(slide, items, x, y, w, h, size=12):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {item}"
        p.font.name = "Noto Sans CJK SC"
        p.font.size = Pt(size)
        p.font.color.rgb = TEXT
        p.space_after = Pt(5)
    return box


def pic(slide, path, x, y, w=None, h=None):
    path = Path(path)
    if w and h:
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
        return slide.shapes.add_picture(
            str(path),
            Inches(draw_x),
            Inches(draw_y),
            width=Inches(draw_w),
            height=Inches(draw_h),
        )
    if w:
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    if h:
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y), height=Inches(h))
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y))


def note(slide, s):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.72), Inches(6.82), Inches(11.9), Inches(0.28))
    shape.fill.solid()
    shape.fill.fore_color.rgb = PALE
    shape.line.color.rgb = GRID
    text(slide, f"图示证明：{s}", 0.83, 6.875, 11.6, 0.16, size=8.6, color=TEXT)


def footer(slide, n):
    text(slide, f"NASA Bearing STDP-SNN | {n}", 10.75, 7.18, 1.8, 0.16, size=7.5, color=TEXT, align=PP_ALIGN.RIGHT)


def table(slide, rows, x, y, w, h, size=8.5):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = shape.table
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(241, 244, 246) if ri == 0 else WHITE
            for p in cell.text_frame.paragraphs:
                p.font.name = "Noto Sans CJK SC"
                p.font.size = Pt(size)
                p.font.bold = ri == 0
                p.font.color.rgb = INK if ri == 0 else TEXT
    return shape


def metric(slide, label, value, x, y, color=MUTED_BLUE):
    text(slide, value, x, y, 1.85, 0.28, size=21, color=color, bold=True)
    text(slide, label, x, y + 0.37, 2.1, 0.24, size=9.5, color=TEXT)


def section(prs, s, sub, n):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    text(slide, s, 0.8, 2.75, 10.8, 0.45, size=25, color=INK, bold=True)
    text(slide, sub, 0.82, 3.28, 10.0, 0.28, size=13, color=TEXT)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.82), Inches(3.72), Inches(4.0), Inches(0.025))
    line.fill.solid()
    line.fill.fore_color.rgb = MUTED_BLUE
    line.line.fill.background()
    footer(slide, n)


def build_ppt(fig: dict[str, Path]):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    n = 1

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    pic(slide, fig["application"], 0.15, 0.62, w=12.95, h=4.62)
    text(slide, "基于 STDP-SNN 的轴承健康度预测与故障部位识别", 0.72, 0.34, 9.4, 0.36, size=21, color=INK, bold=True)
    text(slide, "NASA Bearing Dataset | 连续健康度回归 + 故障部位分类", 0.74, 5.48, 7.0, 0.25, size=12, color=TEXT)
    metric(slide, "验证健康度 MAE", "0.0419", 0.78, 6.08, MUTED_BLUE)
    metric(slide, "健康度 Acc@0.1", "87.1%", 3.1, 6.08, MUTED_GREEN)
    metric(slide, "故障部位 Macro Acc", "99.75%", 5.42, 6.08, MUTED_RED)
    footer(slide, n); n += 1

    section(prs, "Part 1. 总体概览", "问题定义、任务拆分与应用场景", n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "任务定义", "将寿命/健康程度建模为连续预测，将故障部位建模为分类")
    bullets(slide, [
        "Task 1：预测 health_score，范围 0.0 到 1.0，属于回归任务。",
        "Task 2：预测 fault_part，输出 normal / inner_ring / rolling_element 等类别。",
        "健康阶段标签仅作为辅助约束，不作为主要评价指标。",
        "主要评价：健康度 MAE、Acc@0.1；故障部位 macro accuracy。"
    ], 0.85, 1.42, 5.45, 2.85, size=13)
    pic(slide, fig["application"], 6.72, 1.30, w=5.65, h=3.18)
    metric(slide, "健康度平均误差", "4.2%", 1.05, 5.25, MUTED_BLUE)
    metric(slide, "误差 ≤ 0.1 的比例", "87.1%", 3.55, 5.25, MUTED_GREEN)
    metric(slide, "部位识别 macro accuracy", "99.75%", 6.05, 5.25, MUTED_RED)
    note(slide, "任务拆分避免把连续退化过程强行解释为普通多分类。")
    footer(slide, n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "应用场景", "面向高价值旋转部件的在线健康监测")
    pic(slide, fig["application"], 0.65, 1.20, w=7.2, h=4.05)
    bullets(slide, [
        "振动传感器捕获轴承状态变化。",
        "特征被编码为脉冲输入，送入 STDP-SNN。",
        "模型同时输出健康度和故障部位。",
        "淡彩示意图表达航天旋转部件上的潜在部署场景。"
    ], 8.35, 1.42, 3.85, 2.55, size=12.5)
    note(slide, "该应用图用于说明模型从离线数据集走向工程监测系统的逻辑链。")
    footer(slide, n); n += 1

    section(prs, "Part 2. 数据集介绍", "数据描述、可视化与样本构造流程", n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "数据集基本情况", "四个轴承通道，每个轴承提取 20 个工程特征")
    pic(slide, fig["dataset_counts"], 0.72, 1.25, w=6.35, h=2.85)
    table(slide, [
        ["项目", "数值"],
        ["正常特征行数", "222,811"],
        ["故障特征行数", "1,703"],
        ["轴承通道", "B1, B2, B3, B4"],
        ["每轴承特征数", "20"],
        ["实际输入维度", "24 = 20 特征 + 4 维轴承编号"],
    ], 7.55, 1.35, 4.55, 2.45, size=9)
    bullets(slide, ["原始 good/fault 文件比例高度不均衡。", "拆分后可形成与具体轴承对应的故障部位标签。"], 0.95, 4.85, 10.6, 0.75, size=12.5)
    note(slide, "柱状图说明为什么训练中需要类别权重和采样平衡。")
    footer(slide, n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "数据可视化", "原始振动和工程特征都包含可区分的状态信息")
    pic(slide, fig["raw_waveform"], 0.65, 1.25, w=5.95, h=2.45)
    pic(slide, fig["feature_box"], 6.92, 1.25, w=5.45, h=2.35)
    pic(slide, fig["feature_traj"], 0.65, 4.08, w=6.15, h=2.25)
    pic(slide, fig["corr"], 7.35, 3.82, w=3.85, h=2.95)
    note(slide, "波形、箱线图、时间轨迹和相关性图共同说明工程特征能够反映状态差异。")
    footer(slide, n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "数据处理流程", "从四通道振动文件到单轴承监督样本")
    pic(slide, fig["pipeline"], 0.8, 1.35, w=11.7, h=3.0)
    bullets(slide, [
        "每行 80 维特征被拆成 4 条单轴承样本。",
        "B3 默认标为 inner_ring，B4 默认标为 rolling_element。",
        "B1/B2 在故障段仍作为 normal，用于部位分类的负样本。",
        "健康度按故障段时间顺序从 1.0 线性衰减到 0.0。"
    ], 1.0, 4.72, 11.2, 1.05, size=12.5)
    note(slide, "流程图说明标签是如何与具体轴承和具体任务对齐的。")
    footer(slide, n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "健康度标签", "连续 health_score 是 Task 1 的核心监督信号")
    pic(slide, fig["health_distribution"], 0.85, 1.30, w=7.2, h=2.75)
    table(slide, [
        ["health_score", "含义"],
        ["1.0", "健康"],
        ["0.5", "中间退化"],
        ["0.0", "接近失效"],
        ["health_stage", "仅辅助训练和展示"],
    ], 8.45, 1.38, 3.65, 2.2, size=9)
    bullets(slide, [
        "当前不再把健康度当作主分类任务。",
        "MAE 和 Acc@0.1 比 health_stage_acc 更适合评价连续预测。"
    ], 1.0, 4.55, 10.6, 0.8, size=12.5)
    note(slide, "标签分布图说明 Task 1 应被看作连续回归任务。")
    footer(slide, n); n += 1

    section(prs, "Part 3. 模型", "模型结构、数据流与 STDP 器件层", n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "模型结构", "共享 STDP-SNN 主干 + 三个输出头")
    pic(slide, fig["model_flow"], 0.75, 1.25, w=11.7, h=3.55)
    table(slide, [
        ["模块", "设置"],
        ["输入", "24 维"],
        ["时间步", "16"],
        ["STDP 层", "24 -> 384"],
        ["全连接层", "384 -> 192"],
        ["输出", "health_score / health_stage / fault_part"],
    ], 1.0, 5.0, 5.15, 1.35, size=8.5)
    bullets(slide, [
        "health_score 是主要回归输出。",
        "health_stage 用于辅助约束退化顺序。",
        "fault_part 用于故障部位分类。"
    ], 6.75, 5.02, 5.2, 1.15, size=11.5)
    note(slide, "模型结构图说明同一脉冲主干如何同时支持预测和诊断。")
    footer(slide, n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "STDP 器件层", "第一层使用器件拟合的脉冲时序可塑性参数")
    pic(slide, fig["stdp"], 0.85, 1.45, w=6.2, h=3.0)
    bullets(slide, [
        "STDP 根据脉冲时间差调整突触权重。",
        "第一层可进行无监督预训练。",
        "随后通过监督训练优化健康度与故障部位任务。",
        "该设计比普通 MLP 更接近神经形态计算流程。"
    ], 7.55, 1.62, 4.25, 2.45, size=12.5)
    note(slide, "归一化 STDP 窗口说明模型第一层具有明确的器件/脉冲学习解释。")
    footer(slide, n); n += 1

    section(prs, "Part 4. 结果", "健康度回归、故障部位分类与对比分析", n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "最终训练配置", "为降低健康度误差而重新加权损失函数")
    table(slide, [
        ["设置", "取值"],
        ["训练轮数", "100"],
        ["Batch size", "256"],
        ["健康度损失", "L1，权重 2.0"],
        ["辅助阶段损失", "CE，权重 0.05"],
        ["故障分类损失", "CE，权重 0.2"],
        ["采样策略", "故障类别平衡"],
        ["STDP 预训练", "1 epoch"],
    ], 0.85, 1.35, 4.8, 3.1, size=9)
    pic(slide, fig["compare"], 6.2, 1.35, w=5.7, h=2.6)
    bullets(slide, [
        "此前健康度 MAE 约为 0.096，偏高。",
        "最终版本直接优化 L1，并提高健康度回归权重。",
        "故障分类已经较稳定，因此降低其损失权重。"
    ], 6.35, 4.55, 5.5, 1.05, size=12)
    note(slide, "对比柱状图证明回归导向的训练目标显著改善健康度预测。")
    footer(slide, n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "最终训练曲线", "健康度误差下降，同时故障部位分类保持稳定")
    pic(slide, fig["final_metrics"], 0.62, 1.12, w=12.05, h=3.55)
    metric(slide, "最佳验证 MAE", "0.0419", 1.15, 5.42, MUTED_BLUE)
    metric(slide, "最佳 Acc@0.1", "87.09%", 4.75, 5.42, MUTED_GREEN)
    metric(slide, "故障 Macro Acc", "99.75%", 8.35, 5.42, MUTED_RED)
    note(slide, "折线图证明健康度回归达到较低误差，并且故障部位识别没有被牺牲。")
    footer(slide, n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "健康度预测质量", "预测值集中在理想对角线附近，误差多数低于 0.1")
    pic(slide, fig["scatter"], 0.95, 1.25, w=4.75, h=3.95)
    pic(slide, fig["hist"], 6.55, 1.45, w=4.95, h=3.55)
    metric(slide, "MAE", "0.0419", 1.15, 5.55, MUTED_BLUE)
    metric(slide, "Acc@0.1", "87.1%", 3.2, 5.55, MUTED_GREEN)
    note(slide, "散点图和误差直方图直接证明 Task 1 的连续回归效果。")
    footer(slide, n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "故障部位识别质量", "部位分类在验证集上保持接近满分的 macro accuracy")
    pic(slide, fig["confusion"], 0.9, 1.18, w=5.2, h=4.45)
    bullets(slide, [
        "混淆矩阵按目标类别做行归一化。",
        "图中只显示当前验证集中实际出现或被预测到的故障类别。",
        "最佳健康度 checkpoint 的故障部位 macro accuracy 为 0.9975。",
        "训练后期最高约 0.9984。",
        "降低故障损失权重后，Task 2 仍然保持稳定。"
    ], 6.9, 1.55, 4.95, 2.55, size=12.5)
    note(slide, "混淆矩阵证明模型仍能稳定区分 normal、inner_ring 和 rolling_element。")
    footer(slide, n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "结果总结", "最终模型同时满足健康度预测和故障部位识别")
    table(slide, [
        ["指标", "结果", "解释"],
        ["健康度 MAE", "0.0419", "平均误差约 4.2%"],
        ["健康度 Acc@0.1", "0.8709", "87.1% 样本误差不超过 0.1"],
        ["故障部位 Macro Acc", "0.9975", "按类别平均后的部位识别准确率"],
        ["中间版本 MAE", "约 0.096", "说明最终训练目标明显改善"],
    ], 0.95, 1.35, 11.1, 2.3, size=9.5)
    bullets(slide, [
        "当前结果适合作为正式汇报版本。",
        "汇报时应强调 Task 1 是连续健康度回归，不是阶段分类。",
        "当前验证集为随机切分，结果可能比严格时间外推略乐观。"
    ], 1.05, 4.32, 10.5, 1.05, size=12.5)
    note(slide, "该表格汇总了最终可汇报的核心证据。")
    footer(slide, n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "每类图表证明什么", "避免只展示结果，而不说明图的论证作用")
    table(slide, [
        ["图表", "证明内容"],
        ["数据分布柱状图", "数据不平衡，训练需要权重和平衡采样"],
        ["健康度分布图", "Task 1 应作为连续预测任务"],
        ["特征箱线图/轨迹图", "工程特征随状态和故障时间变化"],
        ["模型数据流图", "模型结构匹配回归 + 分类任务拆分"],
        ["最终折线图", "MAE 降到 0.0419，故障 macro acc 保持约 99.8%"],
        ["散点图/误差直方图", "连续健康度预测接近目标值"],
        ["混淆矩阵", "故障部位分类可靠"],
    ], 0.85, 1.25, 11.6, 4.7, size=8.8)
    note(slide, "该页把每张图与其支持的学术论点对应起来。")
    footer(slide, n); n += 1

    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    title(slide, "结论", "STDP-SNN 可以同时进行轴承健康度预测与故障部位识别")
    pic(slide, fig["application"], 6.55, 1.22, w=5.85, h=3.3)
    bullets(slide, [
        "最终版本把健康程度建模为连续 health_score，平均误差约 4.2%。",
        "87.1% 的验证样本健康度误差不超过 0.1。",
        "故障部位 macro accuracy 达到约 99.75%。",
        "模型结构保留了器件拟合 STDP 与脉冲神经网络的神经形态特征。"
    ], 0.92, 1.45, 5.1, 2.7, size=13.2)
    metric(slide, "健康度 MAE", "0.0419", 1.05, 5.35, MUTED_BLUE)
    metric(slide, "故障部位 Macro Acc", "0.9975", 3.55, 5.35, MUTED_RED)
    note(slide, "最终结论围绕两个任务的核心指标展开，避免用辅助 stage accuracy 混淆主任务。")
    footer(slide, n)

    prs.save(OUT_ZH)
    prs.save(OUT_MAIN)


def main():
    figures = make_all_figures()
    build_ppt(figures)
    print(f"saved={OUT_ZH}")
    print(f"updated={OUT_MAIN}")
    print(f"assets={PPT_ASSETS}")


if __name__ == "__main__":
    main()
