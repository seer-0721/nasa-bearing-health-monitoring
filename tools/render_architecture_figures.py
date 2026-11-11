from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "paper_assets"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "axes.unicode_minus": False,
    }
)


def draw_architecture(title: str, nodes: list[tuple[str, str]], branches: list[tuple[str, str]], filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title_color = "#1f2937"
    box_edge = "#334155"
    arrow_color = "#475569"
    main_fill = "#f8fafc"
    encode_fill = "#e9f0f7"
    core_fill = "#edf4ef"
    readout_fill = "#f7efe8"
    out_fill = "#f4f1f8"

    ax.text(0.5, 0.94, title, ha="center", va="center", fontsize=22, weight="bold", color=title_color)
    ax.plot([0.07, 0.93], [0.89, 0.89], color="#cbd5e1", lw=1.2)

    x_positions = [0.11, 0.27, 0.43, 0.59, 0.75]
    y = 0.56
    w = 0.13
    h = 0.15
    fills = [main_fill, encode_fill, core_fill, core_fill, readout_fill]
    centers: dict[str, tuple[float, float]] = {}

    for (key, label), x, fill in zip(nodes, x_positions, fills):
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.01",
            linewidth=1.2,
            edgecolor=box_edge,
            facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=12, color="#111827", linespacing=1.15)
        centers[key] = (x, y)

    for (src, dst) in zip([n[0] for n in nodes[:-1]], [n[0] for n in nodes[1:]]):
        sx, sy = centers[src]
        dx, dy = centers[dst]
        arrow = FancyArrowPatch(
            (sx + w / 2 + 0.01, sy),
            (dx - w / 2 - 0.01, dy),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.25,
            color=arrow_color,
        )
        ax.add_patch(arrow)

    branch_x = 0.90
    branch_y = [0.66, 0.43]
    branch_w = 0.14
    branch_h = 0.13
    for idx, (_, label) in enumerate(branches):
        by = branch_y[idx]
        box = FancyBboxPatch(
            (branch_x - branch_w / 2, by - branch_h / 2),
            branch_w,
            branch_h,
            boxstyle="round,pad=0.012,rounding_size=0.01",
            linewidth=1.2,
            edgecolor=box_edge,
            facecolor=out_fill,
        )
        ax.add_patch(box)
        ax.text(branch_x, by, label, ha="center", va="center", fontsize=12, color="#111827", linespacing=1.15)
        sx, sy = centers[nodes[-1][0]]
        arrow = FancyArrowPatch(
            (sx + w / 2 + 0.01, sy),
            (branch_x - branch_w / 2 - 0.01, by),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.25,
            color=arrow_color,
            connectionstyle="arc3,rad=0.08" if by > sy else "arc3,rad=-0.08",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.22,
        "Two-task output: continuous health-score regression and fault-part classification.",
        ha="center",
        va="center",
        fontsize=13,
        color="#334155",
    )
    ax.text(
        0.5,
        0.17,
        "The auxiliary health stages are used only for diagnostics and are not model outputs.",
        ha="center",
        va="center",
        fontsize=12,
        color="#64748b",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / filename, dpi=300)
    plt.close(fig)


def main() -> None:
    draw_architecture(
        "Simplified STDP-SNN Architecture",
        [
            ("x", "Normalized\nbearing\nfeatures"),
            ("r", "Rate spike\nencoder"),
            ("s", "Single STDP\nsynapse layer"),
            ("l", "LIF neuron\nresponse"),
            ("f", "Spike and\nmembrane\nstatistics"),
        ],
        [
            ("h", "Health-score\nregression"),
            ("c", "Fault-part\nclassification"),
        ],
        "fig_stdp_snn_architecture.png",
    )
    draw_architecture(
        "RC-ESN Baseline Architecture",
        [
            ("x", "Normalized\nbearing\nfeatures"),
            ("q", "Segmented\npseudo\nsequence"),
            ("e", "Fixed sparse\nESN reservoir"),
            ("f", "Final, mean\nand variance\nstates"),
            ("r", "Raw feature\nconcatenation"),
        ],
        [
            ("h", "Health-score\nregression"),
            ("c", "Fault-part\nclassification"),
        ],
        "fig_rc_esn_architecture.png",
    )


if __name__ == "__main__":
    main()
