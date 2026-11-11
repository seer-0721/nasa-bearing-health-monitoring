from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nasa-bearing")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_dense_academic_ppt_zh as base


DOCS = ROOT / "docs"
PPT_ASSETS = DOCS / "ppt_assets_weekly_style_zh"
OUTPUT = DOCS / "nasa_bearing_snn_academic_report_weekly_style_zh.pptx"

FONT = "微软雅黑"
NAVY = RGBColor(0, 0, 68)
RED = RGBColor(255, 0, 0)
GREY = RGBColor(90, 96, 110)
LIGHT = RGBColor(240, 243, 246)
WHITE = RGBColor(255, 255, 255)
BLUE = RGBColor(76, 106, 132)
GREEN = RGBColor(99, 128, 111)
BROWN = RGBColor(148, 91, 84)

P_NAVY = "#000044"
P_RED = "#ff0000"
P_BLUE = "#4c6a84"
P_GREEN = "#63806f"
P_GREY = "#7b8490"
P_LIGHT = "#f2f4f6"


def emu(x: float) -> int:
    return Inches(x)


def set_font(run, size: float, color: RGBColor = NAVY, bold: bool = False) -> None:
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold


def textbox(slide, text: str, x: float, y: float, w: float, h: float, size: float = 15, color: RGBColor = NAVY, bold: bool = False, align=None):
    shape = slide.shapes.add_textbox(emu(x), emu(y), emu(w), emu(h))
    tf = shape.text_frame
    tf.clear()
    tf.margin_left = emu(0.03)
    tf.margin_right = emu(0.03)
    tf.margin_top = emu(0.02)
    tf.margin_bottom = emu(0.02)
    tf.word_wrap = True
    p = tf.paragraphs[0]
    if align is not None:
        p.alignment = align
    r = p.add_run()
    r.text = text
    set_font(r, size, color, bold)
    return shape


def add_title(slide, text: str, subtitle: str | None = None) -> None:
    textbox(slide, text, 0.42, 0.00, 7.8, 0.64, 28, NAVY, True)
    if subtitle:
        textbox(slide, subtitle, 0.43, 0.62, 8.8, 0.28, 8.8, RED, False)


def add_footer(slide, page: int) -> None:
    return None


def add_picture_fit(slide, path: Path, x: float, y: float, w: float, h: float):
    from PIL import Image

    img = Image.open(path)
    iw, ih = img.size
    scale = min(w / iw, h / ih)
    pw = iw * scale
    ph = ih * scale
    return slide.shapes.add_picture(str(path), emu(x + (w - pw) / 2), emu(y + (h - ph) / 2), width=emu(pw), height=emu(ph))


def add_caption(slide, text: str, x: float, y: float, w: float, h: float = 0.48) -> None:
    textbox(slide, f"图注：{text}", x, y, w, h, 9.4, GREY, False)


def add_note(slide, items: list[str], x: float = 10.0, y: float = 0.88, w: float = 3.05, h: float = 4.5) -> None:
    box = slide.shapes.add_textbox(emu(x), emu(y), emu(w), emu(h))
    tf = box.text_frame
    tf.clear()
    tf.margin_left = emu(0.04)
    tf.margin_right = emu(0.02)
    tf.margin_top = emu(0.02)
    tf.margin_bottom = emu(0.02)
    tf.word_wrap = True
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.level = 0
        p.space_after = Pt(8)
        r = p.add_run()
        r.text = item
        set_font(r, 15.5 if idx == 0 else 13.2, NAVY if idx == 0 else GREY, idx == 0)


def add_metric_row(slide, metrics: list[tuple[str, str, RGBColor]], y: float = 5.62) -> None:
    x0 = 0.55
    width = 3.75
    for idx, (label, value, color) in enumerate(metrics):
        x = x0 + idx * (width + 0.25)
        slide.shapes.add_shape(1, emu(x), emu(y), emu(width), emu(0.72)).line.color.rgb = RGBColor(210, 214, 220)
        textbox(slide, label, x + 0.12, y + 0.12, width - 0.24, 0.18, 8.8, GREY)
        textbox(slide, value, x + 0.12, y + 0.34, width - 0.24, 0.30, 18, color, True)


def table(slide, rows: list[list[str]], x: float, y: float, w: float, h: float, font_size: float = 11.0) -> None:
    cols = len(rows[0])
    shape = slide.shapes.add_table(len(rows), cols, emu(x), emu(y), emu(w), emu(h))
    tbl = shape.table
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = val
            cell.margin_left = emu(0.05)
            cell.margin_right = emu(0.05)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            fill = cell.fill
            fill.solid()
            fill.fore_color.rgb = WHITE
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if r == 0 else PP_ALIGN.LEFT
                for run in p.runs:
                    set_font(run, font_size, NAVY if r == 0 else GREY, r == 0)
    return shape


def save_fig(fig: plt.Figure, name: str) -> Path:
    PPT_ASSETS.mkdir(parents=True, exist_ok=True)
    path = PPT_ASSETS / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=700, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_node(ax, x: float, y: float, w: float, h: float, title: str, sub: str = "", fc: str = P_LIGHT, ec: str = P_GREY, title_size: float = 8.5):
    rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=0.9)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center", fontsize=title_size, fontweight="bold", color="#1f2937", linespacing=1.0)
    if sub:
        ax.text(x + w / 2, y + h * 0.31, sub, ha="center", va="center", fontsize=6.9, color="#4b5563", linespacing=1.0)
    return rect


def arrow(ax, x1: float, y1: float, x2: float, y2: float, lw: float = 1.0) -> None:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#5f6876", lw=lw, shrinkA=0, shrinkB=0))


def make_refined_model_figure() -> Path:
    base.configure_plot_style()
    fig, ax = plt.subplots(figsize=(12.2, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.94, "Multi-task STDP-SNN architecture", fontsize=18, fontweight="bold", color=P_NAVY)
    ax.text(0.02, 0.885, "Continuous health regression and fault-part classification share one spiking representation", fontsize=9.5, color="#4b5563")

    draw_node(ax, 0.03, 0.58, 0.12, 0.17, "Raw window", "B1-B4\nvibration", "#f6f7f8")
    draw_node(ax, 0.03, 0.32, 0.12, 0.17, "Bearing ID", "channel\nidentity", "#f6f7f8")
    draw_node(ax, 0.20, 0.45, 0.12, 0.20, "Feature vector", "20-D feature\nper bearing", "#eef3f6")
    draw_node(ax, 0.36, 0.45, 0.12, 0.20, "Spike encoder", "value-to-spike\nsequence", "#eef3f6")

    draw_node(ax, 0.54, 0.64, 0.14, 0.15, "STDP plasticity", "timing-dependent\nfeature expansion", "#f4efe4")
    draw_node(ax, 0.54, 0.43, 0.14, 0.15, "LIF integration", "membrane state\nand spike response", "#eef4ed")
    draw_node(ax, 0.54, 0.22, 0.14, 0.15, "Shared state", "compact spiking\nrepresentation", "#f7f7f7")

    draw_node(ax, 0.78, 0.68, 0.15, 0.13, "health_score", "regression\nmain output", "#eef4ed")
    draw_node(ax, 0.78, 0.47, 0.15, 0.13, "health_stage", "auxiliary order\nconstraint", "#eef3f6")
    draw_node(ax, 0.78, 0.26, 0.15, 0.13, "fault_part", "normal / inner\n/ rolling", "#f4e8e5")

    arrow(ax, 0.15, 0.665, 0.20, 0.56)
    arrow(ax, 0.15, 0.405, 0.20, 0.54)
    arrow(ax, 0.32, 0.55, 0.36, 0.55)
    arrow(ax, 0.48, 0.55, 0.54, 0.71)
    arrow(ax, 0.48, 0.55, 0.54, 0.505)
    arrow(ax, 0.61, 0.64, 0.61, 0.58)
    arrow(ax, 0.61, 0.43, 0.61, 0.37)
    arrow(ax, 0.68, 0.295, 0.78, 0.745)
    arrow(ax, 0.68, 0.295, 0.78, 0.535)
    arrow(ax, 0.68, 0.295, 0.78, 0.325)

    ax.plot([0.515, 0.515], [0.18, 0.82], color="#d5d9de", lw=1.0)
    ax.text(0.505, 0.84, "spiking backbone", ha="right", fontsize=8.0, color="#4b5563")
    ax.text(0.955, 0.745, "MAE / Acc@0.1", fontsize=7.8, color="#4b5563", va="center")
    ax.text(0.955, 0.535, "auxiliary only", fontsize=7.8, color="#4b5563", va="center")
    ax.text(0.955, 0.325, "macro accuracy", fontsize=7.8, color="#4b5563", va="center")
    ax.text(0.02, 0.08, "Key point: health is treated as a continuous target; fault location remains a discrete classification target.", fontsize=9.0, color=P_NAVY)
    return save_fig(fig, "refined_model_architecture")


def build_figures() -> dict[str, Path]:
    base.PPT_ASSETS = PPT_ASSETS
    figs = base.make_figures()
    figs["model_refined"] = make_refined_model_figure()
    return figs


def build_ppt(fig: dict[str, Path]) -> None:
    prs = Presentation()
    prs.slide_width = emu(13.333)
    prs.slide_height = emu(7.5)

    # 1. Overview
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    textbox(slide, "NASA Bearing 轴承健康监测汇报", 0.85, 0.75, 11.6, 0.78, 36, NAVY, True)
    textbox(slide, "应用场景：高价值旋转机械在线监测，例如航天涡轮泵轴承、精密传动轴承等。", 1.05, 2.05, 11.1, 0.42, 22, NAVY, False)
    textbox(slide, "研究目标：从多通道振动信号中同时预测轴承健康度，并定位故障部位。", 1.05, 2.78, 11.1, 0.42, 22, NAVY, False)
    textbox(slide, "方法路线：NASA Bearing 数据集  →  单轴承特征构造  →  STDP-SNN 多任务模型。", 1.05, 3.51, 11.1, 0.42, 22, NAVY, False)
    textbox(slide, "任务定义：Task 1 为连续 health_score 回归；Task 2 为 normal / inner_ring / rolling_element 分类。", 1.05, 4.24, 11.1, 0.42, 22, NAVY, False)
    textbox(slide, "最终结果：Health MAE = 0.0419；Health Acc@0.1 = 87.1%；Fault Macro Acc = 99.75%。", 1.05, 4.97, 11.1, 0.42, 22, NAVY, False)

    # 2. Dataset
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "数据集与标签构造", "从 NASA Bearing 原始窗口拆分为单轴承样本，并映射到两个监督任务")
    add_picture_fit(slide, fig["dataset"], 0.35, 0.95, 6.35, 2.85)
    add_picture_fit(slide, fig["raw"], 0.35, 4.05, 6.35, 2.0)
    add_caption(slide, "数据分布图说明 good/fault 数量和拆分后的类别分布高度不均衡。", 0.52, 3.72, 6.0, 0.34)
    add_caption(slide, "原始波形图显示四个轴承通道均有连续振动响应，是特征提取的直接来源。", 0.52, 6.05, 6.0, 0.42)
    table(slide, [["项目", "当前设置"], ["正常窗口", "222,811"], ["故障窗口", "1,703"], ["样本拆分", "每行拆为 B1-B4 单轴承样本"], ["故障映射", "B3=inner_ring, B4=rolling_element"], ["未出现类", "outer_ring / cage / unknown_fault 不画空类"]], 7.15, 1.12, 2.55, 4.05, 9.2)
    add_note(slide, ["要点", "数据强不均衡，因此训练使用类别权重与平衡采样。", "Task 2 只评价实际出现的类别，避免空类别混淆矩阵误导。"], 10.08, 1.10, 2.9, 3.4)
    add_footer(slide, 2)

    # 3. Processing
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "数据处理流程", "健康度从故障时间顺序构造为连续 health_score，而不是整数阶段分类")
    add_picture_fit(slide, fig["processing"], 0.35, 1.02, 9.15, 4.0)
    add_caption(slide, "流程图说明模型输入来自单轴承特征；分布图说明多数样本接近健康状态；时间目标图说明故障段健康度从 1.0 连续下降到 0.0。", 0.55, 5.18, 8.75, 0.55)
    add_note(slide, ["标签定义", "health_score: Task 1 主目标，连续回归。", "health_stage: 辅助约束，只帮助退化顺序学习。", "fault_part: Task 2 分类目标，对应实际故障部位。"], 10.0, 1.1, 3.0, 3.75)
    add_footer(slide, 3)

    # 4. Feature evidence
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "数据可视化与特征有效性", "特征分布、故障时间线和相关性共同证明输入具有可学习结构")
    add_picture_fit(slide, fig["features"], 0.35, 0.95, 9.25, 4.85)
    add_caption(slide, "箱线图显示 good/fault 特征分布差异；时间线显示 B3/B4 特征随故障过程变化；相关性热图显示特征间存在冗余与互补。", 0.55, 5.9, 8.95, 0.55)
    add_note(slide, ["可证明的内容", "特征不是随机噪声，存在故障相关分布差异。", "连续退化过程支持 health_score 回归。", "相关性结构说明共享表征有意义。"], 10.0, 1.1, 3.0, 3.55)
    add_footer(slide, 4)

    # 5. Model
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "模型结构", "单个 STDP-SNN 主干共享表示，分别输出连续健康度和故障部位")
    add_picture_fit(slide, fig["model_refined"], 0.25, 0.92, 9.75, 4.85)
    add_caption(slide, "模型结构图显示：振动窗口先拆成单轴承特征，再编码为脉冲序列；STDP/LIF 主干学习共享状态，并通过两个任务头完成回归和分类。", 0.55, 5.88, 9.25, 0.62)
    add_note(slide, ["模型说明", "图中只展示任务数据流和输出关系。", "health_score 是主输出，使用连续值评价。", "fault_part 保持离散分类，便于定位内圈或滚动体故障。"], 10.05, 1.05, 2.9, 3.6)
    add_footer(slide, 5)

    # 6. Training
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "训练策略", "将优化重心从阶段分类切换到连续健康度回归")
    add_picture_fit(slide, fig["training"], 0.45, 1.05, 4.4, 2.35)
    add_picture_fit(slide, fig["comparison"], 5.15, 1.05, 4.35, 2.35)
    add_caption(slide, "训练权重图说明最终优化更强调 health_score 连续回归。", 0.65, 3.55, 4.05, 0.42)
    add_caption(slide, "对比柱状图说明回归导向训练显著降低 MAE 并提高 Acc@0.1。", 5.25, 3.55, 4.15, 0.42)
    table(slide, [["训练项", "作用"], ["health_score L1", "主目标，直接压低健康度误差"], ["health_stage CE", "辅助保持退化顺序"], ["fault_part CE", "保持部位分类稳定"], ["平衡采样", "缓解 normal 多数类影响"]], 0.65, 4.35, 8.55, 1.55, 9.4)
    add_note(slide, ["切换原因", "健康程度本质上是连续退化变量。", "强行分箱会让相邻阶段边界变成人为误差来源。", "分类任务保留在故障部位上。"], 10.05, 1.1, 2.9, 3.8)
    add_footer(slide, 6)

    # 7. Results
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "最终训练结果", "健康度误差稳定下降，故障部位分类保持高准确率")
    add_picture_fit(slide, fig["results"], 0.35, 1.00, 9.25, 3.85)
    add_caption(slide, "折线图证明最终阶段验证 MAE 保持在约 0.04-0.05，Acc@0.1 稳定在 0.84-0.87，故障部位 Macro Accuracy 约 0.998。", 0.55, 4.95, 8.9, 0.56)
    table(slide, [["指标", "最终可汇报值"], ["Validation Health MAE", "0.0419"], ["Validation Health Acc@0.1", "0.8709"], ["Validation Fault Macro Acc", "0.9975"]], 0.65, 5.68, 8.45, 0.95, 9.2)
    add_note(slide, ["结果解释", "MAE=0.0419 表示在 0-1 健康度上平均误差约 4.2%。", "Acc@0.1=87.1% 表示多数样本误差不超过 10%。", "故障分类已接近饱和，主瓶颈转为健康度回归。"], 10.05, 1.1, 2.9, 3.95)
    add_footer(slide, 7)

    # 8. Quality and conclusion
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "预测质量与结论", "连续健康度预测和故障部位识别均达到汇报水平")
    add_picture_fit(slide, fig["quality"], 0.35, 0.95, 9.25, 4.7)
    add_caption(slide, "散点图展示回归趋势，误差直方图展示误差集中在 0.1 以内，混淆矩阵展示实际出现的故障类别识别稳定。", 0.55, 5.78, 8.9, 0.55)
    add_note(slide, ["结论", "Task 1: 使用连续 health_score 比整数阶段分类更合理。", "Task 2: normal / inner_ring / rolling_element 分类结果稳定。", "限制: 当前验证为随机切分，严格时间外推仍需单独验证。"], 10.05, 1.1, 2.9, 4.2)
    add_footer(slide, 8)

    prs.save(OUTPUT)


def main() -> None:
    figs = build_figures()
    build_ppt(figs)
    prs = Presentation(OUTPUT)
    print(f"saved={OUTPUT}")
    print(f"slides={len(prs.slides)}")
    print(f"assets={PPT_ASSETS}")


if __name__ == "__main__":
    main()
