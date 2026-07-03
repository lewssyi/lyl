"""Create a journal-style editable C10 framework schematic.

Outputs:
- SVG/PDF/TIFF/PNG rendered with Python/matplotlib for manuscript use.
- VDX/VSDX XML packages with editable vector objects for Microsoft Visio.

This figure intentionally contains only the C10 route:
late clinical concept residual.
"""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import wrap
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7


OUT_DIR = Path("figures") / "c10_framework_nature"
FIG_W = 8.60
FIG_H = 5.00
PAGE_W = FIG_W
PAGE_H = FIG_H


PALETTE = {
    "ink": "#252A31",
    "muted": "#667085",
    "hair": "#AEB6C2",
    "panel": "#F7F9FC",
    "blue_dark": "#334E7D",
    "blue_mid": "#6D8FBE",
    "blue_soft": "#DDE7F6",
    "teal": "#3E9A9A",
    "teal_soft": "#DDF1EE",
    "rose": "#A64E73",
    "rose_soft": "#F3DDE8",
    "amber": "#B9852D",
    "amber_soft": "#F5E8C8",
    "slate_soft": "#ECEFF4",
    "green": "#4D8B57",
    "green_soft": "#E3F1DF",
}


@dataclass
class VShape:
    kind: str
    key: str
    x: float
    y: float
    w: float
    h: float
    text: str
    fill: str
    stroke: str = PALETTE["ink"]
    text_color: str = PALETTE["ink"]
    font_size: float = 6.5
    radius: float = 0.04
    dashed: bool = False
    bold: bool = False


@dataclass
class VLine:
    key: str
    start: tuple[float, float]
    end: tuple[float, float]
    text: str = ""
    color: str = PALETTE["ink"]
    dashed: bool = False
    width: float = 0.9


VISIO_SHAPES: list[VShape] = []
VISIO_LINES: list[VLine] = []


def add_shape(shape: VShape) -> None:
    VISIO_SHAPES.append(shape)


def add_line(line: VLine) -> None:
    VISIO_LINES.append(line)


def add_text(ax, x, y, text, size=7, weight="normal", color=None, ha="center", va="center", linespacing=1.12):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color or PALETTE["ink"],
        ha=ha,
        va=va,
        linespacing=linespacing,
    )


def rounded_box(
    ax,
    key,
    x,
    y,
    w,
    h,
    text,
    fill,
    stroke=None,
    size=6.5,
    weight="normal",
    text_color=None,
    radius=0.045,
    lw=0.85,
    dashed=False,
    visio=True,
):
    stroke = stroke or PALETTE["ink"]
    text_color = text_color or PALETTE["ink"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        linewidth=lw,
        edgecolor=stroke,
        facecolor=fill,
        linestyle=(0, (3, 2)) if dashed else "solid",
        zorder=2,
    )
    ax.add_patch(patch)
    add_text(ax, x + w / 2, y + h / 2, text, size=size, weight=weight, color=text_color)
    if visio:
        add_shape(
            VShape(
                "box",
                key,
                x,
                y,
                w,
                h,
                text,
                fill,
                stroke=stroke,
                text_color=text_color,
                font_size=size,
                radius=radius,
                dashed=dashed,
                bold=weight in {"bold", "semibold"},
            )
        )
    return patch


def arrow(ax, key, start, end, color=None, text="", dashed=False, lw=0.9, rad=0.0, ms=7):
    color = color or PALETTE["ink"]
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=(0, (3, 2)) if dashed else "solid",
        zorder=3,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        add_text(ax, mx, my + 0.055, text, size=5.4, color=color)
    add_line(VLine(key, start, end, text=text, color=color, dashed=dashed, width=lw))
    return patch


def draw_signal(ax, x, y, w, h, color, seed=0, label=""):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 160)
    freq = rng.uniform(4.0, 8.0)
    wave = 0.34 * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    wave += 0.12 * np.sin(2 * np.pi * (freq * 0.35) * t)
    wave += 0.04 * rng.normal(size=t.size)
    xs = x + w * t
    ys = y + h / 2 + h * wave
    ax.plot(xs, ys, color=color, lw=0.75, zorder=4)
    add_text(ax, x - 0.045, y + h / 2, label, size=5.6, weight="bold", color=color, ha="right")


def draw_plus(ax, x, y, r=0.085):
    circ = Circle((x, y), r, facecolor="white", edgecolor=PALETTE["rose"], lw=1.0, zorder=5)
    ax.add_patch(circ)
    ax.plot([x - r * 0.45, x + r * 0.45], [y, y], color=PALETTE["rose"], lw=1.0, zorder=6)
    ax.plot([x, x], [y - r * 0.45, y + r * 0.45], color=PALETTE["rose"], lw=1.0, zorder=6)
    add_shape(VShape("ellipse", "residual_add", x - r, y - r, 2 * r, 2 * r, "+", "#FFFFFF", stroke=PALETTE["rose"], font_size=8, bold=True))


def draw_brace(ax, x0, x1, y, height, color, text):
    verts = [
        (x0, y),
        (x0 + 0.10, y),
        (x0 + 0.10, y + height),
        ((x0 + x1) / 2 - 0.08, y + height),
        ((x0 + x1) / 2, y + height + 0.06),
        ((x0 + x1) / 2 + 0.08, y + height),
        (x1 - 0.10, y + height),
        (x1 - 0.10, y),
        (x1, y),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), fill=False, lw=0.8, edgecolor=color, zorder=3))
    add_text(ax, (x0 + x1) / 2, y + height + 0.16, text, size=5.7, color=color)


def build_matplotlib_figure():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    add_text(ax, 0.18, FIG_H - 0.20, "a", size=8, weight="bold", ha="left")
    add_text(
        ax,
        0.38,
        FIG_H - 0.20,
        "C10 late-concept residual framework",
        size=9,
        weight="bold",
        ha="left",
    )
    add_text(
        ax,
        0.38,
        FIG_H - 0.41,
        "Multimodal EEG/EOG/EMG sleep staging with a separated discriminative path, weak clinical concept residual and post-hoc XAI audit.",
        size=6.4,
        color=PALETTE["muted"],
        ha="left",
    )
    add_shape(VShape("text", "panel_label", 0.10, FIG_H - 0.32, 0.15, 0.16, "a", "#FFFFFF", stroke="#FFFFFF", font_size=8, bold=True))
    add_shape(VShape("text", "title", 0.36, FIG_H - 0.31, 3.0, 0.16, "C10 late-concept residual framework", "#FFFFFF", stroke="#FFFFFF", font_size=9, bold=True))
    add_shape(VShape("text", "subtitle", 0.36, FIG_H - 0.51, 5.9, 0.18, "Multimodal EEG/EOG/EMG sleep staging with a separated discriminative path, weak clinical concept residual and post-hoc XAI audit.", "#FFFFFF", stroke="#FFFFFF", font_size=6.4, text_color=PALETTE["muted"]))

    # Input protocol panel.
    rounded_box(
        ax,
        "recording_protocol",
        0.30,
        3.15,
        1.22,
        0.92,
        "Recording-level protocol\n.npz per PSG night\n30-s epochs; file split",
        PALETTE["blue_soft"],
        stroke=PALETTE["blue_mid"],
        size=6.0,
        weight="bold",
    )
    for idx, (lab, col) in enumerate((("EEG", PALETTE["blue_dark"]), ("EOG", PALETTE["teal"]), ("EMG", PALETTE["amber"]))):
        draw_signal(ax, 0.46, 2.90 - idx * 0.19, 0.88, 0.11, col, seed=idx + 2, label=lab)
    add_shape(VShape("box", "signal_stack", 0.36, 2.36, 1.04, 0.66, "EEG/EOG/EMG traces", "#FFFFFF", stroke=PALETTE["hair"], font_size=5.4))

    rounded_box(
        ax,
        "preprocess",
        1.80,
        3.12,
        1.23,
        0.98,
        "Preprocess + quality\nclinical filters\nrobust z-score\nquality score Q_t",
        PALETTE["teal_soft"],
        stroke=PALETTE["teal"],
        size=6.0,
        weight="bold",
    )

    # Main path.
    rounded_box(
        ax,
        "main_path",
        3.36,
        3.02,
        2.38,
        1.18,
        "Main discriminative path\nraw/segment encoders -> fusion\nsequence TCN -> z_main",
        PALETTE["slate_soft"],
        stroke=PALETTE["blue_dark"],
        size=6.2,
        weight="bold",
        radius=0.06,
        lw=1.0,
    )
    inner_y = 3.14
    inner = [
        ("Raw\nencoder", 3.50, 0.43),
        ("Segment\nencoder", 4.02, 0.48),
        ("Fusion", 4.62, 0.36),
        ("TCN", 5.08, 0.32),
    ]
    for text, x, w in inner:
        rounded_box(
            ax,
            f"inner_{text.split()[0].lower()}",
            x,
            inner_y,
            w,
            0.32,
            text,
            "#FFFFFF",
            stroke=PALETTE["hair"],
            size=5.3,
            radius=0.03,
        )
    for i in range(len(inner) - 1):
        arrow(ax, f"inner_arrow_{i}", (inner[i][1] + inner[i][2], inner_y + 0.16), (inner[i + 1][1], inner_y + 0.16), color=PALETTE["hair"], lw=0.55, ms=5)

    rounded_box(
        ax,
        "main_logits",
        6.08,
        3.23,
        0.62,
        0.54,
        "z_main",
        "#FFFFFF",
        stroke=PALETTE["blue_dark"],
        size=6.4,
        weight="bold",
    )

    # C10 branch.
    rounded_box(
        ax,
        "c10_branch",
        3.30,
        1.78,
        2.48,
        0.90,
        "C10 late clinical concept residual\nsegment evidence -> 9 weak concepts\nstop-gradient summary; prior-initialized W_c",
        PALETTE["rose_soft"],
        stroke=PALETTE["rose"],
        size=6.0,
        weight="bold",
        radius=0.06,
        lw=1.0,
    )

    concept_y = 1.40
    concept_specs = [
        ("delta", 3.34, 0.32, PALETTE["blue_soft"], PALETTE["blue_mid"]),
        ("theta", 3.70, 0.35, PALETTE["blue_soft"], PALETTE["blue_mid"]),
        ("alpha", 4.09, 0.35, PALETTE["blue_soft"], PALETTE["blue_mid"]),
        ("sigma", 4.48, 0.36, PALETTE["blue_soft"], PALETTE["blue_mid"]),
        ("beta", 4.88, 0.31, PALETTE["blue_soft"], PALETTE["blue_mid"]),
        ("slow eye", 5.25, 0.42, PALETTE["teal_soft"], PALETTE["teal"]),
        ("rapid eye", 5.72, 0.46, PALETTE["teal_soft"], PALETTE["teal"]),
        ("low tone", 6.23, 0.43, PALETTE["amber_soft"], PALETTE["amber"]),
        ("active tone", 6.70, 0.52, PALETTE["amber_soft"], PALETTE["amber"]),
    ]
    for name, x, w, fill, edge in concept_specs:
        rounded_box(ax, f"concept_{name.replace(' ', '_')}", x, concept_y, w, 0.20, name, fill, stroke=edge, size=4.9, radius=0.05, lw=0.6)
    draw_brace(ax, 3.34, 5.19, 1.17, 0.06, PALETTE["blue_mid"], "EEG")
    draw_brace(ax, 5.25, 6.18, 1.17, 0.06, PALETTE["teal"], "EOG")
    draw_brace(ax, 6.23, 7.22, 1.17, 0.06, PALETTE["amber"], "EMG")

    rounded_box(
        ax,
        "residual_logits",
        6.12,
        1.98,
        0.84,
        0.48,
        "z_c = 0.05 W_c h_c",
        "#FFFFFF",
        stroke=PALETTE["rose"],
        size=5.7,
        weight="bold",
    )
    draw_plus(ax, 6.95, 3.50, r=0.075)

    rounded_box(
        ax,
        "probabilities",
        7.16,
        3.17,
        0.72,
        0.64,
        "softmax\np(stage_t)",
        PALETTE["green_soft"],
        stroke=PALETTE["green"],
        size=5.8,
        weight="bold",
    )

    rounded_box(
        ax,
        "decoder",
        7.42,
        2.18,
        0.98,
        0.74,
        "Sequence decoding\nN1 calibration\nsmoothing + HMM",
        PALETTE["amber_soft"],
        stroke=PALETTE["amber"],
        size=5.9,
        weight="bold",
    )
    rounded_box(
        ax,
        "hypnogram",
        7.48,
        0.64,
        0.95,
        0.44,
        "final\nhypnogram",
        "#FFFFFF",
        stroke=PALETTE["ink"],
        size=5.8,
        weight="bold",
    )
    # Small hypnogram glyph.
    xs = np.linspace(7.54, 8.36, 9)
    ys = np.array([0.83, 0.76, 0.76, 0.69, 0.63, 0.69, 0.76, 0.70, 0.77])
    ax.step(xs, ys, where="mid", color=PALETTE["blue_dark"], lw=1.0, zorder=4)

    # XAI audit strip.
    rounded_box(
        ax,
        "xai_audit",
        0.34,
        0.28,
        6.92,
        0.74,
        "",
        PALETTE["panel"],
        stroke=PALETTE["hair"],
        size=5.9,
        radius=0.05,
    )
    add_text(ax, 0.56, 0.86, "Post-hoc clinical audit", size=5.9, weight="bold", color=PALETTE["muted"], ha="left")
    add_shape(VShape("text", "xai_audit_title", 0.50, 0.78, 1.2, 0.15, "Post-hoc clinical audit", "#FFFFFF", stroke="#FFFFFF", font_size=5.9, text_color=PALETTE["muted"], bold=True))
    xai_steps = [
        ("xai_lime", 1.18, "physiology-aware\nLIME"),
        ("xai_ridge", 2.38, "Ridge local\nsurrogate"),
        ("xai_dccs", 3.60, "DCCS +\nconcept-DCCS"),
        ("xai_transition", 5.02, "transition_shift\ndashboards"),
    ]
    for key, x, text in xai_steps:
        rounded_box(ax, key, x, 0.43, 0.92 if key != "xai_transition" else 1.18, 0.30, text, "#FFFFFF", stroke=PALETTE["hair"], size=5.2, radius=0.035, lw=0.6)
    arrow(ax, "xai_lime_to_ridge", (2.10, 0.58), (2.38, 0.58), color=PALETTE["hair"], lw=0.55, ms=5)
    arrow(ax, "xai_ridge_to_dccs", (3.30, 0.58), (3.60, 0.58), color=PALETTE["hair"], lw=0.55, ms=5)
    arrow(ax, "xai_dccs_to_transition", (4.52, 0.58), (5.02, 0.58), color=PALETTE["hair"], lw=0.55, ms=5)

    # Connectors.
    arrow(ax, "data_to_preprocess", (1.52, 3.58), (1.80, 3.58), color=PALETTE["blue_dark"], lw=0.85)
    arrow(ax, "preprocess_to_main", (3.03, 3.60), (3.36, 3.60), color=PALETTE["blue_dark"], lw=0.85)
    arrow(ax, "preprocess_to_c10", (2.42, 3.12), (3.31, 2.30), color=PALETTE["rose"], lw=0.85, rad=-0.08, text="segment features")
    arrow(ax, "main_to_logits", (5.74, 3.50), (6.08, 3.50), color=PALETTE["blue_dark"], lw=0.85)
    arrow(ax, "logits_to_plus", (6.70, 3.50), (6.87, 3.50), color=PALETTE["blue_dark"], lw=0.85)
    arrow(ax, "c10_to_residual", (5.78, 2.22), (6.12, 2.22), color=PALETTE["rose"], lw=0.85)
    arrow(ax, "residual_to_plus", (6.96, 2.46), (6.95, 3.42), color=PALETTE["rose"], lw=0.85, rad=0.08)
    arrow(ax, "plus_to_prob", (7.03, 3.50), (7.16, 3.50), color=PALETTE["green"], lw=0.85)
    arrow(ax, "prob_to_decoder", (7.52, 3.17), (7.78, 2.92), color=PALETTE["amber"], lw=0.85, rad=-0.08)
    arrow(ax, "decoder_to_hypnogram", (7.91, 2.18), (7.95, 1.08), color=PALETTE["amber"], lw=0.85)
    arrow(ax, "quality_to_decoder", (2.55, 3.12), (7.42, 2.58), color=PALETTE["muted"], dashed=True, lw=0.55, text="Q_t")
    arrow(ax, "hypnogram_to_xai", (7.48, 0.82), (7.26, 0.62), color=PALETTE["muted"], dashed=True, lw=0.65)

    # Footer integrity note.
    add_text(
        ax,
        0.36,
        0.13,
        "Abbrev.: TCN, temporal convolutional network; DCCS, dynamic channel contribution score. C10 route shown only.",
        size=5.3,
        color=PALETTE["muted"],
        ha="left",
    )
    add_shape(VShape("text", "footnote", 0.34, 0.05, 5.8, 0.18, "Abbrev.: TCN, temporal convolutional network; DCCS, dynamic channel contribution score. C10 route shown only.", "#FFFFFF", stroke="#FFFFFF", font_size=5.3, text_color=PALETTE["muted"]))
    return fig


def save_matplotlib_exports() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_matplotlib_figure()
    base = OUT_DIR / "c10_framework_nature"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def visio_y(y: float) -> float:
    return PAGE_H - y


def wrapped_text(text: str, width: int = 28) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        lines.extend(wrap(raw, width=width, break_long_words=False) or [""])
    return "\n".join(lines)


def cell(name: str, value: str | float | int) -> str:
    return f'<Cell N="{name}" V="{value}"/>'


def visio_shape_xml(shape_id: int, s: VShape) -> str:
    pin_x = s.x + s.w / 2
    pin_y = PAGE_H - s.y - s.h / 2
    text = escape(wrapped_text(s.text, max(8, int(s.w * 16))))
    line_pattern = "2" if s.dashed else "1"
    fill_pattern = "0" if s.kind == "text" else "1"
    no_line = "1" if s.kind == "text" else "0"
    bold = "1" if s.bold else "0"
    geom = f"""
        <Section N="Geometry" IX="0">
          <Cell N="NoFill" V="{1 if s.kind == 'text' else 0}"/>
          <Cell N="NoLine" V="{no_line}"/>
          <Row T="MoveTo" IX="1"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row>
          <Row T="LineTo" IX="2"><Cell N="X" V="{s.w:.4f}"/><Cell N="Y" V="0"/></Row>
          <Row T="LineTo" IX="3"><Cell N="X" V="{s.w:.4f}"/><Cell N="Y" V="{s.h:.4f}"/></Row>
          <Row T="LineTo" IX="4"><Cell N="X" V="0"/><Cell N="Y" V="{s.h:.4f}"/></Row>
          <Row T="LineTo" IX="5"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row>
        </Section>"""
    if s.kind == "ellipse":
        geom = f"""
        <Section N="Geometry" IX="0">
          <Cell N="NoFill" V="0"/>
          <Cell N="NoLine" V="0"/>
          <Row T="Ellipse" IX="1">
            <Cell N="X" V="{s.w / 2:.4f}"/><Cell N="Y" V="{s.h / 2:.4f}"/>
            <Cell N="A" V="{s.w:.4f}"/><Cell N="B" V="{s.h / 2:.4f}"/>
            <Cell N="C" V="{s.w / 2:.4f}"/><Cell N="D" V="{s.h:.4f}"/>
          </Row>
        </Section>"""
    return f"""
      <Shape ID="{shape_id}" NameU="{escape(s.key)}" Type="Shape" LineStyle="0" FillStyle="0" TextStyle="0">
        {cell("PinX", f"{pin_x:.4f}")}{cell("PinY", f"{pin_y:.4f}")}
        {cell("Width", f"{s.w:.4f}")}{cell("Height", f"{s.h:.4f}")}
        {cell("LocPinX", f"{s.w / 2:.4f}")}{cell("LocPinY", f"{s.h / 2:.4f}")}
        {cell("FillForegnd", s.fill)}{cell("FillPattern", fill_pattern)}
        {cell("LineColor", s.stroke)}{cell("LinePattern", line_pattern)}{cell("LineWeight", "0.0100")}
        {cell("VerticalAlign", "1")}
        <Section N="Character"><Row IX="0"><Cell N="Font" V="0"/><Cell N="Size" V="{s.font_size / 72:.4f}"/><Cell N="Color" V="{s.text_color}"/><Cell N="Style" V="{bold}"/></Row></Section>
        <Section N="Paragraph"><Row IX="0"><Cell N="HorzAlign" V="1"/></Row></Section>
        <Text>{text}</Text>
        {geom}
      </Shape>"""


def visio_line_xml(shape_id: int, l: VLine) -> str:
    x1, y1 = l.start
    x2, y2 = l.end
    vy1 = visio_y(y1)
    vy2 = visio_y(y2)
    dx = x2 - x1
    dy = vy2 - vy1
    length = max(math.hypot(dx, dy), 0.001)
    angle = math.atan2(dy, dx)
    pin_x = (x1 + x2) / 2
    pin_y = (vy1 + vy2) / 2
    line_pattern = "2" if l.dashed else "1"
    text = escape(l.text)
    return f"""
      <Shape ID="{shape_id}" NameU="{escape(l.key)}" Type="Shape" LineStyle="0" FillStyle="0" TextStyle="0">
        {cell("PinX", f"{pin_x:.4f}")}{cell("PinY", f"{pin_y:.4f}")}
        {cell("Width", f"{length:.4f}")}{cell("Height", "0.0100")}
        {cell("LocPinX", f"{length / 2:.4f}")}{cell("LocPinY", "0")}
        {cell("Angle", f"{angle:.8f}")}{cell("FillPattern", "0")}
        {cell("LineColor", l.color)}{cell("LinePattern", line_pattern)}{cell("LineWeight", f"{l.width / 72:.4f}")}{cell("EndArrow", "4")}
        <Section N="Character"><Row IX="0"><Cell N="Font" V="0"/><Cell N="Size" V="{5.4 / 72:.4f}"/><Cell N="Color" V="{l.color}"/></Row></Section>
        <Section N="Paragraph"><Row IX="0"><Cell N="HorzAlign" V="1"/></Row></Section>
        <Text>{text}</Text>
        <Section N="Geometry" IX="0"><Cell N="NoFill" V="1"/><Cell N="NoLine" V="0"/>
          <Row T="MoveTo" IX="1"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row>
          <Row T="LineTo" IX="2"><Cell N="X" V="{length:.4f}"/><Cell N="Y" V="0"/></Row>
        </Section>
      </Shape>"""


def page_contents_xml() -> str:
    parts = ["<Shapes>"]
    shape_id = 1
    for line in VISIO_LINES:
        parts.append(visio_line_xml(shape_id, line))
        shape_id += 1
    for shape in VISIO_SHAPES:
        parts.append(visio_shape_xml(shape_id, shape))
        shape_id += 1
    parts.append("</Shapes>")
    return "\n".join(parts)


def save_vdx(path: Path) -> None:
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<VisioDocument xmlns="urn:schemas-microsoft-com:office:visio" version="11.0" xml:space="preserve">
  <DocumentProperties><Title>C10 late-concept residual framework</Title><Creator>Codex</Creator></DocumentProperties>
  <FaceNames><FaceName ID="0" NameU="Arial" Name="Arial"/></FaceNames>
  <StyleSheets><StyleSheet ID="0" NameU="No Style" Name="No Style" Type="Style"/></StyleSheets>
  <Pages>
    <Page ID="0" NameU="C10 framework" Name="C10 framework">
      <PageSheet>
        <Cell N="PageWidth" V="{PAGE_W:.4f}"/>
        <Cell N="PageHeight" V="{PAGE_H:.4f}"/>
        <Cell N="DrawingScale" V="1"/>
        <Cell N="PageScale" V="1"/>
      </PageSheet>
      {page_contents_xml()}
    </Page>
  </Pages>
</VisioDocument>
"""
    path.write_text(xml, encoding="utf-8")


def save_vsdx(path: Path) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/visio/document.xml" ContentType="application/vnd.ms-visio.drawing.main+xml"/>
  <Override PartName="/visio/pages/pages.xml" ContentType="application/vnd.ms-visio.pages+xml"/>
  <Override PartName="/visio/pages/page1.xml" ContentType="application/vnd.ms-visio.page+xml"/>
</Types>
"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.microsoft.com/visio/2010/relationships/document" Target="visio/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""
    doc_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<VisioDocument xmlns="http://schemas.microsoft.com/office/visio/2012/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xml:space="preserve">
  <DocumentSettings/>
  <Colors/>
  <FaceNames><FaceName ID="0" NameU="Arial" Name="Arial"/></FaceNames>
  <StyleSheets><StyleSheet ID="0" NameU="No Style" Name="No Style" Type="Style"/></StyleSheets>
  <DocumentSheet><Cell N="OutputFormat" V="0"/></DocumentSheet>
</VisioDocument>
"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.microsoft.com/visio/2010/relationships/pages" Target="pages/pages.xml"/>
</Relationships>
"""
    pages_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Pages xmlns="http://schemas.microsoft.com/office/visio/2012/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xml:space="preserve">
  <Page ID="0" NameU="C10 framework" Name="C10 framework" ViewScale="1" ViewCenterX="{PAGE_W / 2:.4f}" ViewCenterY="{PAGE_H / 2:.4f}">
    <PageSheet>
      <Cell N="PageWidth" V="{PAGE_W:.4f}"/>
      <Cell N="PageHeight" V="{PAGE_H:.4f}"/>
      <Cell N="DrawingScale" V="1"/>
      <Cell N="PageScale" V="1"/>
    </PageSheet>
    <Rel r:id="rId1"/>
  </Page>
</Pages>
"""
    pages_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.microsoft.com/visio/2010/relationships/page" Target="page1.xml"/>
</Relationships>
"""
    page_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<PageContents xmlns="http://schemas.microsoft.com/office/visio/2012/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xml:space="preserve">
  {page_contents_xml()}
</PageContents>
"""
    core_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>C10 late-concept residual framework</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""
    app_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Visio</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
</Properties>
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("docProps/core.xml", core_xml)
        archive.writestr("docProps/app.xml", app_xml)
        archive.writestr("visio/document.xml", doc_xml)
        archive.writestr("visio/_rels/document.xml.rels", doc_rels)
        archive.writestr("visio/pages/pages.xml", pages_xml)
        archive.writestr("visio/pages/_rels/pages.xml.rels", pages_rels)
        archive.writestr("visio/pages/page1.xml", page_xml)


def write_notes() -> None:
    notes = """# C10 framework figure QA notes

Core conclusion: C10 adds a weak, late clinical-concept residual to a multimodal sequence sleep-staging trunk, while keeping sequence decoding and post-hoc XAI audit separate and reviewable.

Evidence chain:
- Recording-level input and preprocessing establish leakage-safe multimodal data handling.
- Main discriminative path carries raw/segment evidence through sequence modelling.
- C10 branch derives nine clinical concepts from segment evidence and adds a scaled residual logit correction.
- Decoder uses probability calibration, N1-aware smoothing, HMM/Viterbi and quality-aware adaptive Viterbi.
- XAI audit uses physiology-aware LIME, DCCS, concept-DCCS and transition summaries.

Archetype: schematic-led composite.
Backend: Python only for rendering/export/QA.
Export contract: SVG/PDF/TIFF/PNG plus editable Visio XML drawings (VDX/VSDX).
Review risks addressed: only the C10 late-residual route is shown; Q_t is shown as postprocessing-only modulation.
"""
    (OUT_DIR / "c10_framework_nature_qa_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Build once to populate the shared Visio object list.
    save_matplotlib_exports()
    save_vdx(OUT_DIR / "c10_framework_nature.vdx")
    save_vsdx(OUT_DIR / "c10_framework_nature.vsdx")
    write_notes()
    print(f"Wrote C10 Nature-style framework assets to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
