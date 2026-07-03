"""Generate an editable Visio-style C10 framework diagram.

The generated VSDX/VDX files use native vector shapes and text, not a raster
image embed. The SVG/PNG exports are provided for quick review and manuscript
drafting.
"""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import wrap
from xml.sax.saxutils import escape


PAGE_W = 14.0
PAGE_H = 8.0
OUT_DIR = Path("figures") / "c10_framework"

TEXT_COLOR = "#1F2933"
LINE_COLOR = "#2E3440"
MUTED_LINE = "#6B7280"
BG = "#FFFFFF"
LIGHT_BG = "#F7F9FB"

COLORS = {
    "input": "#D8EEF8",
    "preprocess": "#DDF2EA",
    "encoder": "#FFF1CC",
    "sequence": "#E6ECF7",
    "residual": "#F2DCEB",
    "logits": "#EFE7F6",
    "decoder": "#F8E1D7",
    "xai": "#E9F4F8",
    "note": "#F3F4F6",
}


@dataclass
class Box:
    key: str
    x: float
    y: float
    w: float
    h: float
    title: str
    body: str
    fill: str
    stroke: str = LINE_COLOR
    font_size: int = 8
    title_size: int = 9
    radius: float = 0.08
    dashed: bool = False


@dataclass
class Arrow:
    start: tuple[float, float]
    end: tuple[float, float]
    label: str = ""
    color: str = LINE_COLOR
    dashed: bool = False
    width: float = 1.4


@dataclass
class TextItem:
    x: float
    y: float
    w: float
    h: float
    text: str
    font_size: int = 8
    bold: bool = False
    color: str = TEXT_COLOR


BOXES = [
    Box(
        "data",
        0.45,
        1.05,
        1.75,
        1.05,
        "Recording-level input",
        ".npz PSG recordings\n30-s EEG/EOG/EMG epochs\nLabels: W, N1, N2, N3, REM\nSubject/file-level split",
        COLORS["input"],
    ),
    Box(
        "prep",
        2.45,
        1.05,
        1.85,
        1.05,
        "Preprocessing + QC",
        "Clinical channel filters\nPer-recording robust z-score\nEpoch quality score Q_t\nNo amplitude repair/clipping",
        COLORS["preprocess"],
    ),
    Box(
        "enc",
        4.55,
        1.05,
        2.05,
        1.05,
        "Dual evidence encoding",
        "Raw waveform stream\nSemantic/spectral segment stream\n30 dynamic segments per epoch",
        COLORS["encoder"],
    ),
    Box(
        "seq",
        6.85,
        1.05,
        2.05,
        1.05,
        "Sequence backbone",
        "Epoch feature fusion\nTemporal TCN decoder\nMixStyle + stage-CORAL\nTransition-boundary auxiliary target",
        COLORS["sequence"],
    ),
    Box(
        "logits",
        9.25,
        1.05,
        1.35,
        1.05,
        "Main logits",
        "Stage logits from temporal features\nResidual logit addition",
        COLORS["logits"],
    ),
    Box(
        "decode",
        10.95,
        1.05,
        2.55,
        1.05,
        "Prediction + sequence decoder",
        "Softmax probabilities\nTemperature + N1 calibration\nN1-protected smoothing\nClinical HMM/Viterbi\nQuality-aware adaptive HMM",
        COLORS["decoder"],
    ),
    Box(
        "c10",
        6.95,
        2.85,
        3.65,
        1.15,
        "C10: late clinical concept residual",
        "Segment features -> 9 clinical concepts\nStop-gradient concept summary\nPrior-initialized trainable residual logits\nScale = 0.05; add to main logits",
        COLORS["residual"],
        stroke="#9B4F82",
    ),
    Box(
        "xai",
        0.75,
        5.05,
        12.5,
        1.15,
        "Post-hoc explanation and clinical audit",
        "Physiology-aware multichannel LIME: channel x segment perturbations -> Ridge local surrogate\nRaw/calibrated DCCS, concept-DCCS, prior alignment, transition_shift and concept_shift dashboards",
        COLORS["xai"],
        stroke="#2C7DA0",
    ),
]

ARROWS = [
    Arrow((2.20, 1.575), (2.45, 1.575)),
    Arrow((4.30, 1.575), (4.55, 1.575)),
    Arrow((6.60, 1.575), (6.85, 1.575)),
    Arrow((8.90, 1.575), (9.25, 1.575)),
    Arrow((10.60, 1.575), (10.95, 1.575)),
    Arrow((5.60, 2.10), (7.55, 2.85), "segment evidence", color="#9B4F82"),
    Arrow((10.60, 3.42), (9.92, 2.10), "residual logits", color="#9B4F82"),
    Arrow((3.38, 2.10), (11.95, 5.05), "Q_t used only by adaptive Viterbi", color=MUTED_LINE, dashed=True, width=1.1),
    Arrow((9.92, 2.10), (7.00, 5.05), "model + signals", color="#2C7DA0", dashed=True, width=1.1),
    Arrow((12.15, 2.10), (10.55, 5.05), "decoded transitions", color="#2C7DA0", dashed=True, width=1.1),
]

TEXT_ITEMS = [
    TextItem(
        0.45,
        0.20,
        13.10,
        0.35,
        "C10 framework for interpretable multimodal sleep staging",
        font_size=15,
        bold=True,
    ),
    TextItem(
        0.47,
        0.60,
        13.05,
        0.25,
        "Main line: C10 late clinical concept residual for sequence-level sleep staging.",
        font_size=8,
        color="#4B5563",
    ),
    TextItem(
        0.80,
        6.55,
        12.40,
        0.55,
        "Reportable artifacts: classification report, confusion matrix, split predictions, stage-level diagnosis dashboards, and transition-centered XAI dashboards.",
        font_size=8,
        color="#4B5563",
    ),
]


def svg_text_lines(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line:
            lines.append("")
            continue
        lines.extend(wrap(raw_line, width=max_chars, break_long_words=False) or [""])
    return lines


def px(value: float) -> float:
    return value * 96.0


def render_svg(path: Path) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}in" height="{PAGE_H}in" '
        f'viewBox="0 0 {px(PAGE_W):.0f} {px(PAGE_H):.0f}">',
        f'<rect x="0" y="0" width="{px(PAGE_W):.0f}" height="{px(PAGE_H):.0f}" fill="{BG}"/>',
        '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L9,4.5 L0,9 z" fill="#2E3440"/></marker>'
        '<marker id="arrowPurple" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L9,4.5 L0,9 z" fill="#9B4F82"/></marker>'
        '<marker id="arrowBlue" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L9,4.5 L0,9 z" fill="#2C7DA0"/></marker></defs>',
    ]
    for item in TEXT_ITEMS:
        weight = "700" if item.bold else "400"
        x = px(item.x + item.w / 2)
        y = px(item.y + 0.20)
        anchor = "middle"
        for line_idx, line in enumerate(svg_text_lines(item.text, max(20, int(item.w * 16)))):
            parts.append(
                f'<text x="{x:.1f}" y="{y + line_idx * (item.font_size + 4):.1f}" '
                f'font-family="Arial, Calibri, sans-serif" font-size="{item.font_size}" '
                f'font-weight="{weight}" fill="{item.color}" text-anchor="{anchor}">{escape(line)}</text>'
            )

    for arrow in ARROWS:
        x1, y1 = px(arrow.start[0]), px(arrow.start[1])
        x2, y2 = px(arrow.end[0]), px(arrow.end[1])
        marker = "arrowPurple" if arrow.color == "#9B4F82" else "arrowBlue" if arrow.color == "#2C7DA0" else "arrow"
        dash = ' stroke-dasharray="6 4"' if arrow.dashed else ""
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{arrow.color}" stroke-width="{arrow.width}"{dash} marker-end="url(#{marker})"/>'
        )
        if arrow.label:
            lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 - 6
            parts.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" font-family="Arial, Calibri, sans-serif" '
                f'font-size="7" fill="{arrow.color}" text-anchor="middle">{escape(arrow.label)}</text>'
            )

    for box in BOXES:
        parts.append(
            f'<rect x="{px(box.x):.1f}" y="{px(box.y):.1f}" width="{px(box.w):.1f}" height="{px(box.h):.1f}" '
            f'rx="{px(box.radius):.1f}" ry="{px(box.radius):.1f}" fill="{box.fill}" '
            f'stroke="{box.stroke}" stroke-width="1.2"/>'
        )
        title_y = px(box.y) + 17
        parts.append(
            f'<text x="{px(box.x + box.w / 2):.1f}" y="{title_y:.1f}" '
            f'font-family="Arial, Calibri, sans-serif" font-size="{box.title_size}" '
            f'font-weight="700" fill="{TEXT_COLOR}" text-anchor="middle">{escape(box.title)}</text>'
        )
        body_lines = svg_text_lines(box.body, max(18, int(box.w * 17)))
        start_y = title_y + 16
        for idx, line in enumerate(body_lines):
            parts.append(
                f'<text x="{px(box.x + box.w / 2):.1f}" y="{start_y + idx * (box.font_size + 2):.1f}" '
                f'font-family="Arial, Calibri, sans-serif" font-size="{box.font_size}" '
                f'fill="{TEXT_COLOR}" text-anchor="middle">{escape(line)}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def shape_text(title: str, body: str) -> str:
    return f"{title}\n{body}" if body else title


def visio_y(top_y: float) -> float:
    return PAGE_H - top_y


def v_shape_xml(shape_id: int, box: Box, namespace: str = "vsdx") -> str:
    pin_x = box.x + box.w / 2
    pin_y = PAGE_H - box.y - box.h / 2
    text = escape(shape_text(box.title, box.body))
    char_size = box.font_size / 72.0
    line_pattern = "2" if box.dashed else "1"
    return f"""
      <Shape ID="{shape_id}" NameU="{escape(box.key)}" Type="Shape" LineStyle="0" FillStyle="0" TextStyle="0">
        <Cell N="PinX" V="{pin_x:.4f}"/>
        <Cell N="PinY" V="{pin_y:.4f}"/>
        <Cell N="Width" V="{box.w:.4f}"/>
        <Cell N="Height" V="{box.h:.4f}"/>
        <Cell N="LocPinX" V="{box.w / 2:.4f}"/>
        <Cell N="LocPinY" V="{box.h / 2:.4f}"/>
        <Cell N="FillForegnd" V="{box.fill}"/>
        <Cell N="LineColor" V="{box.stroke}"/>
        <Cell N="LinePattern" V="{line_pattern}"/>
        <Cell N="LineWeight" V="0.0140"/>
        <Cell N="VerticalAlign" V="1"/>
        <Section N="Character">
          <Row IX="0"><Cell N="Font" V="0"/><Cell N="Size" V="{char_size:.4f}"/><Cell N="Color" V="{TEXT_COLOR}"/></Row>
        </Section>
        <Section N="Paragraph">
          <Row IX="0"><Cell N="HorzAlign" V="1"/></Row>
        </Section>
        <Text>{text}</Text>
        <Section N="Geometry" IX="0">
          <Cell N="NoFill" V="0"/>
          <Cell N="NoLine" V="0"/>
          <Row T="MoveTo" IX="1"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row>
          <Row T="LineTo" IX="2"><Cell N="X" V="{box.w:.4f}"/><Cell N="Y" V="0"/></Row>
          <Row T="LineTo" IX="3"><Cell N="X" V="{box.w:.4f}"/><Cell N="Y" V="{box.h:.4f}"/></Row>
          <Row T="LineTo" IX="4"><Cell N="X" V="0"/><Cell N="Y" V="{box.h:.4f}"/></Row>
          <Row T="LineTo" IX="5"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row>
        </Section>
      </Shape>"""


def v_text_xml(shape_id: int, item: TextItem) -> str:
    pin_x = item.x + item.w / 2
    pin_y = PAGE_H - item.y - item.h / 2
    text = escape(item.text)
    char_size = item.font_size / 72.0
    weight = "1" if item.bold else "0"
    return f"""
      <Shape ID="{shape_id}" NameU="text_{shape_id}" Type="Shape" LineStyle="0" FillStyle="0" TextStyle="0">
        <Cell N="PinX" V="{pin_x:.4f}"/>
        <Cell N="PinY" V="{pin_y:.4f}"/>
        <Cell N="Width" V="{item.w:.4f}"/>
        <Cell N="Height" V="{item.h:.4f}"/>
        <Cell N="LocPinX" V="{item.w / 2:.4f}"/>
        <Cell N="LocPinY" V="{item.h / 2:.4f}"/>
        <Cell N="FillPattern" V="0"/>
        <Cell N="LinePattern" V="0"/>
        <Cell N="VerticalAlign" V="1"/>
        <Section N="Character">
          <Row IX="0"><Cell N="Font" V="0"/><Cell N="Size" V="{char_size:.4f}"/><Cell N="Color" V="{item.color}"/><Cell N="Style" V="{weight}"/></Row>
        </Section>
        <Section N="Paragraph"><Row IX="0"><Cell N="HorzAlign" V="1"/></Row></Section>
        <Text>{text}</Text>
        <Section N="Geometry" IX="0">
          <Cell N="NoFill" V="1"/><Cell N="NoLine" V="1"/>
          <Row T="MoveTo" IX="1"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row>
          <Row T="LineTo" IX="2"><Cell N="X" V="{item.w:.4f}"/><Cell N="Y" V="0"/></Row>
          <Row T="LineTo" IX="3"><Cell N="X" V="{item.w:.4f}"/><Cell N="Y" V="{item.h:.4f}"/></Row>
          <Row T="LineTo" IX="4"><Cell N="X" V="0"/><Cell N="Y" V="{item.h:.4f}"/></Row>
          <Row T="LineTo" IX="5"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row>
        </Section>
      </Shape>"""


def v_arrow_xml(shape_id: int, arrow: Arrow) -> str:
    x1, y1 = arrow.start
    x2, y2 = arrow.end
    vy1 = visio_y(y1)
    vy2 = visio_y(y2)
    dx = x2 - x1
    dy = vy2 - vy1
    length = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    pin_x = (x1 + x2) / 2
    pin_y = (vy1 + vy2) / 2
    line_pattern = "2" if arrow.dashed else "1"
    label = escape(arrow.label)
    text = f"<Text>{label}</Text>" if label else "<Text/>"
    char_size = 7 / 72.0
    return f"""
      <Shape ID="{shape_id}" NameU="arrow_{shape_id}" Type="Shape" LineStyle="0" FillStyle="0" TextStyle="0">
        <Cell N="PinX" V="{pin_x:.4f}"/>
        <Cell N="PinY" V="{pin_y:.4f}"/>
        <Cell N="Width" V="{length:.4f}"/>
        <Cell N="Height" V="0.0500"/>
        <Cell N="LocPinX" V="{length / 2:.4f}"/>
        <Cell N="LocPinY" V="0"/>
        <Cell N="Angle" V="{angle:.8f}"/>
        <Cell N="FillPattern" V="0"/>
        <Cell N="LineColor" V="{arrow.color}"/>
        <Cell N="LinePattern" V="{line_pattern}"/>
        <Cell N="LineWeight" V="{arrow.width / 72.0:.4f}"/>
        <Cell N="EndArrow" V="4"/>
        <Section N="Character"><Row IX="0"><Cell N="Font" V="0"/><Cell N="Size" V="{char_size:.4f}"/><Cell N="Color" V="{arrow.color}"/></Row></Section>
        <Section N="Paragraph"><Row IX="0"><Cell N="HorzAlign" V="1"/></Row></Section>
        {text}
        <Section N="Geometry" IX="0">
          <Cell N="NoFill" V="1"/><Cell N="NoLine" V="0"/>
          <Row T="MoveTo" IX="1"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row>
          <Row T="LineTo" IX="2"><Cell N="X" V="{length:.4f}"/><Cell N="Y" V="0"/></Row>
        </Section>
      </Shape>"""


def page_contents_xml() -> str:
    shape_id = 1
    body: list[str] = ['<Shapes>']
    for arrow in ARROWS:
        body.append(v_arrow_xml(shape_id, arrow))
        shape_id += 1
    for box in BOXES:
        body.append(v_shape_xml(shape_id, box))
        shape_id += 1
    for item in TEXT_ITEMS:
        body.append(v_text_xml(shape_id, item))
        shape_id += 1
    body.append("</Shapes>")
    return "\n".join(body)


def render_vdx(path: Path) -> None:
    shapes = page_contents_xml()
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<VisioDocument xmlns="urn:schemas-microsoft-com:office:visio" version="11.0" xml:space="preserve">
  <DocumentProperties><Title>C10 framework for interpretable multimodal sleep staging</Title><Creator>Codex</Creator></DocumentProperties>
  <FaceNames><FaceName ID="0" NameU="Calibri" Name="Calibri"/></FaceNames>
  <StyleSheets><StyleSheet ID="0" NameU="No Style" Name="No Style" Type="Style"/></StyleSheets>
  <Pages>
    <Page ID="0" NameU="C10 framework" Name="C10 framework">
      <PageSheet>
        <Cell N="PageWidth" V="{PAGE_W:.4f}"/>
        <Cell N="PageHeight" V="{PAGE_H:.4f}"/>
        <Cell N="DrawingScale" V="1"/>
        <Cell N="PageScale" V="1"/>
      </PageSheet>
      {shapes}
    </Page>
  </Pages>
</VisioDocument>
"""
    path.write_text(xml, encoding="utf-8")


def render_vsdx(path: Path) -> None:
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
  <FaceNames><FaceName ID="0" NameU="Calibri" Name="Calibri"/></FaceNames>
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
      <Cell N="DrawingSizeType" V="0"/>
      <Cell N="DrawingScaleType" V="0"/>
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
  <dc:title>C10 framework for interpretable multimodal sleep staging</dc:title>
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
  <Company>OpenAI</Company>
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


def render_png(path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return

    scale = 160
    img = Image.new("RGB", (int(PAGE_W * scale), int(PAGE_H * scale)), BG)
    draw = ImageDraw.Draw(img)

    def font(size: int, bold: bool = False):
        candidates = [
            "arialbd.ttf" if bold else "arial.ttf",
            "calibrib.ttf" if bold else "calibri.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, int(size * scale / 96 * 1.6))
            except Exception:
                continue
        return ImageFont.load_default()

    for item in TEXT_ITEMS:
        fnt = font(item.font_size, item.bold)
        lines = svg_text_lines(item.text, max(20, int(item.w * 16)))
        y = item.y * scale
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=fnt)
            draw.text(((item.x + item.w / 2) * scale - (bbox[2] - bbox[0]) / 2, y), line, fill=item.color, font=fnt)
            y += (item.font_size + 4) * scale / 96 * 1.6

    for arrow in ARROWS:
        x1, y1 = arrow.start[0] * scale, arrow.start[1] * scale
        x2, y2 = arrow.end[0] * scale, arrow.end[1] * scale
        draw.line((x1, y1, x2, y2), fill=arrow.color, width=max(1, int(arrow.width * scale / 70)))
        angle = math.atan2(y2 - y1, x2 - x1)
        head = 0.08 * scale
        left = (x2 - head * math.cos(angle - math.pi / 6), y2 - head * math.sin(angle - math.pi / 6))
        right = (x2 - head * math.cos(angle + math.pi / 6), y2 - head * math.sin(angle + math.pi / 6))
        draw.polygon([(x2, y2), left, right], fill=arrow.color)
        if arrow.label:
            fnt = font(7)
            bbox = draw.textbbox((0, 0), arrow.label, font=fnt)
            draw.text(((x1 + x2) / 2 - (bbox[2] - bbox[0]) / 2, (y1 + y2) / 2 - 18), arrow.label, fill=arrow.color, font=fnt)

    for box in BOXES:
        rect = [box.x * scale, box.y * scale, (box.x + box.w) * scale, (box.y + box.h) * scale]
        draw.rounded_rectangle(rect, radius=int(box.radius * scale), fill=box.fill, outline=box.stroke, width=2)
        title_font = font(box.title_size, True)
        bbox = draw.textbbox((0, 0), box.title, font=title_font)
        tx = (box.x + box.w / 2) * scale - (bbox[2] - bbox[0]) / 2
        ty = box.y * scale + 0.10 * scale
        draw.text((tx, ty), box.title, fill=TEXT_COLOR, font=title_font)
        body_font = font(box.font_size)
        y = ty + 0.20 * scale
        for line in svg_text_lines(box.body, max(18, int(box.w * 17))):
            bbox = draw.textbbox((0, 0), line, font=body_font)
            draw.text(((box.x + box.w / 2) * scale - (bbox[2] - bbox[0]) / 2, y), line, fill=TEXT_COLOR, font=body_font)
            y += 0.14 * scale
    img.save(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_svg(OUT_DIR / "c10_framework.svg")
    render_vdx(OUT_DIR / "c10_framework.vdx")
    render_vsdx(OUT_DIR / "c10_framework.vsdx")
    render_png(OUT_DIR / "c10_framework_preview.png")
    print(f"Wrote outputs to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()


