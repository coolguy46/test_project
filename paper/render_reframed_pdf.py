from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "reframed_main_draft.md"
OUTPUT = ROOT / "reframed_main_ready.pdf"


def _inline_markup(text: str) -> str:
    text = html.escape(text, quote=False)
    text = text.replace("+/-", "&plusmn;")
    text = text.replace("->", "&rarr;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text


def _styles() -> StyleSheet1:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="PaperTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#172554"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PaperAuthor",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#1d4ed8"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=10.2,
            leading=14,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#111827"),
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AbstractBody",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=10.2,
            leading=14,
            leftIndent=0.15 * inch,
            rightIndent=0.15 * inch,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#111827"),
            spaceAfter=8,
        )
    )
    return styles


def _table_from_lines(lines: list[str], styles: StyleSheet1) -> Table:
    rows: list[list[str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if set(line.replace("|", "").replace(" ", "")) <= {"-", ":"}:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append([_inline_markup(cell) for cell in cells])

    flow_rows = [[Paragraph(cell, styles["Body"]) for cell in row] for row in rows]
    col_count = max(len(row) for row in rows)
    usable = 7.0 * inch
    col_widths = [usable / col_count] * col_count
    table = Table(flow_rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#94a3b8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    return table


def _page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(doc.leftMargin, 0.45 * inch, "Reframed draft for reviewer-safe NeurIPS positioning")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build() -> None:
    styles = _styles()
    story = []

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    i = 0
    in_abstract = False

    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            i += 1
            continue

        if line.startswith("# "):
            story.append(Spacer(1, 0.15 * inch))
            story.append(Paragraph(_inline_markup(line[2:].strip()), styles["PaperTitle"]))
            i += 1
            continue

        if line.startswith("**") and line.endswith("**") and len(line) > 4:
            story.append(Paragraph(_inline_markup(line[2:-2].strip()), styles["PaperAuthor"]))
            i += 1
            continue

        if line.startswith("## "):
            title = line[3:].strip()
            in_abstract = title.lower() == "abstract"
            story.append(Paragraph(_inline_markup(title), styles["Section"]))
            i += 1
            continue

        if line.startswith("### "):
            story.append(Paragraph(_inline_markup(line[4:].strip()), styles["Subsection"]))
            i += 1
            continue

        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            story.append(_table_from_lines(block, styles))
            story.append(Spacer(1, 0.12 * inch))
            continue

        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:].strip())
                i += 1
            flow = ListFlowable(
                [ListItem(Paragraph(_inline_markup(item), styles["Body"])) for item in items],
                bulletType="bullet",
                leftIndent=18,
            )
            story.append(flow)
            story.append(Spacer(1, 0.05 * inch))
            continue

        para_lines = [line.strip()]
        i += 1
        while i < len(lines):
            nxt = lines[i].rstrip()
            if not nxt.strip():
                break
            if nxt.startswith(("# ", "## ", "### ", "- ", "|")):
                break
            if nxt.startswith("**") and nxt.endswith("**"):
                break
            para_lines.append(nxt.strip())
            i += 1
        text = " ".join(para_lines)
        story.append(Paragraph(_inline_markup(text), styles["AbstractBody" if in_abstract else "Body"]))

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=0.78 * inch,
        rightMargin=0.78 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.68 * inch,
        title="Graph-Conditioned Transformers as a Structural Inductive Bias for Physiological Time-Series Shift",
        author="Anonymous Authors",
    )
    doc.build(story, onFirstPage=_page, onLaterPages=_page)


if __name__ == "__main__":
    build()
