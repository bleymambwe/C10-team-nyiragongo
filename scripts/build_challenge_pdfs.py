"""Build the four Cohort Challenge PDFs from their Markdown sources."""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs" / "challenges"
OUTPUT_DIR = ROOT / "doc"

DOCUMENTS = {
    "problem_statement.md": "problem_statement.pdf",
    "data_card.md": "data_card.pdf",
    "impact_statement_card.md": "impact_statement_card.pdf",
    "stakeholder_engagement.md": "stakeholder_engagement.pdf",
}

INK = colors.HexColor("#132238")
BLUE = colors.HexColor("#1B5E83")
TEAL = colors.HexColor("#00A6A6")
MIST = colors.HexColor("#EAF4F6")
MUTED = colors.HexColor("#526273")


def register_fonts() -> tuple[str, str]:
    candidates = [
        (
            Path("C:/Windows/Fonts/aptos.ttf"),
            Path("C:/Windows/Fonts/aptosbd.ttf"),
        ),
        (
            Path("C:/Windows/Fonts/calibri.ttf"),
            Path("C:/Windows/Fonts/calibrib.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("SubmissionSans", str(regular)))
            pdfmetrics.registerFont(TTFont("SubmissionSansBold", str(bold)))
            return "SubmissionSans", "SubmissionSansBold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=24,
            leading=28,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=10,
            leading=15,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=15,
            leading=19,
            textColor=BLUE,
            spaceBefore=13,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName=FONT_BOLD,
            fontSize=11.5,
            leading=15,
            textColor=INK,
            spaceBefore=9,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.2,
            leading=13.2,
            textColor=INK,
            spaceAfter=6,
            allowWidows=0,
            allowOrphans=0,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.2,
            leading=13.2,
            leftIndent=13,
            firstLineIndent=-7,
            bulletIndent=3,
            textColor=INK,
            spaceAfter=3,
            allowWidows=0,
            allowOrphans=0,
        ),
    }


def inline_markup(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    return text


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(INK)
    canvas.rect(0, height - 9 * mm, width, 9 * mm, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, height - 10.5 * mm, width, 1.5 * mm, fill=1, stroke=0)
    canvas.setFont(FONT_BOLD, 7.5)
    canvas.setFillColor(colors.white)
    canvas.drawString(18 * mm, height - 6 * mm, "NYIRAGONGO  /  COHORT 10 SUBMISSION")
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "AI Saturdays Lagos - Nyiragongo - Latent Probing for Toxicity Detection")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_one(source: Path, output: Path) -> None:
    raw = source.read_text(encoding="utf-8").strip().splitlines()
    title = raw[0].removeprefix("# ").strip()
    content = raw[1:]
    s = styles()
    top_space = 12 * mm
    if source.name == "problem_statement.md":
        # Keep the concise statement and its contributor declaration together.
        top_space = 5 * mm
        s["title"].fontSize = 22
        s["title"].leading = 25
        s["h2"].fontSize = 13.5
        s["h2"].leading = 16
        s["h2"].spaceBefore = 8
        s["h2"].spaceAfter = 4
        s["body"].fontSize = 8.7
        s["body"].leading = 12
        s["body"].spaceAfter = 4
    story = [Spacer(1, top_space), Paragraph(inline_markup(title), s["title"])]
    story.extend(
        [
            Paragraph("Latent Probing for Toxicity Detection", s["subtitle"]),
            HRFlowable(width="38%", thickness=2, color=TEAL, spaceBefore=4, spaceAfter=12),
        ]
    )

    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(inline_markup(" ".join(paragraph)), s["body"]))
            paragraph.clear()

    for line in content:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[3:]), s["h2"]))
        elif stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[4:]), s["h3"]))
        elif stripped.startswith("- "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[2:]), s["bullet"], bulletText="-"))
        elif re.match(r"^\d+\. ", stripped):
            flush_paragraph()
            number, item = stripped.split(". ", 1)
            story.append(Paragraph(inline_markup(item), s["bullet"], bulletText=f"{number}."))
        else:
            paragraph.append(stripped.removesuffix("  "))
    flush_paragraph()

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="Blessings Mambwe, Fafemi Adeola, Musonda Musunga, and Hamna Kaleem",
        subject="AI Saturdays Lagos Cohort 10 Challenge Submission",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source_name, output_name in DOCUMENTS.items():
        build_one(SOURCE_DIR / source_name, OUTPUT_DIR / output_name)
        print(f"Built {OUTPUT_DIR / output_name}")


if __name__ == "__main__":
    main()
