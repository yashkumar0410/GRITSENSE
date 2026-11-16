"""Render GRITSENSE_TECHNICAL_REPORT.md into DOCX and PDF with the standard library only.

    python docs/report/build_report.py

The Markdown source in this folder is the single source of truth. Only the
subset of Markdown used by the report is supported: ATX headings, paragraphs,
bullets, numbered items, pipe tables, fenced code blocks and horizontal rules.

PDF output uses the PDF base-14 fonts (Helvetica / Courier), so no font files
or third-party packages are needed. DOCX output is written as OOXML directly.
"""

from __future__ import annotations

import re
import textwrap
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "GRITSENSE_TECHNICAL_REPORT.md"
DOCX_OUT = HERE / "GRITSENSE_TECHNICAL_REPORT.docx"
PDF_OUT = HERE / "GRITSENSE_TECHNICAL_REPORT.pdf"
DOC_TITLE = ("GRITSENSE - COMPLETE SYSTEM ANALYSIS, IMPLEMENTATION DOCUMENTATION "
             "& TECHNICAL AUDIT")

# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------


def parse_markdown(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[dict] = []
    i = 0
    n = len(lines)

    def is_table_row(text: str) -> bool:
        return text.lstrip().startswith("|") and text.rstrip().endswith("|")

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            code: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i].rstrip("\n"))
                i += 1
            i += 1
            blocks.append({"type": "code", "lines": code})
            continue

        if re.fullmatch(r"-{3,}", stripped):
            blocks.append({"type": "rule"})
            i += 1
            continue

        match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if match:
            blocks.append({
                "type": "heading",
                "level": len(match.group(1)),
                "text": match.group(2).strip(),
            })
            i += 1
            continue

        if is_table_row(line):
            rows: list[list[str]] = []
            while i < n and is_table_row(lines[i]):
                raw = lines[i].strip().strip("|")
                cells = [cell.strip() for cell in raw.split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", cell or "--") for cell in cells):
                    rows.append(cells)
                i += 1
            if rows:
                blocks.append({"type": "table", "rows": rows})
            continue

        bullet = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        if bullet:
            blocks.append({
                "type": "bullet",
                "indent": min(len(bullet.group(1)) // 2, 3),
                "text": bullet.group(2).strip(),
            })
            i += 1
            continue

        numbered = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if numbered:
            blocks.append({
                "type": "numbered",
                "marker": numbered.group(2) + ".",
                "indent": min(len(numbered.group(1)) // 2, 3),
                "text": numbered.group(3).strip(),
            })
            i += 1
            continue

        paragraph: list[str] = []
        while i < n:
            candidate = lines[i]
            if not candidate.strip():
                break
            if (candidate.strip().startswith(("```", "#", "|"))
                    or re.fullmatch(r"-{3,}", candidate.strip())
                    or re.match(r"^\s*[-*]\s+", candidate)
                    or re.match(r"^\s*\d+\.\s+", candidate)):
                break
            paragraph.append(candidate.strip())
            i += 1
        if paragraph:
            blocks.append({"type": "paragraph", "text": " ".join(paragraph)})

    return blocks


def plain(text: str) -> str:
    return text.replace("**", "").replace("`", "")


# ---------------------------------------------------------------------------
# Unicode -> WinAnsi-safe transliteration for the PDF
# ---------------------------------------------------------------------------

TRANSLITERATE = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u2192": "->",
    "\u2190": "<-", "\u2265": ">=", "\u2264": "<=", "\u2248": "~",
    "\u00d7": "x", "\u00b7": ".", "\u00b0": "deg", "\u00b1": "+/-",
    "\u2212": "-", "\u03a3": "sum", "\u03a0": "prod", "\u03b1": "alpha",
    "\u03a6": "phi", "\u00b2": "^2", "\u00b3": "^3", "\u00b5": "u",
    "\u2022": "-", "\u2713": "ok", "\u2717": "x", "\u00a0": " ",
}


def ascii_safe(text: str) -> str:
    for source, target in TRANSLITERATE.items():
        text = text.replace(source, target)
    return text.encode("cp1252", "replace").decode("cp1252")


# ---------------------------------------------------------------------------
# PDF writer
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = 595.28, 841.89
MARGIN_L, MARGIN_R = 44.0, 44.0
MARGIN_T, MARGIN_B = 50.0, 46.0
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

HELVETICA = {
    " ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667,
    "'": 191, "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333,
    ".": 278, "/": 278, ":": 278, ";": 278, "<": 584, "=": 584, ">": 584,
    "?": 556, "@": 1015, "A": 667, "B": 667, "C": 722, "D": 722, "E": 667,
    "F": 611, "G": 778, "H": 722, "I": 278, "J": 500, "K": 667, "L": 556,
    "M": 833, "N": 722, "O": 778, "P": 667, "Q": 778, "R": 722, "S": 667,
    "T": 611, "U": 722, "V": 667, "W": 944, "X": 667, "Y": 667, "Z": 611,
    "[": 278, "\\": 278, "]": 278, "^": 469, "_": 556, "`": 333, "a": 556,
    "b": 556, "c": 500, "d": 556, "e": 556, "f": 278, "g": 556, "h": 556,
    "i": 222, "j": 222, "k": 500, "l": 222, "m": 833, "n": 556, "o": 556,
    "p": 556, "q": 556, "r": 333, "s": 500, "t": 278, "u": 556, "v": 500,
    "w": 722, "x": 500, "y": 500, "z": 500, "{": 334, "|": 260, "}": 334,
    "~": 584,
}
for digit in "0123456789":
    HELVETICA[digit] = 556

HELVETICA_BOLD = {
    " ": 278, "!": 333, '"': 474, "#": 556, "$": 556, "%": 889, "&": 722,
    "'": 238, "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333,
    ".": 278, "/": 278, ":": 333, ";": 333, "<": 584, "=": 584, ">": 584,
    "?": 611, "@": 975, "A": 722, "B": 722, "C": 722, "D": 722, "E": 667,
    "F": 611, "G": 778, "H": 722, "I": 278, "J": 556, "K": 722, "L": 611,
    "M": 833, "N": 722, "O": 778, "P": 667, "Q": 778, "R": 722, "S": 667,
    "T": 611, "U": 722, "V": 667, "W": 944, "X": 667, "Y": 667, "Z": 611,
    "[": 333, "\\": 278, "]": 333, "^": 584, "_": 556, "`": 333, "a": 556,
    "b": 611, "c": 556, "d": 611, "e": 556, "f": 333, "g": 611, "h": 611,
    "i": 278, "j": 278, "k": 556, "l": 278, "m": 889, "n": 611, "o": 611,
    "p": 611, "q": 611, "r": 389, "s": 556, "t": 333, "u": 611, "v": 556,
    "w": 778, "x": 556, "y": 556, "z": 500, "{": 389, "|": 280, "}": 389,
    "~": 584,
}
for digit in "0123456789":
    HELVETICA_BOLD[digit] = 556

MONO_ADVANCE = 600


def text_width(text: str, font: str, size: float) -> float:
    if font.startswith("F3") or font.startswith("F4"):
        return len(text) * MONO_ADVANCE * size / 1000.0
    table = HELVETICA_BOLD if font.startswith("F2") else HELVETICA
    return sum(table.get(ch, 556) for ch in text) * size / 1000.0


def wrap_text(text: str, font: str, size: float, width: float) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if text_width(candidate, font, size) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        while text_width(word, font, size) > width and len(word) > 1:
            cut = max(1, int(len(word) * width / max(text_width(word, font, size), 1e-6)))
            lines.append(word[:cut])
            word = word[cut:]
        current = word
    if current:
        lines.append(current)
    return lines


def pdf_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


class PdfDocument:
    def __init__(self) -> None:
        self.pages: list[list[str]] = []
        self.new_page()

    def new_page(self) -> None:
        self.pages.append([])
        self.y = PAGE_H - MARGIN_T

    @property
    def ops(self) -> list[str]:
        return self.pages[-1]

    def space(self, amount: float) -> None:
        self.y -= amount

    def ensure(self, needed: float) -> None:
        if self.y - needed < MARGIN_B:
            self.new_page()

    def text(self, x: float, y: float, content: str, font: str, size: float) -> None:
        if not content:
            return
        self.ops.append(
            f"BT /{font} {size:.2f} Tf {x:.2f} {y:.2f} Td ({pdf_escape(content)}) Tj ET"
        )

    def line_text(self, content: str, font: str, size: float, x: float | None = None) -> None:
        left = MARGIN_L if x is None else x
        self.text(left, self.y, content, font, size)
        self.y -= size * 1.28

    def rect(self, x: float, y: float, w: float, h: float, fill: tuple | None = None,
             stroke: tuple | None = None, width: float = 0.5) -> None:
        parts = []
        if fill is not None:
            parts.append(f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg")
        if stroke is not None:
            parts.append(f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG")
            parts.append(f"{width:.2f} w")
        parts.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re")
        parts.append("B" if (fill is not None and stroke is not None) else
                     ("f" if fill is not None else "S"))
        self.ops.append(" ".join(parts))

    def serialise(self) -> bytes:
        objects: list[bytes] = []
        page_count = len(self.pages)
        font_objs = {
            "F1": b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            "F2": b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
            "F3": b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
            "F4": b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Bold /Encoding /WinAnsiEncoding >>",
        }
        font_ids = {name: 5 + index for index, name in enumerate(font_objs)}
        first_page_obj = 5 + len(font_objs)
        pages_obj = first_page_obj + page_count * 2

        objects.append(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_obj)
        objects.append(b"<< /Type /Pages /Count %d /Kids [%s] >>" % (
            page_count,
            " ".join(f"{first_page_obj + 2 * i} 0 R" for i in range(page_count)).encode(),
        ))
        objects.append(b"<< /Producer (GritSense report builder) >>")
        for name in ("F1", "F2", "F3", "F4"):
            objects.append(font_objs[name])

        for index in range(page_count):
            page_obj = first_page_obj + 2 * index
            content_obj = page_obj + 1
            ops = list(self.pages[index])

            footer_left = ascii_safe("GritSense - technical report")
            footer_right = ascii_safe(f"Page {index + 1} of {page_count}")
            ops.append(
                f"BT /F1 7.5 Tf {MARGIN_L:.2f} 26 Td ({pdf_escape(footer_left)}) Tj ET"
            )
            right_x = PAGE_W - MARGIN_R - text_width(footer_right, "F1", 7.5)
            ops.append(f"BT /F1 7.5 Tf {right_x:.2f} 26 Td ({pdf_escape(footer_right)}) Tj ET")
            ops.append("0.75 0.78 0.82 RG 0.5 w "
                       f"{MARGIN_L:.2f} 36 m {PAGE_W - MARGIN_R:.2f} 36 l S")

            stream = "\n".join(ops).encode("cp1252", "replace")
            objects.append(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %.2f %.2f] "
                b"/Resources << /Font << %s >> >> /Contents %d 0 R >>" % (
                    pages_obj, PAGE_W, PAGE_H,
                    b" ".join(f"/{name} {fid} 0 R".encode() for name, fid in font_ids.items()),
                    content_obj,
                )
            )
            objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for number, body in enumerate(objects, start=1):
            offsets.append(len(out))
            out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
        xref_at = len(out)
        out += f"xref\n0 {len(objects) + 1}\n".encode()
        out += b"0000000000 65535 f \n"
        for offset in offsets[1:]:
            out += f"{offset:010d} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n"
                f"{xref_at}\n%%EOF\n").encode()
        return bytes(out)


def render_pdf(blocks: list[dict], out_path: Path) -> int:
    doc = PdfDocument()
    first_heading = True

    for block in blocks:
        kind = block["type"]

        if kind == "rule":
            if doc.y < PAGE_H - MARGIN_T - 10:
                doc.new_page()
            continue

        if kind == "heading":
            level = block["level"]
            text = ascii_safe(plain(block["text"]))
            if level == 1:
                if not first_heading:
                    doc.space(12)
                first_heading = False
                for line in wrap_text(text, "F2", 17, CONTENT_W):
                    doc.ensure(24)
                    doc.text(MARGIN_L, doc.y, line, "F2", 17)
                    doc.space(21)
                doc.y -= 2
                doc.rect(MARGIN_L, doc.y, CONTENT_W, 1.2, fill=(0.18, 0.73, 0.63))
                doc.space(11)
            elif level == 2:
                doc.space(8)
                doc.ensure(36)
                for line in wrap_text(text, "F2", 13.6, CONTENT_W):
                    doc.text(MARGIN_L, doc.y, line, "F2", 13.6)
                    doc.space(17.4)
                doc.space(3)
            elif level == 3:
                doc.space(6)
                for line in wrap_text(text, "F2", 11.4, CONTENT_W):
                    doc.ensure(18)
                    doc.text(MARGIN_L, doc.y, line, "F2", 11.4)
                    doc.space(14.6)
                doc.space(1)
            else:
                doc.space(3)
                for line in wrap_text(text, "F2", 10.4, CONTENT_W):
                    doc.ensure(16)
                    doc.text(MARGIN_L, doc.y, line, "F2", 10.4)
                    doc.space(13.6)
                doc.space(0.5)
            continue

        if kind == "paragraph":
            text = ascii_safe(plain(block["text"]))
            for line in wrap_text(text, "F1", 10.1, CONTENT_W):
                doc.ensure(13)
                doc.text(MARGIN_L, doc.y, line, "F1", 10.1)
                doc.space(12.5)
            doc.space(3.2)
            continue

        if kind in ("bullet", "numbered"):
            indent = MARGIN_L + 14 + 12 * block.get("indent", 0)
            marker = "•" if kind == "bullet" else block["marker"]
            text = ascii_safe(plain(block["text"]))
            lines = wrap_text(text, "F1", 10.1, CONTENT_W - (indent - MARGIN_L))
            doc.ensure(13)
            doc.text(indent - 11, doc.y, ascii_safe(marker), "F1", 10.1)
            for index, line in enumerate(lines):
                doc.ensure(13)
                doc.text(indent, doc.y, line, "F1", 10.1)
                doc.space(12.3 if index < len(lines) - 1 else 12.8)
            doc.space(1.0)
            continue

        if kind == "code":
            size = 7.0
            leading = 8.4
            char_w = MONO_ADVANCE * size / 1000.0
            usable_chars = max(20, int((CONTENT_W - 14) / char_w))
            wrapped: list[str] = []
            for raw in block["lines"]:
                safe = ascii_safe(raw).replace("\t", "    ")
                if not safe:
                    wrapped.append("")
                    continue
                wrapped.extend(textwrap.wrap(safe, width=usable_chars,
                                             drop_whitespace=False,
                                             break_long_words=True,
                                             break_on_hyphens=False) or [""])
            index = 0
            while index < len(wrapped):
                doc.ensure(leading * 3)
                room = int((doc.y - MARGIN_B) / leading)
                chunk = wrapped[index:index + max(1, room)]
                height = len(chunk) * leading + 8
                top = doc.y + 4
                doc.rect(MARGIN_L, top - height, CONTENT_W, height,
                         fill=(0.957, 0.965, 0.976), stroke=(0.85, 0.87, 0.90), width=0.4)
                doc.space(6)
                for line in chunk:
                    doc.text(MARGIN_L + 7, doc.y, line, "F3", size)
                    doc.space(leading)
                doc.space(6)
                index += len(chunk)
                if index < len(wrapped):
                    doc.new_page()
            doc.space(5)
            continue

        if kind == "table":
            render_pdf_table(doc, block["rows"])

    out_path.write_bytes(doc.serialise())
    return len(doc.pages)


def render_pdf_table(doc: PdfDocument, rows: list[list[str]]) -> None:
    columns = max(len(row) for row in rows)
    rows = [row + [""] * (columns - len(row)) for row in rows]
    size = 6.8
    leading = 8.1
    char_w = MONO_ADVANCE * size / 1000.0

    weights = []
    for column in range(columns):
        longest = max(len(ascii_safe(plain(row[column]))) for row in rows)
        weights.append(min(max(longest, 5), 52))
    total = sum(weights)
    widths = [max(34.0, CONTENT_W * weight / total) for weight in weights]
    scale = CONTENT_W / sum(widths)
    widths = [width * scale for width in widths]

    def wrap_cell(value: str, width: float) -> list[str]:
        usable = max(4, int((width - 5) / char_w))
        safe = ascii_safe(plain(value))
        return textwrap.wrap(safe, width=usable, break_long_words=True,
                             break_on_hyphens=False) or [""]

    header_cells = [wrap_cell(cell, widths[i]) for i, cell in enumerate(rows[0])]
    header_lines = max(len(cell) for cell in header_cells)

    def draw_row(cells: list[list[str]], fill: tuple | None) -> None:
        height = max(len(cell) for cell in cells) * leading + 3.6
        doc.ensure(height + 2)
        top = doc.y
        x = MARGIN_L
        for column, cell in enumerate(cells):
            doc.rect(x, top - height, widths[column], height, fill=fill,
                     stroke=(0.78, 0.80, 0.84), width=0.4)
            y = top - 2.4 - leading + 1.6
            for line in cell:
                doc.text(x + 2.4, y, line, "F4" if fill else "F3", size)
                y -= leading
            x += widths[column]
        doc.y = top - height

    draw_row(header_cells, fill=(0.902, 0.929, 0.949))
    for row in rows[1:]:
        draw_row([wrap_cell(cell, widths[i]) for i, cell in enumerate(row)], fill=None)
    doc.space(5.5)


# ---------------------------------------------------------------------------
# DOCX writer
# ---------------------------------------------------------------------------

NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')


def run_xml(text: str, bold: bool = False, font: str | None = None,
            size_half_points: int | None = None, mono: bool = False) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if mono:
        props.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>')
    elif font:
        props.append(f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}"/>')
    if size_half_points:
        props.append(f'<w:sz w:val="{size_half_points}"/><w:szCs w:val="{size_half_points}"/>')
    props_xml = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f'<w:r>{props_xml}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def inline_runs(text: str, size_half_points: int = 21) -> str:
    pieces = text.split("**")
    runs = []
    for index, piece in enumerate(pieces):
        cleaned = piece.replace("`", "")
        if cleaned:
            runs.append(run_xml(cleaned, bold=index % 2 == 1,
                                size_half_points=size_half_points))
    return "".join(runs)


def paragraph_xml(inner: str, style: str = "Normal", indent: int = 0,
                  space_after: int = 60, keep: bool = False) -> str:
    props = [f'<w:pStyle w:val="{style}"/>']
    if indent:
        props.append(f'<w:ind w:left="{indent}"/>')
    props.append(f'<w:spacing w:after="{space_after}"/>')
    if keep:
        props.append("<w:keepNext/>")
    return f"<w:p><w:pPr>{''.join(props)}</w:pPr>{inner}</w:p>"


def table_xml(rows: list[list[str]]) -> str:
    columns = max(len(row) for row in rows)
    rows = [row + [""] * (columns - len(row)) for row in rows]
    grid = "".join("<w:gridCol w:w=\"1400\"/>" for _ in range(columns))
    borders = ("<w:tblBorders>" + "".join(
        f'<w:{side} w:val="single" w:sz="4" w:color="C6CCD4"/>'
        for side in ("top", "left", "bottom", "right", "insideH", "insideV")
    ) + "</w:tblBorders>")
    body = [f'<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/>{borders}'
            '<w:tblLayout w:type="autofit"/></w:tblPr>'
            f"<w:tblGrid>{grid}</w:tblGrid>"]
    for row_index, row in enumerate(rows):
        cells = []
        for cell in row:
            shade = ('<w:shd w:val="clear" w:color="auto" w:fill="E6EDF4"/>'
                     if row_index == 0 else "")
            inner = inline_runs(cell, size_half_points=15 if row_index else 15)
            cells.append(
                f"<w:tc><w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/>{shade}"
                '<w:vAlign w:val="top"/></w:tcPr>'
                + paragraph_xml(inner or run_xml("", size_half_points=15), space_after=20)
                + "</w:tc>"
            )
        header = "<w:trPr><w:tblHeader/></w:trPr>" if row_index == 0 else ""
        body.append(f"<w:tr>{header}{''.join(cells)}</w:tr>")
    body.append("</w:tbl>")
    body.append(paragraph_xml(run_xml("", size_half_points=8), space_after=40))
    return "".join(body)


def build_docx(blocks: list[dict], out_path: Path) -> None:
    body: list[str] = []
    for block in blocks:
        kind = block["type"]

        if kind == "rule":
            body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
            continue

        if kind == "heading":
            level = block["level"]
            style = {1: "Heading1", 2: "Heading2", 3: "Heading3"}.get(level, "Heading4")
            body.append(paragraph_xml(inline_runs(block["text"], 40 if level == 1 else 24),
                                      style=style, space_after=120, keep=True))
            continue

        if kind == "paragraph":
            body.append(paragraph_xml(inline_runs(block["text"])))
            continue

        if kind == "bullet":
            inner = run_xml("•  ", size_half_points=21) + inline_runs(block["text"])
            body.append(paragraph_xml(inner, indent=280 + 240 * block.get("indent", 0),
                                      space_after=40))
            continue

        if kind == "numbered":
            inner = run_xml(block["marker"] + "  ", bold=True) + inline_runs(block["text"])
            body.append(paragraph_xml(inner, indent=280 + 240 * block.get("indent", 0),
                                      space_after=40))
            continue

        if kind == "code":
            for raw in block["lines"]:
                inner = run_xml(raw.replace("\t", "    ") or " ",
                                mono=True, size_half_points=15)
                body.append(paragraph_xml(inner, style="Code", space_after=0))
            body.append(paragraph_xml(run_xml("", size_half_points=8), space_after=60))
            continue

        if kind == "table":
            body.append(table_xml(block["rows"]))
            continue

    sect = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter1"/>'
            '<w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"'
            ' w:header="709" w:footer="709" w:gutter="0"/></w:sectPr>')
    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f"<w:document {NS}><w:body>{''.join(body)}{sect}</w:body></w:document>")

    styles = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles {NS}>
 <w:docDefaults><w:rPrDefault><w:rPr>
   <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
   <w:sz w:val="21"/><w:szCs w:val="21"/>
 </w:rPr></w:rPrDefault>
 <w:pPrDefault><w:pPr><w:spacing w:after="60" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault>
 </w:docDefaults>
 <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
 <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>
   <w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="240" w:after="120"/>
   <w:outlineLvl w:val="0"/></w:pPr>
   <w:rPr><w:b/><w:sz w:val="36"/><w:color w:val="0F172A"/></w:rPr></w:style>
 <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>
   <w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="240" w:after="80"/>
   <w:outlineLvl w:val="1"/></w:pPr>
   <w:rPr><w:b/><w:sz w:val="28"/><w:color w:val="123A5A"/></w:rPr></w:style>
 <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/>
   <w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="180" w:after="60"/>
   <w:outlineLvl w:val="2"/></w:pPr>
   <w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="1F2937"/></w:rPr></w:style>
 <w:style w:type="paragraph" w:styleId="Heading4"><w:name w:val="heading 4"/>
   <w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="3"/></w:pPr>
   <w:rPr><w:b/><w:sz w:val="21"/></w:rPr></w:style>
 <w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/>
   <w:basedOn w:val="Normal"/><w:pPr><w:shd w:val="clear" w:color="auto" w:fill="F4F6F9"/>
   <w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>
   <w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="15"/></w:rPr></w:style>
</w:styles>"""

    footer = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              f"<w:ftr {NS}><w:p><w:pPr><w:jc w:val=\"center\"/></w:pPr>"
              + run_xml("GritSense - technical report   |   Page ", size_half_points=16)
              + '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
              + '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
              + '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
              + "</w:p></w:ftr>")

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
 <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
 <Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
 <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
 <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
 <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
 <Relationship Id="rIdFooter1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>
</Relationships>"""

    core = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/">
 <dc:title>GritSense - Complete System Analysis, Implementation Documentation and Technical Audit</dc:title>
 <dc:subject>Capstone technical audit</dc:subject>
 <dc:creator>GritSense capstone team</dc:creator>
</cp:coreProperties>"""

    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
 <Application>GritSense report builder</Application>
</Properties>"""

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/_rels/document.xml.rels", document_rels)
        archive.writestr("word/styles.xml", styles)
        archive.writestr("word/footer1.xml", footer)
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)


def main() -> None:
    blocks = parse_markdown(SOURCE)
    pages = render_pdf(blocks, PDF_OUT)
    build_docx(blocks, DOCX_OUT)
    words = len(SOURCE.read_text(encoding="utf-8").split())
    print(f"blocks: {len(blocks)}  words: {words}")
    print(f"PDF : {PDF_OUT.name}  pages: {pages}  size: {PDF_OUT.stat().st_size / 1024:.0f} kB")
    print(f"DOCX: {DOCX_OUT.name}  size: {DOCX_OUT.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
