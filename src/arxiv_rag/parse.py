"""Turn arXiv PDFs into clean paragraphs that keep their page and section.

Output: one JSON line per paragraph in data/parsed/<paper_id>.jsonl
    {"paper_id": ..., "page": 2, "section": "1 Introduction", "text": "..."}

PDFs store glyph positions, not reading order, so this is heuristic. The heuristics
are deliberately simple and each one is unit-tested; failures degrade gracefully
(a missed heading means section=None, never lost text).
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

SECTION_NUMBER = re.compile(r"^\d+(\.\d+)*\.?$")  # "1", "2.1", "3.2.1"
NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+\S")  # "1 Introduction"
UNNUMBERED_HEADINGS = {
    "abstract", "introduction", "conclusion", "conclusions", "references",
    "acknowledgments", "acknowledgements", "appendix", "limitations",
}
STOP_SECTIONS = {"references", "bibliography"}  # everything after is dropped
MARGIN_FRACTION = 0.06  # top/bottom strip treated as running header/footer


def join_lines(lines: list[str]) -> str:
    """Merge hard-wrapped lines into one paragraph, repairing line-end hyphenation.

    "bench-" + "marking"  -> "benchmarking"   (soft hyphen from justification)
    "LLM-"   + "based"    -> "LLM-based"      (real hyphen: prefix is not lowercase)

    Known limit: "evidence-" + "grounding" becomes "evidencegrounding", because a
    lowercase prefix looks the same as a justification break. Fixing that needs a
    dictionary; it is cheap noise compared to leaving every split word broken.
    """
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        prefix = out.rsplit(" ", 1)[-1]
        if out.endswith("-") and line[0].islower():
            soft = re.fullmatch(r"[a-z]{2,}-", prefix)
            out = (out[:-1] if soft else out) + line  # real hyphens keep the "-"
        else:
            out = out + " " + line
    return re.sub(r"\s+", " ", out).strip()


def is_heading_text(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 100 or t.endswith((".", ",", ";")) and not SECTION_NUMBER.match(t):
        return False
    if t.lower().rstrip(":") in UNNUMBERED_HEADINGS:
        return True
    return bool(NUMBERED_HEADING.match(t))


def body_font_size(doc: pymupdf.Document) -> float:
    """The most common font size, weighted by characters, is the body text size."""
    sizes: Counter[float] = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


def block_lines(block: dict) -> tuple[list[str], float, bool]:
    """Return (line texts, max font size, all-bold?) for a text block."""
    lines, max_size, bold_chars, total_chars = [], 0.0, 0, 0
    for line in block["lines"]:
        text = "".join(s["text"] for s in line["spans"])
        lines.append(text)
        for s in line["spans"]:
            n = len(s["text"].strip())
            max_size = max(max_size, s["size"])
            total_chars += n
            if s["flags"] & 16 or "bold" in s["font"].lower():
                bold_chars += n
    return lines, max_size, total_chars > 0 and bold_chars / total_chars > 0.8


def parse_pdf(path: Path, paper_id: str) -> list[dict]:
    doc = pymupdf.open(path)
    body = body_font_size(doc)
    records: list[dict] = []
    section: str | None = None
    seen_abstract = False
    pending_number: str | None = None  # "1" waiting for its title block

    for page_no, page in enumerate(doc, start=1):
        height = page.rect.height
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0 or not block.get("lines"):
                continue
            lines, size, bold = block_lines(block)
            text = join_lines(lines)
            if not text:
                continue

            y0, y1 = block["bbox"][1], block["bbox"][3]
            in_margin = y1 < height * MARGIN_FRACTION or y0 > height * (1 - MARGIN_FRACTION)
            if (in_margin and len(text) < 80) or text.isdigit() or text.startswith("arXiv:"):
                continue  # running header, footer, page number, arXiv stamp

            looks_bold_or_big = bold or size >= body + 0.8
            if looks_bold_or_big and SECTION_NUMBER.match(text):
                pending_number = text.rstrip(".")  # "1" and "Introduction" arrive separately
                continue
            if pending_number is not None:
                text, pending_number = f"{pending_number} {text}", None
                heading = looks_bold_or_big and len(text) <= 100
            else:
                heading = looks_bold_or_big and is_heading_text(text)

            if heading:
                section = text
                seen_abstract = seen_abstract or text.lower().startswith("abstract")
                if re.sub(r"^[\d.\s]+", "", text).lower() in STOP_SECTIONS:
                    return records
                continue
            if not seen_abstract and page_no == 1:
                continue  # front matter: title, authors, affiliations, emails
            records.append({"paper_id": paper_id, "page": page_no, "section": section, "text": text})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse downloaded PDFs into paragraphs")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    out_dir = args.data_dir / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for pdf in sorted((args.data_dir / "pdfs").glob("*.pdf")):
        records = parse_pdf(pdf, pdf.stem)
        with (out_dir / f"{pdf.stem}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sections = {r["section"] for r in records}
        print(f"{pdf.stem}: {len(records)} paragraphs, {len(sections)} sections")


if __name__ == "__main__":
    main()
