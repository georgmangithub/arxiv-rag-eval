"""One canonical text per paper, shared by every chunker and by the gold set.

Parsed paragraphs are joined with PARAGRAPH_SEP into a single string. Each paragraph
keeps its character span in that string, so a chunk from any strategy, and any gold
evidence quote, can be located by character offsets in the same coordinate system.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from arxiv_rag.ingest import pdf_filename

PARAGRAPH_SEP = "\n\n"


@dataclass
class Paragraph:
    text: str
    section: str | None
    page: int
    char_start: int  # text == document.text[char_start:char_end]
    char_end: int


@dataclass
class Document:
    paper_id: str  # filename stem, e.g. "2401.12345v2" or "cs_0112017v1"
    title: str
    text: str
    paragraphs: list[Paragraph]


def build_document(paper_id: str, title: str, records: list[dict]) -> Document:
    paragraphs, parts, pos = [], [], 0
    for r in records:
        if parts:
            pos += len(PARAGRAPH_SEP)
        paragraphs.append(Paragraph(r["text"], r["section"], r["page"], pos, pos + len(r["text"])))
        parts.append(r["text"])
        pos += len(r["text"])
    return Document(paper_id, title, PARAGRAPH_SEP.join(parts), paragraphs)


def load_titles(data_dir: Path) -> dict[str, str]:
    """Map filename stem -> title. Parsed files are keyed by stem, not raw arXiv id."""
    path = data_dir / "metadata.jsonl"
    if not path.exists():
        return {}
    titles: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                m = json.loads(line)
                titles[pdf_filename(m["paper_id"])[: -len(".pdf")]] = m["title"]
            except (ValueError, KeyError, TypeError, AttributeError):
                print(f"WARNING: {path}:{n} is not a valid metadata record (half-written?); skipping it")
    return titles


def id_to_stem(paper_id: str) -> str:
    """arXiv id -> filename stem, by ingest.pdf_filename's rule ("cs/0112017v1" -> "cs_0112017v1")."""
    return pdf_filename(paper_id)[: -len(".pdf")]


def ids_to_stems(paper_ids: list[str]) -> list[str]:
    return [id_to_stem(i) for i in paper_ids]


def load_documents(data_dir: Path = Path("data"), paper_ids: list[str] | None = None) -> list[Document]:
    """Parsed documents; `paper_ids` are filename stems (see ids_to_stems), default all parsed.

    Raises FileNotFoundError naming every requested stem that has no parsed file.
    """
    titles = load_titles(data_dir)
    parsed_dir = data_dir / "parsed"
    stems = paper_ids if paper_ids is not None else sorted(p.stem for p in parsed_dir.glob("*.jsonl"))
    missing = [s for s in stems if not (parsed_dir / f"{s}.jsonl").is_file()]
    if missing:
        raise FileNotFoundError(f"no parsed file in {parsed_dir} for: {', '.join(missing)}")
    docs = []
    for stem in stems:
        with (parsed_dir / f"{stem}.jsonl").open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        docs.append(build_document(stem, titles.get(stem, ""), records))
    return docs
