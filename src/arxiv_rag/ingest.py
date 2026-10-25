"""Download arXiv papers (PDF + metadata) into a local, resumable cache.

Design goals:
- Idempotent: running it twice downloads nothing new.
- Polite: the arxiv client sleeps between requests (arXiv asks for >= 3 s).
- Inspectable: one JSON line of metadata per paper, next to the PDFs.
"""
import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import arxiv
import requests

DEFAULT_DATA_DIR = Path("data")
DOWNLOAD_DELAY_SECONDS = 3.0


@dataclass
class Paper:
    paper_id: str  # e.g. "2401.12345v2"
    title: str
    abstract: str
    authors: list[str]
    published: str  # ISO date
    categories: list[str]
    pdf_url: str


def paper_from_result(result: arxiv.Result) -> Paper:
    return Paper(
        paper_id=result.get_short_id(),
        title=" ".join(result.title.split()),
        abstract=" ".join(result.summary.split()),
        authors=[a.name for a in result.authors],
        published=result.published.date().isoformat(),
        categories=list(result.categories),
        pdf_url=result.pdf_url,
    )


def pdf_filename(paper_id: str) -> str:
    # Old-style ids look like "cs/0112017v1"; the slash is not valid in a filename.
    return paper_id.replace("/", "_") + ".pdf"


def download_pdf(url: str, dest: Path, timeout: float = 60.0) -> None:
    """Stream a PDF to `dest` atomically: write to a .part file, then rename.

    A crash mid-download leaves only a .part file, never a truncated PDF that
    later looks like a valid cache hit.
    """
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    tmp.replace(dest)


def load_known_ids(metadata_path: Path) -> set[str]:
    if not metadata_path.exists():
        return set()
    with metadata_path.open(encoding="utf-8") as f:
        return {json.loads(line)["paper_id"] for line in f if line.strip()}


def ingest(query: str, max_results: int, data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, int]:
    pdf_dir = data_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = data_dir / "metadata.jsonl"
    known = load_known_ids(metadata_path)

    client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    with metadata_path.open("a", encoding="utf-8") as meta_file:
        for result in client.results(search):
            paper = paper_from_result(result)
            pdf_path = pdf_dir / pdf_filename(paper.paper_id)
            if paper.paper_id in known and pdf_path.exists():
                counts["skipped"] += 1
                continue
            try:
                download_pdf(paper.pdf_url, pdf_path)
            except requests.RequestException as exc:  # network errors, 404s, timeouts
                print(f"FAILED {paper.paper_id}: {exc}")
                pdf_path.with_suffix(".part").unlink(missing_ok=True)
                counts["failed"] += 1
                continue
            time.sleep(DOWNLOAD_DELAY_SECONDS)  # be polite to arxiv.org
            # Metadata is written only after the PDF is safely on disk.
            if paper.paper_id not in known:
                meta_file.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")
                meta_file.flush()
                known.add(paper.paper_id)
            counts["downloaded"] += 1
            print(f"ok {paper.paper_id}  {paper.title[:70]}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download arXiv papers into data/")
    parser.add_argument("--query", default="cat:cs.CL AND all:retrieval", help="arXiv search query")
    parser.add_argument("--max", type=int, default=20, dest="max_results")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    print(ingest(args.query, args.max_results, args.data_dir))


if __name__ == "__main__":
    main()
