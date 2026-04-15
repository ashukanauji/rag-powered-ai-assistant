from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    reader = PdfReader(str(file_path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def chunk_text(text: str, source: str, chunk_size: int = 700, overlap: int = 120) -> list[dict]:
    """Split long text into overlapping chunks.

    Overlap improves factual continuity across chunk boundaries.
    """

    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: list[dict] = []
    start = 0
    index = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        segment = clean[start:end]
        chunks.append({"text": segment, "source": source, "chunk_index": index})
        if end == len(clean):
            break
        start = max(0, end - overlap)
        index += 1
    return chunks
