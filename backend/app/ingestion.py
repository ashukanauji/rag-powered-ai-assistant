from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from .exceptions import IngestionError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type: {suffix}")

    try:
        if suffix == ".txt":
            return file_path.read_text(encoding="utf-8", errors="ignore")

        reader = PdfReader(str(file_path))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    except Exception as exc:
        logger.exception("Failed to extract text from %s", file_path)
        raise IngestionError("Failed to extract text from uploaded file") from exc


def chunk_text(text: str, source: str, chunk_size: int = 700, overlap: int = 120) -> list[dict]:
    """Split long text into overlapping chunks."""
    if chunk_size <= 0:
        raise IngestionError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise IngestionError("overlap must be >= 0 and < chunk_size")

    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: list[dict] = []
    start = 0
    index = 0
    text_len = len(clean)
    while start < text_len:
        end = min(text_len, start + chunk_size)
        segment = clean[start:end]
        chunks.append({"text": segment, "source": source, "chunk_index": index})
        if end == text_len:
            break
        start = end - overlap
        index += 1
    return chunks
