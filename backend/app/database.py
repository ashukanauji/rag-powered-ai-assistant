from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    """Represents a chunk returned by dense retrieval."""

    chunk_id: str
    text: str
    source: str
    score: float


class VectorStore:
    """Persistent ChromaDB wrapper.

    Why this abstraction: keeps all DB concerns in one place and allows
    swapping ChromaDB for FAISS in the future without changing API routes.
    """

    def __init__(
        self,
        persist_dir: str = "backend/data/chroma",
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.encoder = SentenceTransformer(embedding_model)

    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0

        ids = [str(uuid.uuid4()) for _ in chunks]
        docs = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": c.get("source", "unknown"),
                "chunk_index": c.get("chunk_index", 0),
            }
            for c in chunks
        ]
        embeddings = self.encoder.encode(docs, convert_to_numpy=True).tolist()
        self.collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
        return len(ids)

    def query(self, query: str, top_k: int = 5, source_filter: str | None = None) -> list[RetrievedChunk]:
        where = {"source": source_filter} if source_filter else None
        query_embedding = self.encoder.encode([query], convert_to_numpy=True).tolist()[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[RetrievedChunk] = []
        for idx, text in enumerate(docs):
            metadata = metas[idx] if idx < len(metas) else {}
            distance = float(distances[idx]) if idx < len(distances) else 0.0
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"dense-{idx}",
                    text=text,
                    source=metadata.get("source", "unknown"),
                    score=1.0 / (1.0 + distance),
                )
            )
        return chunks


class CorpusStore:
    """JSONL-based corpus for BM25 retrieval index rebuild.

    Tradeoff: simple and transparent persistence; slower than a dedicated
    search engine for very large corpora, but enough for SMB workloads.
    """

    def __init__(self, file_path: str = "backend/data/corpus.jsonl") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("", encoding="utf-8")

    def append_chunks(self, chunks: list[dict[str, Any]]) -> None:
        with self.file_path.open("a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def load_chunks(self) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        with self.file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(json.loads(line))
        return chunks
