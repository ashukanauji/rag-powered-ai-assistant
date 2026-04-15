from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Generator
from dataclasses import dataclass

from groq import Groq
from rank_bm25 import BM25Okapi

from .cache import CacheClient
from .config import get_settings
from .database import CorpusStore, RetrievedChunk, VectorStore
from .exceptions import GenerationError, RetrievalError

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    answer: str
    sources: list[str]
    context_chunks: list[RetrievedChunk]
    from_cache: bool = False


class RAGService:
    """Coordinates retrieval + generation."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.vector_store = VectorStore()
        self.corpus_store = CorpusStore()
        self.cache = CacheClient()

        self._bm25_lock = threading.RLock()
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict] = []
        self._rebuild_bm25()

        self.groq_client: Groq | None = Groq(api_key=self.settings.groq_api_key) if self.settings.groq_api_key else None
        self.model = self.settings.groq_model

    def _rebuild_bm25(self) -> None:
        with self._bm25_lock:
            self._bm25_docs = self.corpus_store.load_chunks()
            if not self._bm25_docs:
                self._bm25 = None
                return
            tokenized = [d["text"].lower().split() for d in self._bm25_docs]
            self._bm25 = BM25Okapi(tokenized)
            logger.info("BM25 index rebuilt with %s chunks", len(self._bm25_docs))

    def index_chunks(self, chunks: list[dict]) -> int:
        inserted = self.vector_store.add_chunks(chunks)
        if inserted > 0:
            self.corpus_store.append_chunks(chunks)
            self._rebuild_bm25()
        return inserted

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        with self._bm25_lock:
            if not self._bm25:
                return []

            scores = self._bm25.get_scores(query.lower().split())
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
            docs = self._bm25_docs

        return [
            RetrievedChunk(
                chunk_id=f"bm25-{idx}",
                text=docs[idx]["text"],
                source=docs[idx].get("source", "unknown"),
                score=float(score),
            )
            for idx, score in ranked
        ]

    def _hybrid_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        try:
            dense = self.vector_store.query(query=query, top_k=top_k)
            sparse = self._bm25_search(query=query, top_k=top_k)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError("Hybrid retrieval failed") from exc

        merged: dict[str, RetrievedChunk] = {}
        for c in dense + sparse:
            key = f"{c.source}:{c.text[:120]}"
            if key not in merged or c.score > merged[key].score:
                merged[key] = c

        sorted_chunks = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return sorted_chunks[:top_k]

    @staticmethod
    def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
        context = "\n\n".join([f"[Source: {chunk.source}]\n{chunk.text}" for chunk in chunks])
        return (
            "You are a production AI knowledge assistant. Answer ONLY using the provided context. "
            "If context is insufficient, state that clearly. Always add a short source list at the end.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    @staticmethod
    def _cache_key(question: str, top_k: int) -> str:
        raw = f"q={question.strip().lower()}|k={top_k}"
        return "rag:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _generate_answer(self, prompt: str) -> str:
        if not self.groq_client:
            return "Groq API key is not configured. Retrieved context is available but generation is disabled."

        try:
            completion = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:
            logger.exception("Generation failed")
            raise GenerationError("Failed to generate response from LLM provider") from exc

    def answer(self, question: str, top_k: int = 5) -> RAGResult:
        key = self._cache_key(question, top_k)
        cached = self.cache.get(key)
        if cached:
            return RAGResult(
                answer=cached["answer"],
                sources=cached["sources"],
                context_chunks=[],
                from_cache=True,
            )

        chunks = self._hybrid_retrieve(question, top_k)
        if not chunks:
            return RAGResult(
                answer="I couldn't find relevant context in uploaded documents.",
                sources=[],
                context_chunks=[],
            )

        prompt = self._build_prompt(question, chunks)
        answer = self._generate_answer(prompt)

        sources = sorted({c.source for c in chunks})
        payload = {"answer": answer, "sources": sources}
        self.cache.set(key, payload, ttl_seconds=self.settings.cache_ttl_seconds)
        return RAGResult(answer=answer, sources=sources, context_chunks=chunks)

    def stream_answer(self, question: str, top_k: int = 5) -> Generator[str, None, None]:
        chunks = self._hybrid_retrieve(question, top_k)
        if not chunks:
            yield "data: I couldn't find relevant context in uploaded documents.\n\n"
            yield "data: [DONE]\n\n"
            return

        prompt = self._build_prompt(question, chunks)
        if not self.groq_client:
            yield "data: Groq API key is not configured; cannot stream generated response.\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            stream = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                stream=True,
            )
            aggregated: list[str] = []
            for part in stream:
                token = part.choices[0].delta.content or ""
                if token:
                    aggregated.append(token)
                    yield f"data: {token}\n\n"

            sources = sorted({c.source for c in chunks})
            yield f"data: Sources: {', '.join(sources)}\n\n"
            yield "data: [DONE]\n\n"

            key = self._cache_key(question, top_k)
            self.cache.set(
                key,
                {"answer": "".join(aggregated), "sources": sources},
                ttl_seconds=self.settings.cache_ttl_seconds,
            )
        except Exception as exc:
            logger.exception("Streaming generation failed")
            raise GenerationError("Failed to stream generated response") from exc
