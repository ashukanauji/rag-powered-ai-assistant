from __future__ import annotations

import hashlib
import os
from collections.abc import Generator
from dataclasses import dataclass

from groq import Groq
from rank_bm25 import BM25Okapi

from .cache import CacheClient
from .database import CorpusStore, RetrievedChunk, VectorStore


@dataclass
class RAGResult:
    answer: str
    sources: list[str]
    context_chunks: list[RetrievedChunk]
    from_cache: bool = False


class RAGService:
    """Coordinates retrieval + generation.

    Why RAG: it grounds LLM output in your data, reducing hallucination and
    enabling citation. It also avoids expensive fine-tuning for changing docs.
    """

    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.corpus_store = CorpusStore()
        self.cache = CacheClient()

        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict] = []
        self._rebuild_bm25()

        api_key = os.getenv("GROQ_API_KEY")
        self.groq_client: Groq | None = Groq(api_key=api_key) if api_key else None
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def _rebuild_bm25(self) -> None:
        self._bm25_docs = self.corpus_store.load_chunks()
        if not self._bm25_docs:
            self._bm25 = None
            return
        tokenized = [d["text"].lower().split() for d in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)

    def index_chunks(self, chunks: list[dict]) -> int:
        inserted = self.vector_store.add_chunks(chunks)
        if inserted > 0:
            self.corpus_store.append_chunks(chunks)
            self._rebuild_bm25()
        return inserted

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self._bm25:
            return []

        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[RetrievedChunk] = []
        for idx, score in ranked:
            doc = self._bm25_docs[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=f"bm25-{idx}",
                    text=doc["text"],
                    source=doc.get("source", "unknown"),
                    score=float(score),
                )
            )
        return results

    def _hybrid_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        dense = self.vector_store.query(query=query, top_k=top_k)
        sparse = self._bm25_search(query=query, top_k=top_k)

        merged: dict[str, RetrievedChunk] = {}
        for c in dense + sparse:
            key = f"{c.source}:{c.text[:120]}"
            if key not in merged or c.score > merged[key].score:
                merged[key] = c

        sorted_chunks = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return sorted_chunks[:top_k]

    @staticmethod
    def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
        context = "\n\n".join(
            [f"[Source: {chunk.source}]\n{chunk.text}" for chunk in chunks]
        )
        return (
            "You are a production AI knowledge assistant. Answer ONLY using the provided context. "
            "If context is insufficient, state that clearly. Always add a short source list at the end.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    @staticmethod
    def _cache_key(question: str, top_k: int) -> str:
        raw = f"q={question}|k={top_k}"
        return "rag:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

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
        if self.groq_client:
            completion = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            answer = completion.choices[0].message.content or ""
        else:
            answer = "Groq API key is not configured. Retrieved context is available but generation is disabled."

        sources = sorted({c.source for c in chunks})
        payload = {"answer": answer, "sources": sources}
        self.cache.set(key, payload)
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

        stream = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            stream=True,
        )
        aggregated = []
        for part in stream:
            token = part.choices[0].delta.content or ""
            if token:
                aggregated.append(token)
                yield f"data: {token}\n\n"

        sources = sorted({c.source for c in chunks})
        yield f"data: \n\nSources: {', '.join(sources)}\n\n"
        yield "data: [DONE]\n\n"

        key = self._cache_key(question, top_k)
        self.cache.set(key, {"answer": "".join(aggregated), "sources": sources})
