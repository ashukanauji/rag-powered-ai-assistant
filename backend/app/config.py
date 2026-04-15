from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment variables."""

    app_name: str = os.getenv("APP_NAME", "Production AI Knowledge Assistant")
    app_version: str = os.getenv("APP_VERSION", "1.1.0")
    cors_allow_origins: tuple[str, ...] = tuple(
        item.strip() for item in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if item.strip()
    )

    api_key: str = os.getenv("API_KEY", "")
    redis_url: str = os.getenv("REDIS_URL", "")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "backend/data/chroma")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "documents")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    corpus_file_path: str = os.getenv("CORPUS_FILE_PATH", "backend/data/corpus.jsonl")

    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "1800"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
