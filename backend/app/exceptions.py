from __future__ import annotations


class AppError(Exception):
    """Base class for domain/application errors."""


class IngestionError(AppError):
    """Raised when uploaded content cannot be parsed or chunked."""


class RetrievalError(AppError):
    """Raised when retrieval fails due to vector/sparse index issues."""


class GenerationError(AppError):
    """Raised when LLM generation fails."""
