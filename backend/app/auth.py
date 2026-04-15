from __future__ import annotations

from fastapi import Header, HTTPException, status

from .config import get_settings


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Simple API key authentication dependency.

    In production, prefer JWT/OIDC with role-based authorization.
    """

    expected = get_settings().api_key
    if not expected:
        # If API_KEY is not configured, auth is intentionally disabled for dev mode.
        return

    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
