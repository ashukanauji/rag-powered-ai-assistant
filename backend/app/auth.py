from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Simple API key authentication dependency.

    In production, prefer JWT/OIDC with role-based authorization.
    """

    expected = os.getenv("API_KEY")
    if not expected:
        # If API_KEY is not configured, auth is intentionally disabled for dev mode.
        return

    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
