from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .auth import verify_api_key
from .config import get_settings
from .exceptions import AppError, IngestionError
from .ingestion import chunk_text, extract_text
from .logging_utils import configure_logging
from .rag import RAGService

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title=settings.app_name, version=settings.app_version)
rag_service = RAGService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = False


@app.exception_handler(IngestionError)
async def ingestion_exception_handler(_: Request, exc: IngestionError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(AppError)
async def app_exception_handler(_: Request, exc: AppError) -> JSONResponse:
    logger.exception("Request failed")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(file: UploadFile = File(...), chunk_size: int = Form(default=700)) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    raw = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb}MB limit")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        temp_path = Path(tmp.name)

    try:
        text = extract_text(temp_path)
        chunks = chunk_text(text=text, source=file.filename or temp_path.name, chunk_size=chunk_size)
        if not chunks:
            raise IngestionError("No extractable text found in document")

        inserted = rag_service.index_chunks(chunks)
        logger.info("Indexed %s chunks from %s", inserted, file.filename)
    finally:
        temp_path.unlink(missing_ok=True)

    return JSONResponse(
        {
            "message": "Document indexed successfully",
            "chunks_indexed": inserted,
            "filename": file.filename,
        }
    )


@app.post("/query", dependencies=[Depends(verify_api_key)])
def query_documents(payload: QueryRequest):
    if payload.stream:
        return StreamingResponse(
            rag_service.stream_answer(question=payload.question, top_k=payload.top_k),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    result = rag_service.answer(question=payload.question, top_k=payload.top_k)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "from_cache": result.from_cache,
    }
