from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .auth import verify_api_key
from .ingestion import chunk_text, extract_text
from .rag import RAGService


app = FastAPI(title="Production AI Knowledge Assistant", version="1.0.0")
rag_service = RAGService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(file: UploadFile = File(...), chunk_size: int = Form(default=700)) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)

    try:
        text = extract_text(temp_path)
        chunks = chunk_text(text=text, source=file.filename or temp_path.name, chunk_size=chunk_size)
        inserted = rag_service.index_chunks(chunks)
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
