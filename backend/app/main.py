"""HTTP API: upload a document, ask questions, stream answers."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anthropic
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import documents
from .assistant import Assistant
from .config import Settings
from .config import settings as default_settings
from .store import DocumentStore, RateLimiter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"
SAMPLES = {"sales": "brightline-sales-2025.csv", "report": "brightline-annual-report-2025.pdf"}


class Question(BaseModel):
    question: str = Field(min_length=1)


def create_app(settings: Settings = default_settings, client: anthropic.Anthropic | None = None) -> FastAPI:
    app = FastAPI(title="Quill: AI Document Assistant", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    store = DocumentStore(settings.document_ttl_minutes * 60, settings.max_documents)
    limiter = RateLimiter(settings.questions_per_hour)
    state: dict[str, Any] = {"assistant": Assistant(client, settings) if client else None}

    def assistant() -> Assistant:
        if state["assistant"] is None:
            try:
                state["assistant"] = Assistant(anthropic.Anthropic(), settings)
            except anthropic.AnthropicError as exc:
                raise HTTPException(503, "The server has no Claude API key configured.") from exc
        return state["assistant"]

    def get_doc(doc_id: str) -> documents.Document:
        doc = store.get(doc_id)
        if doc is None:
            raise HTTPException(404, "Document not found. It may have expired; upload it again.")
        return doc

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": settings.model}

    @app.get("/api/limits")
    def limits() -> dict[str, Any]:
        return {
            "max_upload_mb": settings.max_upload_mb,
            "max_pdf_pages": settings.max_pdf_pages,
            "max_csv_rows": settings.max_csv_rows,
            "document_ttl_minutes": settings.document_ttl_minutes,
            "max_question_chars": settings.max_question_chars,
        }

    @app.post("/api/documents", status_code=201)
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)
        try:
            doc = documents.load(
                file.filename or "upload",
                data,
                max_bytes=max_bytes,
                max_pages=settings.max_pdf_pages,
                max_rows=settings.max_csv_rows,
            )
        except documents.UploadError as exc:
            raise HTTPException(400, str(exc)) from exc
        store.add(doc)
        return doc.summary()

    @app.post("/api/samples/{name}", status_code=201)
    def load_sample(name: str) -> dict[str, Any]:
        if name not in SAMPLES or not (SAMPLES_DIR / SAMPLES[name]).is_file():
            raise HTTPException(404, "Sample not found.")
        path = SAMPLES_DIR / SAMPLES[name]
        doc = documents.load(
            path.name,
            path.read_bytes(),
            max_bytes=settings.max_upload_mb * 1024 * 1024,
            max_pages=settings.max_pdf_pages,
            max_rows=settings.max_csv_rows,
        )
        store.add(doc)
        return doc.summary()

    @app.get("/api/documents/{doc_id}")
    def get_document(doc_id: str) -> dict[str, Any]:
        return get_doc(doc_id).summary()

    @app.delete("/api/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str) -> None:
        store.remove(doc_id)

    @app.post("/api/documents/{doc_id}/reset", status_code=204)
    def reset_conversation(doc_id: str) -> None:
        doc = get_doc(doc_id)
        lock = store.lock(doc_id)
        if lock is None or not lock.acquire(blocking=False):
            raise HTTPException(409, "Wait for the current answer to finish.")
        try:
            doc.messages.clear()
            doc.turns = 0
        finally:
            lock.release()

    @app.post("/api/documents/{doc_id}/ask")
    def ask(doc_id: str, body: Question, request: Request) -> StreamingResponse:
        doc = get_doc(doc_id)
        question = body.question.strip()
        if not question:
            raise HTTPException(422, "Ask a question.")
        if len(question) > settings.max_question_chars:
            raise HTTPException(422, f"Keep questions under {settings.max_question_chars} characters.")
        if doc.turns >= settings.max_turns_per_document:
            raise HTTPException(429, "This conversation is long. Start a new chat to keep going.")
        if not limiter.allow(_client_key(request)):
            raise HTTPException(429, "You've hit the hourly question limit for this demo. Try again later.")

        bot = assistant()
        lock = store.lock(doc_id)
        if lock is None or not lock.acquire(blocking=False):
            raise HTTPException(409, "Wait for the current answer to finish.")

        def events() -> Iterator[str]:
            try:
                for event in bot.answer(doc, question):
                    yield f"data: {json.dumps(event, default=str)}\n\n"
            finally:
                lock.release()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if FRONTEND_DIR.is_dir():
        # Serve the front end too, so one process runs the whole app locally.
        # API routes are registered first, so they take precedence.
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return app


def _client_key(request: Request) -> str:
    # Render and most hosts put the real client IP first in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


app = create_app()
