"""HTTP API: check an upload, ask questions, stream answers.

The server is stateless. The browser keeps the file and the conversation and
sends both with each question, so any instance (or a fresh serverless
function) can answer it. Nothing a visitor uploads is ever stored.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import anthropic
import openai
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import documents
from .assistant import Assistant, ClaudeAssistant
from .config import Settings
from .config import settings as default_settings
from .openai_assistant import OpenAIAssistant
from .store import RateLimiter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"
SAMPLES_DIR = ROOT / "samples"
SAMPLES = {"sales": "brightline-sales-2025.csv", "report": "brightline-annual-report-2025.pdf"}
# A visitor's own key arrives in one of these headers, and the header picks the provider.
USER_KEY_HEADER = "X-Anthropic-Key"
OPENAI_KEY_HEADER = "X-OpenAI-Key"

# Content blocks the browser may send back in its copy of a Claude conversation.
# Documents are added by the server; anything else is refused.
HISTORY_BLOCKS = {"text", "tool_use", "tool_result", "thinking", "redacted_thinking", "fallback"}


def _user_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key, max_retries=1)


def _openai_user_client(api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key, max_retries=1)


def create_app(
    settings: Settings = default_settings,
    client: anthropic.Anthropic | None = None,
    client_factory: Callable[[str], anthropic.Anthropic] = _user_client,
    openai_client: openai.OpenAI | None = None,
    openai_client_factory: Callable[[str], openai.OpenAI] = _openai_user_client,
) -> FastAPI:
    """Build the app.

    `client` and `openai_client` are the server's own clients (tests pass
    fakes). Visitors can also send their own key in the X-Anthropic-Key or
    X-OpenAI-Key header; it's used for that request only and never stored or
    logged.
    """
    app = FastAPI(title="Quill: AI Document Assistant", version="1.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", USER_KEY_HEADER, OPENAI_KEY_HEADER],
    )

    limiter = RateLimiter(settings.questions_per_hour)
    # Which provider the server pays for itself, if any. Claude wins when both keys are set.
    if client is not None or settings.server_key:
        server_provider = "anthropic"
    elif openai_client is not None or settings.openai_server_key:
        server_provider = "openai"
    else:
        server_provider = None
    has_server_key = server_provider is not None
    state: dict[str, Any] = {"assistant": None}
    if client is not None:
        state["assistant"] = ClaudeAssistant(client, settings)
    elif openai_client is not None and server_provider == "openai":
        state["assistant"] = OpenAIAssistant(openai_client, settings)
    max_bytes = settings.max_upload_mb * 1024 * 1024

    def server_assistant() -> Assistant:
        if state["assistant"] is None:
            try:
                if server_provider == "openai":
                    state["assistant"] = OpenAIAssistant(openai.OpenAI(), settings)
                else:
                    state["assistant"] = ClaudeAssistant(anthropic.Anthropic(), settings)
            except (anthropic.AnthropicError, openai.OpenAIError) as exc:
                raise HTTPException(503, "The server has no API key configured.") from exc
        return state["assistant"]

    async def read_upload(file: UploadFile) -> documents.Document:
        data = await file.read(max_bytes + 1)
        try:
            return documents.load(
                file.filename or "upload",
                data,
                max_bytes=max_bytes,
                max_pages=settings.max_pdf_pages,
                max_rows=settings.max_csv_rows,
            )
        except documents.UploadError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        models = {"anthropic": settings.model, "openai": settings.openai_model}
        return {
            "status": "ok",
            # The model behind the server's own key, or Claude's when visitors bring a key.
            "model": models[server_provider or "anthropic"],
            "models": models,
            "key_required": not has_server_key,
        }

    @app.get("/api/limits")
    def limits() -> dict[str, Any]:
        return {
            "max_upload_mb": settings.max_upload_mb,
            "max_pdf_pages": settings.max_pdf_pages,
            "max_csv_rows": settings.max_csv_rows,
            "max_question_chars": settings.max_question_chars,
        }

    @app.post("/api/documents")
    async def inspect(file: UploadFile = File(...)) -> dict[str, Any]:
        """Check an upload and describe it. Nothing is kept."""
        return (await read_upload(file)).summary()

    @app.get("/api/samples/{name}")
    def sample(name: str) -> FileResponse:
        if name not in SAMPLES or not (SAMPLES_DIR / SAMPLES[name]).is_file():
            raise HTTPException(404, "Sample not found.")
        return FileResponse(SAMPLES_DIR / SAMPLES[name], filename=SAMPLES[name])

    @app.post("/api/ask")
    async def ask(
        request: Request,
        file: UploadFile = File(...),
        question: str = Form(...),
        history: str = Form("[]"),
    ) -> StreamingResponse:
        question = question.strip()
        if not question:
            raise HTTPException(422, "Ask a question.")
        if len(question) > settings.max_question_chars:
            raise HTTPException(422, f"Keep questions under {settings.max_question_chars} characters.")
        user_key = request.headers.get(USER_KEY_HEADER, "").strip()
        openai_key = request.headers.get(OPENAI_KEY_HEADER, "").strip()
        if user_key and openai_key:
            raise HTTPException(400, "Send one API key, not two.")
        provider = "anthropic" if user_key else "openai" if openai_key else server_provider

        messages = _parse_history(history, settings.max_history_kb * 1024, provider or "anthropic")
        if sum(1 for m in messages if m["role"] == "user" and isinstance(m["content"], str)) >= (
            settings.max_turns_per_document
        ):
            raise HTTPException(429, "This conversation is long. Start a new chat to keep going.")

        bot: Assistant
        if user_key:
            if not user_key.startswith("sk-ant-") or len(user_key) > 300:
                raise HTTPException(401, "That doesn't look like an Anthropic API key. It should start with sk-ant-.")
            bot = ClaudeAssistant(client_factory(user_key), settings)
        elif openai_key:
            if not openai_key.startswith("sk-") or openai_key.startswith("sk-ant-") or len(openai_key) > 300:
                raise HTTPException(401, "That doesn't look like an OpenAI API key. It should start with sk-.")
            bot = OpenAIAssistant(openai_client_factory(openai_key), settings)
        elif has_server_key:
            if not limiter.allow(_client_key(request)):
                raise HTTPException(429, "You've hit the hourly question limit for this demo. Try again later.")
            bot = server_assistant()
        else:
            raise HTTPException(401, "Add your Anthropic or OpenAI API key to ask questions.")

        doc = await read_upload(file)
        doc.messages = messages

        def events() -> Iterator[str]:
            for event in bot.answer(doc, question):
                yield f"data: {json.dumps(event, default=str)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if FRONTEND_DIR.is_dir():
        # Serve the front end too, so one process runs the whole app.
        # API routes are registered first, so they take precedence.
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return app


def _parse_history(raw: str, max_bytes: int, provider: str = "anthropic") -> list[dict[str, Any]]:
    """Validate the conversation the browser sent back.

    The two providers keep a conversation in different shapes, so a history
    made with one is refused by the other and the visitor starts a new chat.
    """
    if len(raw.encode()) > max_bytes:
        raise HTTPException(413, "This conversation is too long to continue. Start a new chat.")
    try:
        messages = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, "The conversation history is malformed. Start a new chat.") from exc

    bad = HTTPException(422, "The conversation history is malformed. Start a new chat.")
    if not isinstance(messages, list):
        raise bad
    if provider == "openai":
        if not _valid_openai_history(messages):
            raise bad
        return messages
    if len(messages) % 2:
        raise bad
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise bad
        if m["role"] != ("user" if i % 2 == 0 else "assistant"):
            raise bad
        content = m["content"]
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or not content:
            raise bad
        if not all(isinstance(b, dict) and b.get("type") in HISTORY_BLOCKS for b in content):
            raise bad
    if messages and not isinstance(messages[0]["content"], str):
        raise bad
    return messages


def _valid_openai_history(messages: list[Any]) -> bool:
    """Check a Chat Completions conversation: questions, answers, tool calls and their results, in order."""

    def strings(d: dict[str, Any], *keys: str) -> bool:
        return set(d) == set(keys) and all(isinstance(d[k], str) for k in keys)

    waiting: set[str] = set()  # tool calls that still need a result
    previous = "assistant"
    for m in messages:
        if not isinstance(m, dict):
            return False
        role = m.get("role")
        if role == "user":
            if previous != "assistant" or waiting or not strings(m, "role", "content"):
                return False
        elif role == "assistant":
            if previous == "assistant" or waiting or not isinstance(m.get("content"), str):
                return False
            if not set(m) <= {"role", "content", "tool_calls"}:
                return False
            calls = m.get("tool_calls", [])
            if not isinstance(calls, list) or ("tool_calls" in m and not calls):
                return False
            for c in calls:
                ok = (
                    isinstance(c, dict)
                    and set(c) == {"id", "type", "function"}
                    and isinstance(c["id"], str)
                    and c["type"] == "function"
                    and isinstance(c["function"], dict)
                    and strings(c["function"], "name", "arguments")
                )
                if not ok:
                    return False
                waiting.add(c["id"])
        elif role == "tool":
            if not strings(m, "role", "tool_call_id", "content") or m["tool_call_id"] not in waiting:
                return False
            waiting.discard(m["tool_call_id"])
        else:
            return False
        previous = role
    # A stored conversation always ends with a finished answer.
    return not messages or (previous == "assistant" and not waiting)


def _client_key(request: Request) -> str:
    # Hosting proxies (Vercel and most others) put the real client IP first in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


app = create_app()
