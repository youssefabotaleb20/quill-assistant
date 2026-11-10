"""Runtime settings, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model: str = field(default_factory=lambda: os.environ.get("CLAUDE_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: os.environ.get("CLAUDE_EFFORT", "medium"))
    max_tokens: int = field(default_factory=lambda: _int("CLAUDE_MAX_TOKENS", 32000))
    # Re-run a request on another model server-side if a safety classifier declines it.
    use_fallbacks: bool = field(default_factory=lambda: _bool("CLAUDE_FALLBACKS", True))
    max_tool_rounds: int = field(default_factory=lambda: _int("MAX_TOOL_ROUNDS", 8))

    # Vercel caps a request body at 4.5 MB, and each question sends the file.
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 4))
    max_pdf_pages: int = field(default_factory=lambda: _int("MAX_PDF_PAGES", 100))
    max_csv_rows: int = field(default_factory=lambda: _int("MAX_CSV_ROWS", 200_000))
    max_question_chars: int = field(default_factory=lambda: _int("MAX_QUESTION_CHARS", 2000))
    max_turns_per_document: int = field(default_factory=lambda: _int("MAX_TURNS_PER_DOCUMENT", 30))
    # Cap on the conversation history the browser sends back with a question.
    max_history_kb: int = field(default_factory=lambda: _int("MAX_HISTORY_KB", 200))

    # True when the server has its own Claude key. Without one, every visitor
    # must send their own key (bring your own key), so hosting costs nothing.
    server_key: bool = field(default_factory=lambda: bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()))

    # Simple per-IP limit so a public demo can't run up the API bill.
    # Only applies to questions answered with the server's own key.
    questions_per_hour: int = field(default_factory=lambda: _int("QUESTIONS_PER_HOUR", 30))

    # Comma-separated list of front-end origins allowed to call the API.
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip()
            for o in os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
            if o.strip()
        )
    )


settings = Settings()
