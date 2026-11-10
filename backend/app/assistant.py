"""The question-answering loop: Claude, tools, and a stream of UI events.

`answer()` is a generator of small dicts that the API layer sends to the
browser as server-sent events:

    {"type": "status",   "text": "Querying the data"}
    {"type": "text",     "text": "<markdown fragment>"}
    {"type": "citation", "quote": "...", "pages": [3, 3]}
    {"type": "table",    "title": "...", "columns": [...], "rows": [[...]]}
    {"type": "chart",    "spec": {...}}
    {"type": "error",    "message": "..."}
    {"type": "history",  "messages": [...]}   # the conversation so far, for the next question
    {"type": "done",     "usage": {...}}

The server keeps no state between requests. The browser holds the file and
the conversation and sends both with each question; that's what lets the app
run on serverless hosting.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from typing import Any

import anthropic
from pydantic import ValidationError

from . import data_tools, prompts
from .config import Settings
from .documents import Document

log = logging.getLogger(__name__)

TOOL_RESULT_ROWS = 100  # rows of a query result that go back to Claude
FALLBACK_BETA = "server-side-fallback-2026-07-01"

Event = dict[str, Any]


class Assistant:
    def __init__(self, client: anthropic.Anthropic, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    # ------------------------------------------------------------------ #

    def answer(self, doc: Document, question: str) -> Iterator[Event]:
        """Answer one question about `doc`, streaming events as they happen.

        The conversation is only kept if the turn completes. If anything goes
        wrong, including the browser disconnecting mid-answer, the history is
        rolled back to how it was before the question, so the next question
        starts from a valid conversation.
        """
        checkpoint = len(doc.messages)
        doc.messages.append({"role": "user", "content": question})
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        completed = False

        try:
            for _ in range(self.settings.max_tool_rounds + 1):
                response = yield from self._stream_round(doc)
                _add_usage(usage, response.usage)

                if response.stop_reason == "refusal":
                    raise _UserFacing("Claude declined to answer that question. Try rephrasing it.")

                tool_uses = [b for b in response.content if b.type == "tool_use"]
                if tool_uses and response.stop_reason == "max_tokens":
                    raise _UserFacing("The answer ran too long. Try asking a narrower question.")
                doc.messages.append({"role": "assistant", "content": response.content})

                if not tool_uses:
                    completed = True
                    if response.stop_reason == "max_tokens":
                        yield {"type": "text", "text": "\n\n_The answer was cut off because it hit the length limit._"}
                    yield {"type": "history", "messages": to_json(doc.messages)}
                    yield {"type": "done", "usage": usage}
                    return

                results = []
                for block in tool_uses:
                    events, result = self._run_tool(doc, block.name, block.input)
                    yield from events
                    results.append({"type": "tool_result", "tool_use_id": block.id, **result})
                doc.messages.append({"role": "user", "content": results})

            raise _UserFacing("That question needed too many steps. Try breaking it into smaller questions.")

        except Exception as exc:  # noqa: BLE001 - every failure becomes a message for the user
            yield {"type": "error", "message": _error_message(exc)}
        finally:
            if not completed:
                del doc.messages[checkpoint:]

    # ------------------------------------------------------------------ #

    def _wire_messages(self, doc: Document) -> list[dict[str, Any]]:
        """The conversation as sent to Claude.

        For a PDF, the document goes at the start of the first message on
        every request (the browser's copy of the history leaves it out). The
        bytes are identical each time, so after the first question it's read
        from the prompt cache.
        """
        if doc.kind != "pdf" or not doc.messages:
            return doc.messages
        assert doc.pdf_bytes is not None
        first = doc.messages[0]
        text = first["content"]
        blocks = [{"type": "text", "text": text}] if isinstance(text, str) else list(text)
        document = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(doc.pdf_bytes).decode("ascii"),
            },
            "title": doc.filename,
            "citations": {"enabled": True},
            "cache_control": {"type": "ephemeral"},
        }
        return [{"role": "user", "content": [document, *blocks]}, *doc.messages[1:]]

    def _request(self, doc: Document) -> dict[str, Any]:
        if doc.kind == "pdf":
            system, tools = prompts.PDF_SYSTEM, prompts.PDF_TOOLS
        else:
            assert doc.profile is not None
            system, tools = prompts.csv_system(doc.filename, doc.profile), prompts.CSV_TOOLS

        params: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "tools": tools,
            "messages": self._wire_messages(doc),
            "output_config": {"effort": self.settings.effort},
            # Cache the whole prefix up to the latest message, so each new
            # question only pays full price for what's new.
            "cache_control": {"type": "ephemeral"},
        }
        if self.settings.use_fallbacks:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def _stream_round(self, doc: Document) -> Iterator[Event]:
        """Stream one model response, yielding text and citations. Returns the final message."""
        params = self._request(doc)
        api = self.client.beta.messages if self.settings.use_fallbacks else self.client.messages

        for attempt in range(2):
            try:
                with api.stream(**params) as stream:
                    for event in stream:
                        if event.type != "content_block_delta":
                            continue
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield {"type": "text", "text": delta.text}
                        elif delta.type == "citations_delta":
                            yield _citation_event(delta.citation)
                    return stream.get_final_message()
            except ValueError:
                # The SDK couldn't parse a streamed tool input at all. No
                # tool_use block exists to answer, so re-issue the request once.
                if attempt == 1:
                    raise _UserFacing("Claude's tool call came back malformed. Please try again.") from None
        raise AssertionError("unreachable")

    def _run_tool(self, doc: Document, name: str, raw_input: Any) -> tuple[list[Event], dict[str, Any]]:
        """Run a tool call. Returns UI events and the tool_result fields for Claude."""
        if not isinstance(raw_input, dict):
            return [], _tool_error({"INVALID_JSON": json.dumps(raw_input, default=str)})
        try:
            if name == "query_data" and doc.df is not None:
                query = data_tools.Query.model_validate(raw_input)
                result = data_tools.run_query(doc.df, query)
                table = data_tools.table(result)
                events = [
                    {"type": "status", "text": "Queried the data"},
                    {"type": "table", "title": "Data used", **table},
                ]
                return events, {"content": _table_for_claude(table)}

            if name == "chart_data" and doc.df is not None:
                req = data_tools.ChartRequest.model_validate(raw_input)
                spec = data_tools.build_chart(doc.df, req)
                points = len(spec.get("labels") or spec["datasets"][0]["data"])
                summary = {"chart": "shown to the user", "points": points, "data": spec}
                return [{"type": "chart", "spec": spec}], {"content": json.dumps(summary, default=str)[:20000]}

            if name == "plot_values" and doc.kind == "pdf":
                spec = data_tools.values_chart(data_tools.ValuesChart.model_validate(raw_input))
                return [{"type": "chart", "spec": spec}], {"content": "Chart shown to the user."}

            return [], _tool_error(f"Unknown tool '{name}' for a {doc.kind} document.")
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5])
            return [], _tool_error(f"Invalid tool input: {problems}")
        except data_tools.QueryError as exc:
            return [], _tool_error(str(exc))


class _UserFacing(Exception):
    """An error whose message can be shown to the user as-is."""


def _error_message(exc: Exception) -> str:
    if isinstance(exc, _UserFacing):
        return str(exc)
    if isinstance(exc, anthropic.AuthenticationError):
        return "The Claude API key was rejected. Check that it's correct and still active."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "This Claude API key isn't allowed to use this model."
    if isinstance(exc, anthropic.RateLimitError):
        return "Claude is busy right now. Wait a moment and try again."
    if isinstance(exc, anthropic.BadRequestError):
        log.warning("Bad request to Claude: %s", exc.message)
        if "credit balance" in exc.message.lower():
            return "This Claude API key has no credit left. Add credit at console.anthropic.com or use another key."
        return "Claude couldn't process this request. Try a shorter question or a smaller file."
    if isinstance(exc, anthropic.APIStatusError):
        log.warning("Claude API error %s: %s", exc.status_code, exc.message)
        return "Claude returned an error. Please try again."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Couldn't reach Claude. Please try again."
    log.exception("Unexpected error while answering", exc_info=exc)
    return "Something went wrong. Please try again."


def to_json(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make a conversation JSON-safe, turning SDK content blocks into plain dicts."""
    out = []
    for m in messages:
        content = m["content"]
        if not isinstance(content, str):
            content = [b if isinstance(b, dict) else b.to_dict(mode="json", exclude_none=True) for b in content]
        out.append({"role": m["role"], "content": content})
    return out


def _tool_error(message: Any) -> dict[str, Any]:
    text = message if isinstance(message, str) else json.dumps(message)
    return {"content": text, "is_error": True}


def _table_for_claude(table: dict[str, Any]) -> str:
    rows = table["rows"]
    payload = {"columns": table["columns"], "rows": rows[:TOOL_RESULT_ROWS], "row_count": len(rows)}
    if len(rows) > TOOL_RESULT_ROWS:
        payload["note"] = f"Only the first {TOOL_RESULT_ROWS} rows are shown; aggregate or filter for the rest."
    return json.dumps(payload, default=str)


def _citation_event(citation: Any) -> Event:
    start = getattr(citation, "start_page_number", None)
    end = getattr(citation, "end_page_number", None)
    # end_page_number is exclusive in the API.
    pages = [start, max(start, end - 1)] if isinstance(start, int) and isinstance(end, int) else None
    return {"type": "citation", "quote": getattr(citation, "cited_text", ""), "pages": pages}


def _add_usage(total: dict[str, int], usage: Any) -> None:
    for key in total:
        total[key] += getattr(usage, key, None) or 0
