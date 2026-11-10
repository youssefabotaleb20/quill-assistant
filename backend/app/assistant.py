"""The question-answering loop: a model, tools, and a stream of UI events.

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

`Assistant` holds the loop and the tool handling, which are the same for
every model provider. A provider subclass only has to stream one model
response and say how its messages are shaped: `ClaudeAssistant` here, and
`OpenAIAssistant` in openai_assistant.py.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import anthropic
from pydantic import ValidationError

from . import data_tools, prompts
from .config import Settings
from .documents import Document

log = logging.getLogger(__name__)

TOOL_RESULT_ROWS = 100  # rows of a query result that go back to the model
FALLBACK_BETA = "server-side-fallback-2026-07-01"

Event = dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    input: Any  # the parsed arguments; anything but a dict is answered with an error


@dataclass
class Round:
    """One model response, in a shape every provider can produce."""

    stop: Literal["end", "length", "refusal"]
    message: dict[str, Any]  # the assistant message to add to the conversation
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class Assistant:
    """The provider-neutral loop. Subclasses talk to one model API."""

    name = "The model"  # how error messages refer to the provider
    sdk: Any = None  # the provider's SDK module, for its exception classes

    def __init__(self, client: Any, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    # What a provider implements ---------------------------------------- #

    def _stream_round(self, doc: Document) -> Iterator[Event]:
        """Stream one model response, yielding text and citations. Returns a Round."""
        raise NotImplementedError

    def _tool_messages(self, results: list[tuple[ToolCall, dict[str, Any]]]) -> list[dict[str, Any]]:
        """The message(s) that carry tool results back to the model."""
        raise NotImplementedError

    def _history(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The conversation as JSON for the browser to keep."""
        return messages

    def _credit_error(self, exc: Exception) -> str | None:
        """A message if `exc` means the key has run out of credit."""
        return None

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
                rnd = yield from self._stream_round(doc)
                for key in usage:
                    usage[key] += rnd.usage.get(key, 0)

                if rnd.stop == "refusal":
                    raise _UserFacing(f"{self.name} declined to answer that question. Try rephrasing it.")
                if rnd.tool_calls and rnd.stop == "length":
                    raise _UserFacing("The answer ran too long. Try asking a narrower question.")
                doc.messages.append(rnd.message)

                if not rnd.tool_calls:
                    completed = True
                    if rnd.stop == "length":
                        yield {"type": "text", "text": "\n\n_The answer was cut off because it hit the length limit._"}
                    yield {"type": "history", "messages": self._history(doc.messages)}
                    yield {"type": "done", "usage": usage}
                    return

                results = []
                for call in rnd.tool_calls:
                    events, result = self._run_tool(doc, call.name, call.input)
                    yield from events
                    results.append((call, result))
                doc.messages.extend(self._tool_messages(results))

            raise _UserFacing("That question needed too many steps. Try breaking it into smaller questions.")

        except Exception as exc:  # noqa: BLE001 - every failure becomes a message for the user
            yield {"type": "error", "message": self._error_message(exc)}
        finally:
            if not completed:
                del doc.messages[checkpoint:]

    # ------------------------------------------------------------------ #

    def _run_tool(self, doc: Document, name: str, raw_input: Any) -> tuple[list[Event], dict[str, Any]]:
        """Run a tool call. Returns UI events and the result for the model: `content`, plus `is_error` on failure."""
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
                return events, {"content": _table_for_model(table)}

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

    def _error_message(self, exc: Exception) -> str:
        """Turn any failure into a sentence for the user. The SDKs share their exception names."""
        name, sdk = self.name, self.sdk
        if isinstance(exc, _UserFacing):
            return str(exc)
        credit = self._credit_error(exc)
        if credit:
            return credit
        if isinstance(exc, sdk.AuthenticationError):
            return f"The {name} API key was rejected. Check that it's correct and still active."
        if isinstance(exc, sdk.PermissionDeniedError):
            return f"This {name} API key isn't allowed to use this model."
        if isinstance(exc, sdk.RateLimitError):
            return f"{name} is busy right now. Wait a moment and try again."
        if isinstance(exc, sdk.BadRequestError):
            log.warning("Bad request to %s: %s", name, exc.message)
            return f"{name} couldn't process this request. Try a shorter question or a smaller file."
        if isinstance(exc, sdk.APIStatusError):
            log.warning("%s API error %s: %s", name, exc.status_code, exc.message)
            return f"{name} returned an error. Please try again."
        if isinstance(exc, sdk.APIConnectionError):
            return f"Couldn't reach {name}. Please try again."
        log.exception("Unexpected error while answering", exc_info=exc)
        return "Something went wrong. Please try again."


class ClaudeAssistant(Assistant):
    """Claude through the Anthropic Messages API: native PDF input, citations and prompt caching."""

    name = "Claude"
    sdk = anthropic

    def _tool_messages(self, results: list[tuple[ToolCall, dict[str, Any]]]) -> list[dict[str, Any]]:
        blocks = [{"type": "tool_result", "tool_use_id": call.id, **result} for call, result in results]
        return [{"role": "user", "content": blocks}]

    def _history(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return to_json(messages)

    def _credit_error(self, exc: Exception) -> str | None:
        if isinstance(exc, anthropic.BadRequestError) and "credit balance" in exc.message.lower():
            return "This Claude API key has no credit left. Add credit at console.anthropic.com or use another key."
        return None

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
        """Stream one model response, yielding text and citations. Returns a Round."""
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
                    return _round(stream.get_final_message())
            except ValueError:
                # The SDK couldn't parse a streamed tool input at all. No
                # tool_use block exists to answer, so re-issue the request once.
                if attempt == 1:
                    raise _UserFacing("Claude's tool call came back malformed. Please try again.") from None
        raise AssertionError("unreachable")


class _UserFacing(Exception):
    """An error whose message can be shown to the user as-is."""


def _round(message: Any) -> Round:
    """A finished Claude message as a Round."""
    stop = {"refusal": "refusal", "max_tokens": "length"}.get(message.stop_reason, "end")
    calls = [ToolCall(b.id, b.name, b.input) for b in message.content if b.type == "tool_use"]
    usage = {
        key: getattr(message.usage, key, None) or 0
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    }
    return Round(stop, {"role": "assistant", "content": message.content}, calls, usage)


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


def _table_for_model(table: dict[str, Any]) -> str:
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
