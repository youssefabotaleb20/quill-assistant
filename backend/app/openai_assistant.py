"""The same assistant on OpenAI's Chat Completions API.

The loop and the tools are shared with Claude (see assistant.py). What
differs is the wire format:

- Tools are function tools, and a tool call's arguments arrive as a JSON
  string that is parsed here.
- A PDF goes in as a file part on the first message. OpenAI has no
  citations feature, so the prompt asks for page references in the text.
- Prompt caching is automatic for a repeated prefix; there is nothing to mark.
- Each tool result is its own `tool` message.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from typing import Any

import openai

from . import prompts
from .assistant import Assistant, Event, Round, ToolCall
from .documents import Document


class OpenAIAssistant(Assistant):
    name = "OpenAI"
    sdk = openai

    def _tool_messages(self, results: list[tuple[ToolCall, dict[str, Any]]]) -> list[dict[str, Any]]:
        return [
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": ("Error: " if result.get("is_error") else "") + result["content"],
            }
            for call, result in results
        ]

    def _credit_error(self, exc: Exception) -> str | None:
        # An empty balance comes back as a 429 with this code, not as a rate limit to wait out.
        if isinstance(exc, openai.RateLimitError) and getattr(exc, "code", None) == "insufficient_quota":
            return "This OpenAI API key has no credit left. Add credit at platform.openai.com or use another key."
        return None

    # ------------------------------------------------------------------ #

    def _wire_messages(self, doc: Document) -> list[dict[str, Any]]:
        """The conversation as sent to OpenAI: system prompt first, and the PDF on the first question."""
        if doc.kind == "pdf":
            system = prompts.PDF_SYSTEM + prompts.PDF_PAGE_REFERENCES
        else:
            assert doc.profile is not None
            system = prompts.csv_system(doc.filename, doc.profile)

        messages = list(doc.messages)
        if doc.kind == "pdf" and messages:
            assert doc.pdf_bytes is not None
            data = base64.standard_b64encode(doc.pdf_bytes).decode("ascii")
            file_part = {
                "type": "file",
                "file": {"filename": doc.filename, "file_data": f"data:application/pdf;base64,{data}"},
            }
            messages[0] = {"role": "user", "content": [file_part, {"type": "text", "text": messages[0]["content"]}]}
        return [{"role": "system", "content": system}, *messages]

    def _request(self, doc: Document) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.settings.openai_model,
            "messages": self._wire_messages(doc),
            "tools": prompts.openai_tools(prompts.PDF_TOOLS if doc.kind == "pdf" else prompts.CSV_TOOLS),
            "max_completion_tokens": self.settings.openai_max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.settings.openai_effort:
            params["reasoning_effort"] = self.settings.openai_effort
        return params

    def _stream_round(self, doc: Document) -> Iterator[Event]:
        """Stream one model response, yielding text. Returns a Round."""
        text: list[str] = []
        refusal: list[str] = []
        calls: dict[int, dict[str, str]] = {}  # tool calls arrive in fragments, keyed by index
        finish = None
        usage: dict[str, int] = {}

        for chunk in self.client.chat.completions.create(**self._request(doc)):
            if getattr(chunk, "usage", None):
                usage = _usage(chunk.usage)
            for choice in chunk.choices:
                finish = choice.finish_reason or finish
                delta = choice.delta
                if delta is None:
                    continue
                if getattr(delta, "content", None):
                    text.append(delta.content)
                    yield {"type": "text", "text": delta.content}
                if getattr(delta, "refusal", None):
                    refusal.append(delta.refusal)
                for part in getattr(delta, "tool_calls", None) or []:
                    call = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                    call["id"] = part.id or call["id"]
                    if part.function:
                        call["name"] = part.function.name or call["name"]
                        call["arguments"] += part.function.arguments or ""

        ordered = [calls[i] for i in sorted(calls)]
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text)}
        if ordered:
            message["tool_calls"] = [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in ordered
            ]

        if refusal or finish == "content_filter":
            stop = "refusal"
        elif finish == "length":
            stop = "length"
        else:
            stop = "end"
        tool_calls = [ToolCall(c["id"], c["name"], _arguments(c["arguments"])) for c in ordered]
        return Round(stop, message, tool_calls, usage)


def _arguments(raw: str) -> Any:
    """Parse a tool call's arguments. A string that isn't JSON is returned as is, and the tool answers with an error."""
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw


def _usage(usage: Any) -> dict[str, int]:
    """OpenAI's token counts under the names the UI already uses. Cached tokens are part of prompt_tokens."""
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None) or 0
    return {
        "input_tokens": max((usage.prompt_tokens or 0) - cached, 0),
        "output_tokens": usage.completion_tokens or 0,
        "cache_read_input_tokens": cached,
    }
