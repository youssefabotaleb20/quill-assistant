from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


@pytest.fixture
def sales_csv() -> bytes:
    return (SAMPLES / "brightline-sales-2025.csv").read_bytes()


@pytest.fixture
def report_pdf() -> bytes:
    return (SAMPLES / "brightline-annual-report-2025.pdf").read_bytes()


# --------------------------------------------------------------------------- #
# A stand-in for the Anthropic client. Each scripted response is a list of
# content blocks plus a stop_reason; the fake streams text deltas for text
# blocks and records every request it receives.
# --------------------------------------------------------------------------- #


class Block(NS):
    """A content block that serialises like the SDK's models do."""

    def to_dict(self, mode: str = "python", exclude_none: bool = False) -> dict[str, Any]:
        out = {}
        for k, v in vars(self).items():
            if k == "citations":
                v = [vars(c) for c in v] or None
            if v is None and exclude_none:
                continue
            out[k] = v
        return out


def text(t: str, citations: list[dict[str, Any]] | None = None) -> Block:
    return Block(type="text", text=t, citations=[NS(type="page_location", **c) for c in citations or []])


def tool_use(name: str, input: Any, id: str = "toolu_1") -> Block:
    return Block(type="tool_use", name=name, input=input, id=id)


class FakeStream:
    def __init__(self, blocks: list[NS], stop_reason: str) -> None:
        self.blocks = blocks
        self.stop_reason = stop_reason

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def __iter__(self):
        for block in self.blocks:
            if block.type == "text":
                for c in block.citations:
                    yield NS(type="content_block_delta", delta=NS(type="citations_delta", citation=c))
                for word in block.text.split(" "):
                    yield NS(type="content_block_delta", delta=NS(type="text_delta", text=word + " "))

    def get_final_message(self) -> NS:
        usage = NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=50, cache_creation_input_tokens=0)
        return NS(content=self.blocks, stop_reason=self.stop_reason, usage=usage)


class FakeMessages:
    def __init__(self, script: list[tuple[list[NS], str]]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def stream(self, **params: Any) -> FakeStream:
        # Snapshot the messages: the real SDK serialises them at call time.
        self.requests.append({**params, "messages": list(params["messages"])})
        if not self.script:
            raise AssertionError("no scripted response left")
        blocks, stop = self.script.pop(0)
        return FakeStream(blocks, stop)


class FakeClient:
    def __init__(self, script: list[tuple[list[NS], str]]) -> None:
        self.messages = FakeMessages(script)
        self.beta = NS(messages=self.messages)


# --------------------------------------------------------------------------- #
# A stand-in for the OpenAI client. Each scripted response is a dict with
# optional "text", "tool_calls" ([(id, name, arguments-as-JSON-string)]),
# "finish" and "refusal". The fake streams it as Chat Completions chunks,
# splitting text and tool arguments into fragments the way the real API does.
# --------------------------------------------------------------------------- #


def _chunk(delta: NS | None = None, finish: str | None = None, usage: NS | None = None) -> NS:
    choices = [] if delta is None and finish is None else [NS(delta=delta, finish_reason=finish)]
    return NS(choices=choices, usage=usage)


class FakeCompletions:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def create(self, **params: Any):
        self.requests.append({**params, "messages": list(params["messages"])})
        if not self.script:
            raise AssertionError("no scripted response left")
        step = self.script.pop(0)
        calls = step.get("tool_calls", [])

        def stream():
            for word in step.get("text", "").split(" ") if step.get("text") else []:
                yield _chunk(NS(content=word + " ", refusal=None, tool_calls=None))
            if step.get("refusal"):
                yield _chunk(NS(content=None, refusal=step["refusal"], tool_calls=None))
            for index, (call_id, name, arguments) in enumerate(calls):
                half = len(arguments) // 2
                first = NS(index=index, id=call_id, function=NS(name=name, arguments=arguments[:half]))
                rest = NS(index=index, id=None, function=NS(name=None, arguments=arguments[half:]))
                yield _chunk(NS(content=None, refusal=None, tool_calls=[first]))
                yield _chunk(NS(content=None, refusal=None, tool_calls=[rest]))
            finish = step.get("finish") or ("tool_calls" if calls else "stop")
            yield _chunk(NS(content=None, refusal=None, tool_calls=None), finish=finish)
            details = NS(cached_tokens=40)
            yield _chunk(usage=NS(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=details))

        return stream()


class FakeOpenAI:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.completions = FakeCompletions(script)
        self.chat = NS(completions=self.completions)
