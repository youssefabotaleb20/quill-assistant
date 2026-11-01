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


def text(t: str, citations: list[dict[str, Any]] | None = None) -> NS:
    return NS(type="text", text=t, citations=[NS(**c) for c in citations or []])


def tool_use(name: str, input: Any, id: str = "toolu_1") -> NS:
    return NS(type="tool_use", name=name, input=input, id=id)


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
