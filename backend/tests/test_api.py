from __future__ import annotations

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

from .conftest import FakeClient, text, tool_use


def make(script, **overrides):
    fake = FakeClient(script)
    settings = replace(Settings(), **overrides) if overrides else Settings()
    return TestClient(create_app(settings, client=fake)), fake


def events(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def upload(client, name, data):
    r = client.post("/api/documents", files={"file": (name, data)})
    assert r.status_code == 201, r.text
    return r.json()


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #


def test_upload_csv(sales_csv):
    client, _ = make([])
    doc = upload(client, "sales.csv", sales_csv)
    assert doc["kind"] == "csv"
    assert doc["rows"] == 2400
    assert {"name": "order_date", "dtype": "date"} in doc["columns"]
    assert len(doc["sample"]["rows"]) == 5


def test_upload_pdf(report_pdf):
    client, _ = make([])
    doc = upload(client, "report.pdf", report_pdf)
    assert doc["kind"] == "pdf"
    assert doc["pages"] == 2
    assert "Brightline" in doc["preview"]


@pytest.mark.parametrize(
    "name, data, message",
    [
        ("notes.docx", b"PK\x03\x04", "PDF or a CSV"),
        ("empty.csv", b"", "empty"),
        ("fake.pdf", b"hello", "doesn't look like a PDF"),
        ("broken.pdf", b"%PDF-1.4 garbage", "couldn't be read"),
    ],
)
def test_upload_rejects_bad_files(name, data, message):
    client, _ = make([])
    r = client.post("/api/documents", files={"file": (name, data)})
    assert r.status_code == 400
    assert message in r.json()["detail"]


def test_upload_size_limit():
    client, _ = make([], max_upload_mb=1)
    r = client.post("/api/documents", files={"file": ("big.csv", b"a,b\n" + b"1,2\n" * 300_000)})
    assert r.status_code == 400
    assert "larger than 1 MB" in r.json()["detail"]


def test_samples_are_an_allowlist():
    client, _ = make([])
    assert client.post("/api/samples/sales").json()["rows"] == 2400
    assert client.post("/api/samples/report").json()["pages"] == 2
    assert client.post("/api/samples/brightline-sales-2025.csv").status_code == 404
    assert client.post("/api/samples/..%2Fbackend%2Fapp%2Fmain.py").status_code >= 400


def test_frontend_is_served():
    client, _ = make([])
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>" in r.text


# --------------------------------------------------------------------------- #
# Asking questions
# --------------------------------------------------------------------------- #


def test_csv_question_runs_query_and_chart(sales_csv):
    query = {
        "group_by": ["region"],
        "metrics": [{"agg": "sum", "column": "revenue"}],
        "sort": [{"column": "sum_revenue"}],
    }
    client, fake = make(
        [
            (
                [
                    tool_use("query_data", query, id="t1"),
                    tool_use(
                        "chart_data", {"chart_type": "bar", "title": "Revenue by region", "query": query}, id="t2"
                    ),
                ],
                "tool_use",
            ),
            ([text("Cairo leads revenue.")], "end_turn"),
        ]
    )
    doc = upload(client, "sales.csv", sales_csv)
    out = events(client.post(f"/api/documents/{doc['id']}/ask", json={"question": "Which region sells most?"}))

    kinds = [e["type"] for e in out]
    assert kinds[0] == "status" and "table" in kinds and "chart" in kinds and kinds[-1] == "done"
    table = next(e for e in out if e["type"] == "table")
    assert table["rows"][0][0] == "Cairo"
    chart = next(e for e in out if e["type"] == "chart")["spec"]
    assert chart["labels"][0] == "Cairo"
    assert "".join(e["text"] for e in out if e["type"] == "text").strip() == "Cairo leads revenue."
    assert out[-1]["usage"]["input_tokens"] == 200

    # Second request carries both tool results, matched to their ids.
    second = fake.messages.requests[1]
    results = second["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert not any(r.get("is_error") for r in results)
    assert json.loads(results[0]["content"])["rows"][0][0] == "Cairo"

    # The CSV profile is in the system prompt and the request is cacheable.
    assert "order_date" in second["system"] and second["cache_control"] == {"type": "ephemeral"}
    assert {t["name"] for t in second["tools"]} == {"query_data", "chart_data"}
    assert second["fallbacks"] == "default"
    assert second["output_config"] == {"effort": "medium"}


def test_bad_tool_input_goes_back_to_claude_as_error(sales_csv):
    client, fake = make(
        [
            ([tool_use("query_data", {"group_by": ["nope"]})], "tool_use"),
            ([text("Fixed it.")], "end_turn"),
        ]
    )
    doc = upload(client, "sales.csv", sales_csv)
    out = events(client.post(f"/api/documents/{doc['id']}/ask", json={"question": "q"}))
    assert out[-1]["type"] == "done"
    result = fake.messages.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "unknown column" in result["content"]


def test_pdf_question_sends_document_once_with_citations(report_pdf):
    citation = {"cited_text": "Total revenue reached $241,820", "start_page_number": 1, "end_page_number": 2}
    client, fake = make(
        [
            ([text("Revenue was $241,820.", citations=[citation])], "end_turn"),
            ([text("Cairo.")], "end_turn"),
        ]
    )
    doc = upload(client, "report.pdf", report_pdf)
    out = events(client.post(f"/api/documents/{doc['id']}/ask", json={"question": "Total revenue?"}))
    cite = next(e for e in out if e["type"] == "citation")
    assert cite == {"type": "citation", "quote": "Total revenue reached $241,820", "pages": [1, 1]}

    events(client.post(f"/api/documents/{doc['id']}/ask", json={"question": "Top region?"}))
    first, second = fake.messages.requests
    block = first["messages"][0]["content"][0]
    assert block["type"] == "document" and block["citations"] == {"enabled": True}
    assert block["source"]["media_type"] == "application/pdf"
    # Same first message on the second turn (cache prefix intact), plus the new question.
    assert second["messages"][0] == first["messages"][0]
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
    assert second["messages"][-1]["content"] == "Top region?"
    assert {t["name"] for t in second["tools"]} == {"plot_values"}


def test_pdf_chart_from_values(report_pdf):
    client, _ = make(
        [
            (
                [
                    tool_use(
                        "plot_values",
                        {
                            "chart_type": "bar",
                            "title": "Quarterly revenue",
                            "labels": ["Q1", "Q2"],
                            "series": [{"name": "Revenue", "values": [47982, 54803]}],
                        },
                    )
                ],
                "tool_use",
            ),
            ([text("Here it is.")], "end_turn"),
        ]
    )
    doc = upload(client, "report.pdf", report_pdf)
    out = events(client.post(f"/api/documents/{doc['id']}/ask", json={"question": "Chart revenue by quarter"}))
    chart = next(e for e in out if e["type"] == "chart")["spec"]
    assert chart["labels"] == ["Q1", "Q2"] and chart["datasets"][0]["data"] == [47982, 54803]


def test_refusal_rolls_back_history(sales_csv):
    client, fake = make([([], "refusal"), ([text("ok")], "end_turn")])
    doc = upload(client, "sales.csv", sales_csv)
    out = events(client.post(f"/api/documents/{doc['id']}/ask", json={"question": "bad"}))
    assert out[-1]["type"] == "error" and "declined" in out[-1]["message"]
    events(client.post(f"/api/documents/{doc['id']}/ask", json={"question": "good"}))
    assert fake.messages.requests[1]["messages"] == [{"role": "user", "content": "good"}]


def test_tool_round_limit(sales_csv):
    loop = ([tool_use("query_data", {"metrics": [{"agg": "count"}]})], "tool_use")
    client, _ = make([loop, loop, loop], max_tool_rounds=2)
    doc = upload(client, "sales.csv", sales_csv)
    out = events(client.post(f"/api/documents/{doc['id']}/ask", json={"question": "q"}))
    assert out[-1]["type"] == "error" and "too many steps" in out[-1]["message"]


def test_fallbacks_can_be_turned_off(sales_csv):
    client, fake = make([([text("hi")], "end_turn")], use_fallbacks=False)
    doc = upload(client, "sales.csv", sales_csv)
    events(client.post(f"/api/documents/{doc['id']}/ask", json={"question": "q"}))
    assert "fallbacks" not in fake.messages.requests[0] and "betas" not in fake.messages.requests[0]


def test_question_validation_and_rate_limit(sales_csv):
    client, _ = make([([text("a")], "end_turn")], questions_per_hour=1, max_question_chars=10)
    doc = upload(client, "sales.csv", sales_csv)
    url = f"/api/documents/{doc['id']}/ask"
    assert client.post(url, json={"question": "   "}).status_code == 422
    assert client.post(url, json={"question": "x" * 11}).status_code == 422
    assert client.post(url, json={"question": "hi"}).status_code == 200
    assert client.post(url, json={"question": "again"}).status_code == 429


def test_unknown_document():
    client, _ = make([])
    assert client.post("/api/documents/nope/ask", json={"question": "q"}).status_code == 404


def test_reset_clears_conversation(sales_csv):
    client, fake = make([([text("a")], "end_turn"), ([text("b")], "end_turn")])
    doc = upload(client, "sales.csv", sales_csv)
    events(client.post(f"/api/documents/{doc['id']}/ask", json={"question": "one"}))
    assert client.post(f"/api/documents/{doc['id']}/reset").status_code == 204
    events(client.post(f"/api/documents/{doc['id']}/ask", json={"question": "two"}))
    assert fake.messages.requests[1]["messages"] == [{"role": "user", "content": "two"}]


def test_api_error_and_disconnect_roll_back(report_pdf):
    import anthropic
    import httpx2

    from app.assistant import Assistant
    from app.documents import load

    class Exploding:
        def __init__(self):
            self.messages = self
            self.beta = self

        def stream(self, **params):
            request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=request), body=None)

    doc = load("r.pdf", report_pdf, max_bytes=10**7, max_pages=100, max_rows=10)
    out = list(Assistant(Exploding(), Settings()).answer(doc, "q"))
    assert out == [{"type": "error", "message": "Claude is busy right now. Wait a moment and try again."}]
    assert doc.messages == []

    # The browser goes away after the first tool event: history is rolled back.
    fake = FakeClient(
        [
            (
                [
                    tool_use(
                        "plot_values",
                        {"chart_type": "bar", "title": "t", "labels": ["a"], "series": [{"name": "s", "values": [1]}]},
                    )
                ],
                "tool_use",
            )
        ]
    )
    gen = Assistant(fake, Settings()).answer(doc, "q")
    assert next(gen)["type"] == "chart"
    gen.close()
    assert doc.messages == []


# --------------------------------------------------------------------------- #
# Bring your own key
# --------------------------------------------------------------------------- #


def make_byok(script, **overrides):
    fake = FakeClient(script)
    keys: list[str] = []

    def factory(key):
        keys.append(key)
        return fake

    settings = replace(Settings(), server_key=False, **overrides)
    return TestClient(create_app(settings, client_factory=factory)), fake, keys


def test_without_server_key_a_visitor_key_is_required(sales_csv):
    client, _, keys = make_byok([])
    assert client.get("/api/health").json()["key_required"] is True
    doc = upload(client, "sales.csv", sales_csv)  # uploads and samples don't need a key
    r = client.post(f"/api/documents/{doc['id']}/ask", json={"question": "q"})
    assert r.status_code == 401 and "API key" in r.json()["detail"]
    r = client.post(f"/api/documents/{doc['id']}/ask", json={"question": "q"}, headers={"X-Anthropic-Key": "nope"})
    assert r.status_code == 401 and "sk-ant-" in r.json()["detail"]
    assert keys == []


def test_visitor_key_is_used_for_their_request_and_not_rate_limited(sales_csv):
    client, fake, keys = make_byok([([text("one")], "end_turn"), ([text("two")], "end_turn")], questions_per_hour=1)
    doc = upload(client, "sales.csv", sales_csv)
    headers = {"X-Anthropic-Key": "sk-ant-test-123"}
    for q in ("first", "second"):
        out = events(client.post(f"/api/documents/{doc['id']}/ask", json={"question": q}, headers=headers))
        assert out[-1]["type"] == "done"
    assert keys == ["sk-ant-test-123", "sk-ant-test-123"]
    assert len(fake.messages.requests) == 2


def test_server_key_reports_no_key_needed():
    client, _ = make([])
    assert client.get("/api/health").json()["key_required"] is False
