# Quill: AI Document Assistant

Upload a PDF or a CSV, ask questions in plain words, and get answers with page citations, data tables and charts. Built with Python, FastAPI, Pandas, NumPy and the Claude API.

- **PDFs:** Claude reads the whole document and cites the pages and passages behind each answer. Ask for a chart and it plots figures taken from the document.
- **CSVs:** Claude never sees the full table and never guesses numbers. It writes structured queries (filters, grouping, time buckets, aggregations) that the server validates and runs on the full dataset with Pandas, then explains the result and draws charts from it.

Try it with the built-in samples: a year of sales for a fictional coffee company and its annual report.

## How it works

```
Browser (static HTML/JS, served by the same app)
   │  upload file / ask question
   ▼
FastAPI (Docker container on a free Hugging Face Space)
   ├─ documents.py   parse the upload (pypdf for PDFs, Pandas for CSVs), profile the columns
   ├─ assistant.py   the Claude loop: stream the answer, run tool calls, send results back
   ├─ data_tools.py  validated query spec → Pandas/NumPy → result table or Chart.js spec
   └─ prompts.py     system prompts and tool definitions
   │  server-sent events: text, citations, tables, charts
   ▼
Browser renders Markdown, citations and interactive charts as they arrive
```

### Design choices

- **Tools, not generated code.** For CSVs, Claude calls `query_data` and `chart_data` with a JSON query spec. The spec is validated with Pydantic and checked against the real columns; there is no `eval`, so a prompt hidden in a cell can't run code. A bad spec returns a clear error to Claude, which fixes the query and tries again.
- **Streaming.** Answers stream token by token over server-sent events, interleaved with tool activity, data tables and charts.
- **Citations.** PDFs are sent as document blocks with citations enabled, so answers point to page numbers and quoted passages.
- **Prompt caching.** The document (or the CSV profile) sits at the start of a stable prompt prefix, and the conversation is append-only, so follow-up questions read the document from cache at a fraction of the cost.
- **Failure handling.** API errors, refusals, length limits and runaway tool loops each produce a clear message, and the conversation rolls back to its last good state. Server-side fallbacks retry a declined request on another model.
- **Bring your own key.** The public demo has no server-side API key. Visitors paste their own Anthropic key, which the browser keeps for that tab only and sends with each question; the server uses it for that request and never stores or logs it. Hosting on a free Hugging Face Space costs nothing and needs no card. Set `ANTHROPIC_API_KEY` on the server if you'd rather pay for visitors' questions.
- **Demo safety.** Uploads are size-, page- and row-limited, kept in memory for an hour and never written to disk. When the server's own key is used, questions are rate-limited per client so a public demo can't run up the bill.

## Run it locally

Needs Python 3.12+ and an [Anthropic API key](https://console.anthropic.com/settings/keys). Either export it as below, or leave it unset and paste it into the page.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export ANTHROPIC_API_KEY=sk-ant-...   # or put it in ../.env and add --env-file ../.env
uvicorn app.main:app --reload
```

Open http://localhost:8000. FastAPI serves the front end too, so there's nothing else to start.

Run the tests (no API key needed; Claude is replaced by a scripted fake):

```bash
cd backend
pytest -q
ruff check . && ruff format --check .
```

## Configuration

All settings are environment variables; see [`.env.example`](.env.example) and [`backend/app/config.py`](backend/app/config.py).

| Variable | Default | What it does |
|---|---|---|
| `ANTHROPIC_API_KEY` | (unset) | Server-side Claude key. Unset means visitors must bring their own |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Model used for answers |
| `CLAUDE_EFFORT` | `medium` | How hard the model thinks: `low`, `medium`, `high` |
| `CLAUDE_FALLBACKS` | `true` | Retry a declined request on another model server-side |
| `CORS_ORIGINS` | localhost | Front-end origins allowed to call the API |
| `QUESTIONS_PER_HOUR` | `30` | Per-client limit on questions answered with the server's key |
| `MAX_UPLOAD_MB` / `MAX_PDF_PAGES` / `MAX_CSV_ROWS` | `10` / `100` / `200000` | Upload limits |
| `DOCUMENT_TTL_MINUTES` | `60` | How long an upload is kept in memory |

## Deploy

The app runs as one Docker container that serves both the API and the front end. The [`Dockerfile`](Dockerfile) works on any container host; this repo deploys it to a free [Hugging Face Space](https://huggingface.co/docs/hub/spaces-sdks-docker), which needs no card.

1. On Hugging Face, create a Space with the **Docker** SDK (blank template) and free CPU hardware.
2. Create a Hugging Face access token with **write** permission.
3. In this repo's settings, add a secret `HF_TOKEN` with the token, and a variable `HF_SPACE` with the Space's id, for example `your-hf-username/quill-assistant`.

From then on, every push to `main` runs [`space.yml`](.github/workflows/space.yml), which pushes a snapshot to the Space; the Space builds the image and serves the app. A free Space sleeps after a couple of days without visitors and wakes on the next visit.

**CI.** [`ci.yml`](.github/workflows/ci.yml) runs lint and tests, and builds the Docker image and smoke-tests it, on every push and pull request.

## Project layout

```
backend/
  app/            FastAPI app, Claude loop, Pandas query engine, prompts
  tests/          pytest suite with a scripted stand-in for the Claude API
frontend/         static site: index.html, app.js, styles.css (Chart.js, marked, DOMPurify)
samples/          demo CSV and PDF, plus the script that generates them
Dockerfile        the deploy image (API + front end)
```

## License

MIT
