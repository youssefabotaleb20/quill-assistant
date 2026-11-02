# Quill: AI Document Assistant

Upload a PDF or a CSV, ask questions in plain words, and get answers with page citations, data tables and charts. Built with Python, FastAPI, Pandas, NumPy and the Claude API.

- **PDFs:** Claude reads the whole document and cites the pages and passages behind each answer. Ask for a chart and it plots figures taken from the document.
- **CSVs:** Claude never sees the full table and never guesses numbers. It writes structured queries (filters, grouping, time buckets, aggregations) that the server validates and runs on the full dataset with Pandas, then explains the result and draws charts from it.

Try it with the built-in samples: a year of sales for a fictional coffee company and its annual report.

## How it works

```
Browser (static HTML/JS, GitHub Pages)
   │  upload file / ask question
   ▼
FastAPI (Render)
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
- **Demo safety.** Uploads are size-, page- and row-limited, kept in memory for an hour and never written to disk. Questions are rate-limited per client so a public demo can't run up the API bill.

## Run it locally

Needs Python 3.12+ and an [Anthropic API key](https://console.anthropic.com/).

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
| `ANTHROPIC_API_KEY` | (required) | Claude API key |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Model used for answers |
| `CLAUDE_EFFORT` | `medium` | How hard the model thinks: `low`, `medium`, `high` |
| `CLAUDE_FALLBACKS` | `true` | Retry a declined request on another model server-side |
| `CORS_ORIGINS` | localhost | Front-end origins allowed to call the API |
| `QUESTIONS_PER_HOUR` | `30` | Per-client question limit |
| `MAX_UPLOAD_MB` / `MAX_PDF_PAGES` / `MAX_CSV_ROWS` | `10` / `100` / `200000` | Upload limits |
| `DOCUMENT_TTL_MINUTES` | `60` | How long an upload is kept in memory |

## Deploy

**API on Render.** In Render, choose New › Blueprint and pick this repo. [`render.yaml`](render.yaml) sets everything up; paste your `ANTHROPIC_API_KEY` when asked. The free plan sleeps when idle, so the first request after a while takes about a minute.

**Front end on GitHub Pages.** In the repo settings, set Pages › Source to "GitHub Actions", and add a repository variable `API_BASE_URL` with the Render URL (Settings › Secrets and variables › Actions › Variables). Every push to `main` that touches `frontend/` deploys the site via [`pages.yml`](.github/workflows/pages.yml).

**CI.** [`ci.yml`](.github/workflows/ci.yml) runs lint and the test suite on every push and pull request.

## Project layout

```
backend/
  app/            FastAPI app, Claude loop, Pandas query engine, prompts
  tests/          pytest suite with a scripted stand-in for the Claude API
frontend/         static site: index.html, app.js, styles.css (Chart.js, marked, DOMPurify)
samples/          demo CSV and PDF, plus the script that generates them
render.yaml       Render Blueprint for the API
```

## License

MIT
