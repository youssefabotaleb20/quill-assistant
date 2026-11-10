# Quill: AI Document Assistant

Upload a PDF or a CSV, ask questions in plain words, and get answers with page citations, data tables and charts. Built with Python, FastAPI, Pandas, NumPy and the Claude and OpenAI APIs.

**Live demo:** https://quill-assistant-chi.vercel.app (bring your own Anthropic or OpenAI API key)

- **Two providers:** paste an Anthropic key and Claude answers; paste an OpenAI key and an OpenAI model answers. The same tools, limits and front end serve both.
- **PDFs:** the model reads the whole document and points to the pages behind each answer (Claude with built-in citations, OpenAI with page references in the text). Ask for a chart and it plots figures taken from the document.
- **CSVs:** the model never sees the full table and never guesses numbers. It writes structured queries (filters, grouping, time buckets, aggregations) that the server validates and runs on the full dataset with Pandas, then explains the result and draws charts from it.

Try it with the built-in samples: a year of sales for a fictional coffee company and its annual report.

## How it works

```
Browser (static HTML/JS, served by the same app)
   │  keeps the file and the conversation; sends both with each question
   ▼
FastAPI (stateless; one serverless function on Vercel)
   ├─ documents.py         parse the upload (pypdf for PDFs, Pandas for CSVs), profile the columns
   ├─ assistant.py         the shared loop (stream, run tool calls, send results back) and the Claude provider
   ├─ openai_assistant.py  the OpenAI provider (Chat Completions)
   ├─ data_tools.py        validated query spec → Pandas/NumPy → result table or Chart.js spec
   └─ prompts.py           system prompts and tool definitions
   │  server-sent events: text, citations, tables, charts
   ▼
Browser renders Markdown, citations and interactive charts as they arrive
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the request flow, the security model and the reasons behind each decision.

### Design choices

- **Tools, not generated code.** For CSVs, the model calls `query_data` and `chart_data` with a JSON query spec. The spec is validated with Pydantic and checked against the real columns; there is no `eval`, so a prompt hidden in a cell can't run code. A bad spec returns a clear error to the model, which fixes the query and tries again.
- **Streaming.** Answers stream token by token over server-sent events, interleaved with tool activity, data tables and charts.
- **Citations.** With Claude, PDFs are sent as document blocks with citations enabled, so answers point to page numbers and quoted passages. OpenAI has no citations feature, so its prompt asks for page references in the text instead.
- **Prompt caching.** The document (or the CSV profile) sits at the start of a stable prompt prefix, and the conversation is append-only, so follow-up questions read the document from cache at a fraction of the cost. Claude's cache is marked explicitly; OpenAI caches a repeated prefix on its own.
- **Failure handling.** API errors, refusals, length limits and runaway tool loops each produce a clear message, and the conversation rolls back to its last good state. With Claude, server-side fallbacks retry a declined request on another model.
- **Bring your own key.** The public demo has no server-side API key. Visitors paste their own Anthropic or OpenAI key, which the browser keeps for that tab only and sends with each question (`X-Anthropic-Key` or `X-OpenAI-Key`, picked from the key's prefix); the server uses it for that request and never stores or logs it. Hosting on Vercel's free Hobby plan costs nothing and needs no card. Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` on the server if you'd rather pay for visitors' questions.
- **Stateless server.** Nothing is stored between requests. The browser holds the file and the conversation and sends both to `POST /api/ask`; the server re-reads the file, checks the history's shape and size, answers, and returns the updated history. For PDFs the server re-attaches the document block itself, so the history the browser keeps stays small. That is what lets it run as a serverless function.
- **Demo safety.** Uploads are size-, page- and row-limited and never written to disk. When the server's own key is used, questions are rate-limited per client so a public demo can't run up the bill.

## Run it locally

Needs Python 3.12+ and an [Anthropic](https://console.anthropic.com/settings/keys) or [OpenAI](https://platform.openai.com/api-keys) API key. Either export it as below, or leave it unset and paste it into the page.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY=sk-...; or put it in ../.env and add --env-file ../.env
uvicorn app.main:app --reload
```

Open http://localhost:8000. FastAPI serves the front end too, so there's nothing else to start.

Run the tests (no API key needed and nothing is spent; both APIs are replaced by scripted fakes):

```bash
cd backend
pytest -q
ruff check . && ruff format --check .
```

## Configuration

All settings are environment variables; see [`.env.example`](.env.example) and [`backend/app/config.py`](backend/app/config.py).

| Variable | Default | What it does |
|---|---|---|
| `ANTHROPIC_API_KEY` | (unset) | Server-side Claude key. With neither key set, visitors must bring their own |
| `OPENAI_API_KEY` | (unset) | Server-side OpenAI key, used only when `ANTHROPIC_API_KEY` isn't set |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Claude model used for answers |
| `CLAUDE_EFFORT` | `medium` | How hard Claude thinks: `low`, `medium`, `high` |
| `CLAUDE_FALLBACKS` | `true` | Retry a declined Claude request on another model server-side |
| `OPENAI_MODEL` | `gpt-5.4-mini` | OpenAI model used for answers |
| `OPENAI_EFFORT` | (unset) | OpenAI reasoning effort (`low`, `medium`, `high`); unset uses the model's default |
| `CORS_ORIGINS` | localhost | Front-end origins allowed to call the API |
| `QUESTIONS_PER_HOUR` | `30` | Per-client limit on questions answered with the server's key |
| `MAX_UPLOAD_MB` / `MAX_PDF_PAGES` / `MAX_CSV_ROWS` | `4` / `100` / `200000` | Upload limits (Vercel caps a request body at 4.5 MB) |
| `MAX_TURNS_PER_DOCUMENT` / `MAX_HISTORY_KB` | `30` / `200` | Limits on the conversation the browser sends back |

## Deploy

The public demo runs on [Vercel](https://vercel.com/docs/frameworks/backend/fastapi)'s free Hobby plan, which needs no card. [`index.py`](index.py) at the repo root exposes the FastAPI app, [`vercel.json`](vercel.json) gives the function up to 300 seconds for long answers, and the root [`requirements.txt`](requirements.txt) lists its packages.

1. On vercel.com, choose **Add New → Project** and import this GitHub repo.
2. Keep the detected FastAPI preset and the repo root as the root directory. No environment variables are needed, since visitors bring their own key.
3. Deploy. Every push to `main` redeploys.

The [`Dockerfile`](Dockerfile) runs the same app in one container, for any container host or local use.

**CI.** [`ci.yml`](.github/workflows/ci.yml) runs lint and tests, checks the Vercel entrypoint imports, and builds the Docker image and smoke-tests it, on every push and pull request.

## Project layout

```
backend/
  app/            FastAPI app, the answer loop with Claude and OpenAI providers, Pandas query engine, prompts
  tests/          pytest suite with scripted stand-ins for the Claude and OpenAI APIs
docs/             ARCHITECTURE.md
frontend/         static site: index.html, app.js, styles.css (Chart.js, marked, DOMPurify)
samples/          demo CSV and PDF, plus the script that generates them
index.py          Vercel entrypoint (exposes the FastAPI app)
vercel.json       Vercel function settings
Dockerfile        container image (API + front end), for other hosts
```

## License

MIT
