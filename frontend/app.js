"use strict";

const API = (window.APP_CONFIG && window.APP_CONFIG.apiBase || "").replace(/\/$/, "");

const SUGGESTIONS = {
  csv: [
    "Which region brings in the most revenue?",
    "Chart monthly revenue over the year",
    "How does average order value differ by channel?",
    "Is there a relationship between units and revenue?",
  ],
  pdf: [
    "Summarize this document in five bullet points",
    "What are the main risks mentioned?",
    "Chart revenue by quarter",
    "What are the plans for next year?",
  ],
};

const $ = (id) => document.getElementById(id);
// The server keeps nothing between requests: the browser holds the file and
// the conversation, and sends both with each question.
const state = { doc: null, file: null, history: [], busy: false, keyRequired: false };
const KEY_STORE = "quill.apiKey";

function getKey() {
  try { return sessionStorage.getItem(KEY_STORE) || ""; } catch { return state.memoryKey || ""; }
}

function setKey(value) {
  state.memoryKey = value;
  try { value ? sessionStorage.setItem(KEY_STORE, value) : sessionStorage.removeItem(KEY_STORE); } catch { /* private mode */ }
}

function refreshKeyPanel() {
  const panel = $("key-panel");
  panel.hidden = !state.keyRequired;
  const key = getKey();
  panel.classList.toggle("missing", state.keyRequired && !key);
  $("key-status").textContent = key ? `Saved for this tab (…${key.slice(-4)}).` : "";
  if (key) $("api-key").value = "";
}

// ---------------------------------------------------------------- helpers

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

function md(text) {
  return DOMPurify.sanitize(marked.parse(text, { breaks: false, gfm: true }));
}

function formatBytes(n) {
  return n < 1024 * 1024 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function formatCell(v) {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return String(v);
}

async function api(path, options = {}) {
  const res = await fetch(API + path, options);
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : body.detail[0]?.msg || detail;
    } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

// ---------------------------------------------------------------- start screen

function setUploadStatus(text, isError = false) {
  const s = $("upload-status");
  s.textContent = text;
  s.classList.toggle("error", isError);
}

async function uploadFile(file) {
  if (!file) return;
  if (state.maxBytes && file.size > state.maxBytes) {
    setUploadStatus(`That file is larger than ${state.maxBytes / 1024 / 1024} MB.`, true);
    return;
  }
  setUploadStatus(`Reading ${file.name}…`);
  const form = new FormData();
  form.append("file", file);
  try {
    openDocument(await api("/api/documents", { method: "POST", body: form }), file);
    setUploadStatus("");
  } catch (err) {
    setUploadStatus(err.message, true);
  }
}

async function loadSample(name, button) {
  document.querySelectorAll("[data-sample]").forEach((b) => (b.disabled = true));
  setUploadStatus("Loading the sample…");
  try {
    const res = await fetch(`${API}/api/samples/${name}`);
    if (!res.ok) throw new Error("Couldn't load the sample.");
    const fileName = decodeURIComponent((res.headers.get("content-disposition") || "").split("filename=")[1] || name).replace(/"/g, "");
    const file = new File([await res.blob()], fileName, { type: res.headers.get("content-type") || "" });
    await uploadFile(file);
  } catch (err) {
    setUploadStatus(err.message, true);
  } finally {
    document.querySelectorAll("[data-sample]").forEach((b) => (b.disabled = false));
  }
}

function showStart() {
  state.doc = null;
  state.file = null;
  state.history = [];
  $("workspace").hidden = true;
  $("start").hidden = false;
  $("file-input").value = "";
}

// ---------------------------------------------------------------- workspace

function openDocument(doc, file) {
  state.doc = doc;
  state.file = file;
  state.history = [];
  $("start").hidden = true;
  $("workspace").hidden = false;

  $("doc-kind").textContent = doc.kind.toUpperCase();
  $("doc-name").textContent = doc.filename;
  const details = $("doc-details");
  details.replaceChildren();

  if (doc.kind === "csv") {
    $("doc-meta").textContent = `${doc.rows.toLocaleString()} rows · ${doc.columns.length} columns · ${formatBytes(doc.size_bytes)}`;
    details.append(el("div", { class: "columns" },
      ...doc.columns.map((c) => el("span", { class: "col", title: c.dtype }, c.name, el("small", { text: c.dtype })))));
  } else {
    $("doc-meta").textContent = `${doc.pages} page${doc.pages === 1 ? "" : "s"} · ${formatBytes(doc.size_bytes)}`;
    if (doc.preview) details.append(el("p", { class: "preview", text: doc.preview }));
  }

  const list = $("suggestions");
  list.replaceChildren(...SUGGESTIONS[doc.kind].map((q) => {
    const b = el("button", { class: "suggestion", type: "button", text: q });
    b.addEventListener("click", () => ask(q));
    return b;
  }));

  resetMessages();
  refreshKeyPanel();
  (state.keyRequired && !getKey() ? $("api-key") : $("question")).focus();
}

function resetMessages() {
  $("messages").replaceChildren(el("div", { class: "empty-chat" },
    el("strong", { text: "Ready when you are." }),
    state.doc.kind === "csv"
      ? "Ask about totals, trends, rankings or comparisons. Ask for a chart any time."
      : "Ask for a summary, a specific fact, or a chart of figures in the document."));
}

function scrollToEnd() {
  const m = $("messages");
  m.scrollTop = m.scrollHeight;
}

function setBusy(busy) {
  state.busy = busy;
  $("send").disabled = busy;
  $("new-chat").disabled = busy;
  document.querySelectorAll(".suggestion").forEach((b) => (b.disabled = busy));
}

// ---------------------------------------------------------------- asking

async function ask(question) {
  question = question.trim();
  if (!question || state.busy || !state.doc) return;
  if (state.keyRequired && !getKey()) {
    refreshKeyPanel();
    $("key-status").textContent = "Add your API key first.";
    $("api-key").focus();
    return;
  }
  $("question").value = "";
  autosize();
  $("messages").querySelector(".empty-chat")?.remove();

  $("messages").append(el("div", { class: "msg-user" }, el("p", { text: question })));
  const view = new AnswerView();
  $("messages").append(view.root);
  scrollToEnd();
  setBusy(true);

  try {
    const form = new FormData();
    form.append("file", state.file);
    form.append("question", question);
    form.append("history", JSON.stringify(state.history));
    const res = await fetch(`${API}/api/ask`, {
      method: "POST",
      headers: getKey() ? { "X-Anthropic-Key": getKey() } : {},
      body: form,
    });
    if (!res.ok) {
      let detail = `Request failed (${res.status})`;
      try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
      if (res.status === 401) { setKey(""); refreshKeyPanel(); $("api-key").focus(); }
      view.error(detail);
      return;
    }
    await readEvents(res.body, (event) => view.handle(event));
  } catch (err) {
    view.error("Lost the connection to the server. Please try again.");
  } finally {
    view.finish();
    setBusy(false);
    $("question").focus();
  }
}

async function readEvents(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) {
          try { onEvent(JSON.parse(line.slice(6))); } catch { /* skip malformed */ }
        }
      }
    }
  }
}

class AnswerView {
  constructor() {
    this.text = "";
    this.sources = [];
    this.root = el("div", { class: "msg-bot" });
    this.statusLine = el("div", { class: "status-line" });
    this.answer = el("div", { class: "answer" }, el("div", { class: "typing", "aria-label": "Thinking" }, el("span"), el("span"), el("span")));
    this.extras = el("div", { class: "msg-bot" });
    this.root.append(this.statusLine, this.answer, this.extras);
    this.pending = false;
  }

  handle(e) {
    switch (e.type) {
      case "text":
        this.text += e.text;
        this.render();
        break;
      case "status":
        this.statusLine.append(el("span", { class: "status-pill", text: e.text }));
        break;
      case "table":
        this.extras.append(dataTable(e));
        break;
      case "chart":
        this.extras.prepend(chartCard(e.spec));
        break;
      case "citation":
        this.addSource(e);
        break;
      case "error":
        this.error(e.message);
        break;
      case "history":
        state.history = e.messages;
        break;
      case "done":
        if (e.usage) {
          const u = e.usage;
          const cached = u.cache_read_input_tokens ? ` · ${u.cache_read_input_tokens.toLocaleString()} cached` : "";
          this.usage = el("div", { class: "usage", text: `${(u.input_tokens + u.cache_read_input_tokens).toLocaleString()} in${cached} · ${u.output_tokens.toLocaleString()} out` });
        }
        break;
    }
    scrollToEnd();
  }

  render() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      this.answer.innerHTML = md(this.text);
      scrollToEnd();
    });
  }

  addSource(c) {
    const key = `${c.pages}|${c.quote}`;
    if (this.sources.some((s) => s.key === key)) return;
    this.sources.push({ key, ...c });
  }

  error(message) {
    this.errored = true;
    this.extras.append(el("div", { class: "error-note", role: "alert", text: message }));
  }

  finish() {
    if (!this.text) this.answer.remove();
    else this.answer.innerHTML = md(this.text);
    if (!this.statusLine.childElementCount) this.statusLine.remove();
    if (this.sources.length) {
      const list = el("ol");
      for (const s of this.sources) {
        const page = s.pages ? (s.pages[0] === s.pages[1] ? `p. ${s.pages[0]}` : `pp. ${s.pages[0]}–${s.pages[1]}`) : "";
        list.append(el("li", {}, page && el("span", { class: "page", text: page }), el("q", { text: s.quote.trim() })));
      }
      this.root.append(el("div", { class: "sources" }, el("h4", { text: "Sources" }), list));
    }
    if (this.usage) this.root.append(this.usage);
  }
}

// ---------------------------------------------------------------- tables and charts

function dataTable(t) {
  const numeric = t.columns.map((_, i) => t.rows.length > 0 && t.rows.every((r) => r[i] === null || typeof r[i] === "number"));
  const head = el("tr", {}, ...t.columns.map((c) => el("th", { text: c })));
  const body = t.rows.map((r) => el("tr", {}, ...r.map((v, i) => el("td", { class: numeric[i] ? "num" : "", text: formatCell(v) }))));
  const n = t.rows.length;
  return el("details", { class: "card data" },
    el("summary", { text: `${t.title || "Data"} · ${n} row${n === 1 ? "" : "s"}` }),
    el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, head), el("tbody", {}, ...body))));
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function chartCard(spec) {
  const canvas = el("canvas", { role: "img", "aria-label": spec.title });
  const card = el("figure", { class: "card chart-card" }, el("h4", { text: spec.title }), el("div", { class: "chart-box" }, canvas));
  card.style.margin = "0";

  const palette = [1, 2, 3, 4, 5, 6].map((i) => cssVar(`--chart-${i}`));
  const text = cssVar("--muted");
  const grid = cssVar("--grid");
  const pie = spec.type === "pie";

  const datasets = spec.datasets.map((d, i) => ({
    label: d.label,
    data: d.data,
    backgroundColor: pie ? spec.labels.map((_, j) => palette[j % palette.length]) : palette[i % palette.length] + (spec.type === "line" ? "" : "cc"),
    borderColor: pie ? cssVar("--surface") : palette[i % palette.length],
    borderWidth: pie ? 2 : spec.type === "line" ? 2.5 : 0,
    borderRadius: spec.type === "bar" ? 6 : 0,
    pointRadius: spec.type === "scatter" ? 3 : spec.type === "line" ? 2.5 : 0,
    tension: 0.3,
  }));

  const axis = (title) => ({
    grid: { color: grid },
    ticks: { color: text, maxRotation: 45, autoSkip: true },
    title: { display: !!title, text: title || "", color: text },
  });

  requestAnimationFrame(() => {
    new Chart(canvas, {
      type: spec.type,
      data: { labels: spec.labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: matchMedia("(prefers-reduced-motion: reduce)").matches ? false : { duration: 500 },
        plugins: {
          legend: { display: pie || datasets.length > 1, labels: { color: text, boxWidth: 12 } },
          tooltip: { callbacks: { label: (ctx) => {
            const v = spec.type === "scatter" ? `(${formatCell(ctx.raw.x)}, ${formatCell(ctx.raw.y)})` : formatCell(ctx.raw);
            return `${ctx.dataset.label}: ${v}`;
          } } },
        },
        scales: pie ? {} : { x: axis(spec.x_label), y: { ...axis(spec.y_label), beginAtZero: spec.type === "bar" } },
      },
    });
  });
  return card;
}

// ---------------------------------------------------------------- wiring

function autosize() {
  const q = $("question");
  q.style.height = "auto";
  q.style.height = Math.min(q.scrollHeight, 180) + "px";
}

function init() {
  const input = $("file-input");
  const zone = $("dropzone");
  input.addEventListener("change", () => uploadFile(input.files[0]));
  ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove("drag"); }));
  zone.addEventListener("drop", (e) => uploadFile(e.dataTransfer.files[0]));
  document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => loadSample(b.dataset.sample, b)));

  $("ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("question").value); });
  $("question").addEventListener("input", autosize);
  $("key-panel").addEventListener("submit", (e) => {
    e.preventDefault();
    const value = $("api-key").value.trim();
    if (!value) { setKey(""); refreshKeyPanel(); return; }
    if (!value.startsWith("sk-ant-")) { $("key-status").textContent = "Anthropic keys start with sk-ant-."; return; }
    setKey(value);
    refreshKeyPanel();
    $("question").focus();
  });
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask($("question").value); }
  });
  $("new-file").addEventListener("click", () => { if (!state.busy) showStart(); });
  $("new-chat").addEventListener("click", () => {
    if (!state.doc || state.busy) return;
    state.history = [];
    resetMessages();
  });

  api("/api/health").then((h) => {
    $("model-badge").textContent = h.model;
    $("model-badge").hidden = false;
    state.keyRequired = !!h.key_required;
    $("byok-note").hidden = !state.keyRequired;
    if (state.doc) refreshKeyPanel();
  }).catch(() => setUploadStatus("The API isn't reachable right now. If it's on a free host, it may be waking up; try again in a minute.", true));
  api("/api/limits").then((l) => {
    state.maxBytes = l.max_upload_mb * 1024 * 1024;
    $("limits-hint").textContent = `PDF up to ${l.max_pdf_pages} pages or CSV, up to ${l.max_upload_mb} MB`;
  }).catch(() => {});
}

init();
