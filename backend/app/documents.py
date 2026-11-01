"""Turn an uploaded file into a Document the assistant can work with."""

from __future__ import annotations

import io
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from . import data_tools


class UploadError(ValueError):
    """The upload can't be used; the message is safe to show the user."""


@dataclass
class Document:
    id: str
    kind: Literal["pdf", "csv"]
    filename: str
    size_bytes: int
    created_at: float
    pdf_bytes: bytes | None = None
    page_count: int | None = None
    preview: str = ""
    df: pd.DataFrame | None = None
    profile: dict[str, Any] | None = None
    # Full Claude conversation, appended to and never edited, so the cached
    # prefix (document + earlier turns) stays valid across questions.
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
        }
        if self.kind == "pdf":
            out.update(pages=self.page_count, preview=self.preview)
        else:
            assert self.profile is not None and self.df is not None
            out.update(
                rows=self.profile["rows"],
                columns=[{"name": c["name"], "dtype": c["dtype"]} for c in self.profile["columns"]],
                sample=data_tools.table(self.df.head(5)),
            )
        return out


def load(filename: str, data: bytes, *, max_bytes: int, max_pages: int, max_rows: int) -> Document:
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > max_bytes:
        raise UploadError(f"The file is larger than {max_bytes // (1024 * 1024)} MB.")

    name = filename.rsplit("/", 1)[-1][:200] or "upload"
    lowered = name.lower()
    doc_id = secrets.token_urlsafe(12)

    if lowered.endswith(".pdf") or data[:5] == b"%PDF-":
        return _load_pdf(doc_id, name, data, max_pages)
    if lowered.endswith((".csv", ".tsv", ".txt")):
        return _load_csv(doc_id, name, data, max_rows)
    raise UploadError("Upload a PDF or a CSV file.")


def _load_pdf(doc_id: str, name: str, data: bytes, max_pages: int) -> Document:
    if data[:5] != b"%PDF-":
        raise UploadError("That file doesn't look like a PDF.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise UploadError("Password-protected PDFs aren't supported.")
        pages = len(reader.pages)
        preview = (reader.pages[0].extract_text() or "").strip()[:600] if pages else ""
    except PdfReadError as exc:
        raise UploadError("The PDF couldn't be read.") from exc
    if pages == 0:
        raise UploadError("The PDF has no pages.")
    if pages > max_pages:
        raise UploadError(f"The PDF has {pages} pages; the limit is {max_pages}.")
    return Document(
        id=doc_id,
        kind="pdf",
        filename=name,
        size_bytes=len(data),
        created_at=time.time(),
        pdf_bytes=data,
        page_count=pages,
        preview=preview,
    )


def _load_csv(doc_id: str, name: str, data: bytes, max_rows: int) -> Document:
    text = _decode(data)
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", nrows=max_rows + 1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise UploadError("The CSV couldn't be parsed. Check that it has a header row.") from exc
    if df.empty or len(df.columns) == 0:
        raise UploadError("The CSV has no data rows.")
    if len(df) > max_rows:
        raise UploadError(f"The CSV has more than {max_rows:,} rows.")

    df.columns = _unique_names(df.columns)
    df = data_tools.coerce_dates(df)
    return Document(
        id=doc_id,
        kind="csv",
        filename=name,
        size_bytes=len(data),
        created_at=time.time(),
        df=df,
        profile=data_tools.profile(df),
    )


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UploadError("The CSV's text encoding isn't supported.")


def _unique_names(columns: pd.Index) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, raw in enumerate(columns):
        name = str(raw).strip() or f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out
