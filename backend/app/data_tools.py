"""Safe, structured analysis over a pandas DataFrame.

Claude never sends code. It sends a small JSON query spec (filters, grouping,
aggregations, sorting) that is validated with pydantic and executed here with
pandas and NumPy. Column names are checked against the DataFrame, so a bad
spec produces a clear error message that goes back to Claude to correct.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

MAX_ROWS = 1000  # cap on query results and scatter points

FilterOp = Literal["==", "!=", ">", ">=", "<", "<=", "contains", "in", "not_in", "is_null", "not_null"]
AggName = Literal["count", "sum", "mean", "median", "min", "max", "std", "nunique", "p25", "p75", "p90"]
TimeUnit = Literal["day", "week", "month", "quarter", "year"]


class QueryError(ValueError):
    """A query spec that is valid JSON but can't run against this dataset."""


class Filter(BaseModel):
    column: str
    op: FilterOp
    value: Any = None


class GroupKey(BaseModel):
    column: str
    time_unit: TimeUnit | None = Field(default=None, description="Bucket a date column by this unit before grouping.")


class Metric(BaseModel):
    agg: AggName
    column: str | None = Field(default=None, description="Not needed for count.")
    alias: str | None = None

    def name(self) -> str:
        if self.alias:
            return self.alias
        return "count" if self.column is None else f"{self.agg}_{self.column}"


class Sort(BaseModel):
    column: str
    descending: bool = True


class Query(BaseModel):
    filters: list[Filter] = Field(default_factory=list)
    group_by: list[GroupKey] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list, description="Columns to return when not aggregating.")
    sort: list[Sort] = Field(default_factory=list)
    limit: int = Field(default=50, ge=1, le=MAX_ROWS)

    @field_validator("group_by", mode="before")
    @classmethod
    def _plain_strings(cls, value: Any) -> Any:
        # Accept ["region"] as shorthand for [{"column": "region"}].
        if isinstance(value, list):
            return [{"column": v} if isinstance(v, str) else v for v in value]
        return value


class ChartRequest(BaseModel):
    chart_type: Literal["bar", "line", "pie", "scatter"]
    title: str
    query: Query = Field(default_factory=Query)
    x: str | None = Field(default=None, description="Column for the x axis (scatter, or override).")
    y: list[str] = Field(default_factory=list, description="Result columns to plot as series.")


class ValuesChart(BaseModel):
    """A chart whose numbers Claude read from a document (PDF), not computed here."""

    chart_type: Literal["bar", "line", "pie"]
    title: str
    labels: list[str] = Field(min_length=1, max_length=100)
    series: list[dict[str, Any]] = Field(min_length=1, max_length=6)
    x_label: str | None = None
    y_label: str | None = None

    @field_validator("series")
    @classmethod
    def _series_shape(cls, series: list[dict[str, Any]], info: Any) -> list[dict[str, Any]]:
        labels = info.data.get("labels") or []
        for s in series:
            if not isinstance(s.get("name"), str) or not isinstance(s.get("values"), list):
                raise ValueError("each series needs a 'name' string and a 'values' list")
            if len(s["values"]) != len(labels):
                raise ValueError(f"series '{s['name']}' has {len(s['values'])} values for {len(labels)} labels")
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in s["values"]):
                raise ValueError(f"series '{s['name']}' must contain only numbers")
        return series


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #


def profile(df: pd.DataFrame, sample_rows: int = 5) -> dict[str, Any]:
    """Describe a DataFrame compactly enough to put in a prompt."""
    columns = []
    for name in df.columns:
        s = df[name]
        info: dict[str, Any] = {
            "name": str(name),
            "dtype": _kind(s),
            "nulls": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
        }
        if info["dtype"] == "number":
            values = s.dropna().to_numpy(dtype=float)
            if values.size:
                p25, p50, p75 = np.percentile(values, [25, 50, 75])
                info.update(
                    min=_num(values.min()),
                    p25=_num(p25),
                    median=_num(p50),
                    p75=_num(p75),
                    max=_num(values.max()),
                    mean=_num(values.mean()),
                )
        elif info["dtype"] == "date":
            info.update(min=str(s.min()), max=str(s.max()))
        else:
            top = s.astype("string").value_counts(dropna=True).head(8)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
        columns.append(info)

    return {
        "rows": int(len(df)),
        "columns": columns,
        "sample": _records(df.head(sample_rows)),
    }


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    return "text"


def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse text columns that look like dates, so time grouping works."""
    for name in df.columns:
        s = df[name]
        if s.dtype != object and not pd.api.types.is_string_dtype(s):
            continue
        lowered = str(name).lower()
        if not any(word in lowered for word in ("date", "time", "day", "month", "created", "updated", "at")):
            continue
        parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.9:
            df[name] = parsed
    return df


# --------------------------------------------------------------------------- #
# Query execution
# --------------------------------------------------------------------------- #


def run_query(df: pd.DataFrame, q: Query) -> pd.DataFrame:
    _check_columns(df, q)
    out = df
    for f in q.filters:
        out = out[_mask(out, f)]

    if q.group_by or q.metrics:
        out = _aggregate(out, q)
    elif q.columns:
        out = out[q.columns]

    for s in q.sort:
        if s.column not in out.columns:
            raise QueryError(f"cannot sort by '{s.column}'; result columns are {list(map(str, out.columns))}")
    if q.sort:
        out = out.sort_values(
            by=[s.column for s in q.sort],
            ascending=[not s.descending for s in q.sort],
            kind="stable",
        )
    return out.head(q.limit).reset_index(drop=True)


def _check_columns(df: pd.DataFrame, q: Query) -> None:
    known = set(map(str, df.columns))
    wanted = [f.column for f in q.filters] + [g.column for g in q.group_by] + list(q.columns)
    wanted += [m.column for m in q.metrics if m.column]
    missing = [c for c in wanted if c not in known]
    if missing:
        raise QueryError(f"unknown column(s) {missing}; available columns are {sorted(known)}")
    for g in q.group_by:
        if g.time_unit and not pd.api.types.is_datetime64_any_dtype(df[g.column]):
            raise QueryError(f"column '{g.column}' is not a date, so it can't be grouped by {g.time_unit}")
    for m in q.metrics:
        if m.column is None and m.agg != "count":
            raise QueryError(f"aggregation '{m.agg}' needs a column")
        if (
            m.column
            and m.agg not in ("count", "nunique", "min", "max")
            and not pd.api.types.is_numeric_dtype(df[m.column])
        ):
            raise QueryError(f"aggregation '{m.agg}' needs a numeric column; '{m.column}' is {_kind(df[m.column])}")


def _mask(df: pd.DataFrame, f: Filter) -> pd.Series:
    s = df[f.column]
    v = f.value
    if f.op == "is_null":
        return s.isna()
    if f.op == "not_null":
        return s.notna()
    if f.op in ("in", "not_in"):
        if not isinstance(v, list):
            raise QueryError(f"'{f.op}' needs a list value")
        m = s.isin([_like(s, x) for x in v])
        return ~m if f.op == "not_in" else m
    if f.op == "contains":
        return s.astype("string").str.contains(str(v), case=False, regex=False, na=False)
    v = _like(s, v)
    try:
        return {
            "==": s == v,
            "!=": s != v,
            ">": s > v,
            ">=": s >= v,
            "<": s < v,
            "<=": s <= v,
        }[f.op]
    except TypeError as exc:
        raise QueryError(f"can't compare column '{f.column}' with {f.value!r}") from exc


def _like(s: pd.Series, v: Any) -> Any:
    """Convert a JSON value to the column's type (numbers, dates)."""
    if v is None:
        return v
    if pd.api.types.is_datetime64_any_dtype(s):
        try:
            return pd.Timestamp(v)
        except (ValueError, TypeError) as exc:
            raise QueryError(f"{v!r} is not a date") from exc
    if pd.api.types.is_numeric_dtype(s) and isinstance(v, str):
        try:
            return float(v)
        except ValueError as exc:
            raise QueryError(f"{v!r} is not a number") from exc
    return v


_PANDAS_AGG = {
    "count": "count",
    "sum": "sum",
    "mean": "mean",
    "median": "median",
    "min": "min",
    "max": "max",
    "std": "std",
    "nunique": "nunique",
}
_PERCENTILE = {"p25": 25, "p75": 75, "p90": 90}


def _aggregate(df: pd.DataFrame, q: Query) -> pd.DataFrame:
    metrics = q.metrics or [Metric(agg="count")]
    work = df.copy()
    keys: list[str] = []
    for g in q.group_by:
        if g.time_unit:
            key = f"{g.column}_{g.time_unit}"
            work[key] = _bucket(work[g.column], g.time_unit)
        else:
            key = g.column
        keys.append(key)

    def compute(frame: pd.DataFrame) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for m in metrics:
            if m.column is None:
                row[m.name()] = int(len(frame))
                continue
            col = frame[m.column]
            if m.agg in _PERCENTILE:
                values = col.dropna().to_numpy(dtype=float)
                row[m.name()] = float(np.percentile(values, _PERCENTILE[m.agg])) if values.size else None
            else:
                row[m.name()] = getattr(col, _PANDAS_AGG[m.agg])()
        return row

    if not keys:
        return pd.DataFrame([compute(work)])

    rows = []
    for key_values, frame in work.groupby(keys, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        rows.append({**dict(zip(keys, key_values, strict=True)), **compute(frame)})
    return pd.DataFrame(rows)


def _bucket(s: pd.Series, unit: str) -> pd.Series:
    if unit == "day":
        return s.dt.strftime("%Y-%m-%d")
    if unit == "week":
        return s.dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    if unit == "month":
        return s.dt.strftime("%Y-%m")
    if unit == "quarter":
        return s.dt.to_period("Q").astype("string")
    return s.dt.strftime("%Y")


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


def build_chart(df: pd.DataFrame, req: ChartRequest) -> dict[str, Any]:
    """Run the chart's query and shape the result for Chart.js."""
    if req.chart_type == "scatter":
        if not req.x or len(req.y) != 1:
            raise QueryError("a scatter chart needs 'x' and exactly one 'y' column")
        q = req.query.model_copy(
            update={"columns": [req.x, req.y[0]], "group_by": [], "metrics": [], "limit": MAX_ROWS}
        )
        result = run_query(df, q).dropna()
        points = [{"x": _num(a), "y": _num(b)} for a, b in zip(result[req.x], result[req.y[0]], strict=True)]
        return {
            "type": "scatter",
            "title": req.title,
            "datasets": [{"label": req.y[0], "data": points}],
            "x_label": req.x,
            "y_label": req.y[0],
        }

    result = run_query(df, req.query)
    if result.empty:
        raise QueryError("the chart query returned no rows")
    x = req.x or (_group_key_names(req.query)[0] if req.query.group_by else str(result.columns[0]))
    if x not in result.columns:
        raise QueryError(f"x column '{x}' is not in the result; result columns are {list(map(str, result.columns))}")
    y = req.y or [c for c in map(str, result.columns) if c != x and pd.api.types.is_numeric_dtype(result[c])]
    missing = [c for c in y if c not in result.columns]
    if missing or not y:
        raise QueryError(f"y column(s) {missing or y} not found; result columns are {list(map(str, result.columns))}")
    if req.chart_type == "pie" and len(y) != 1:
        raise QueryError("a pie chart takes exactly one y column")

    return {
        "type": req.chart_type,
        "title": req.title,
        "labels": [str(v) for v in result[x]],
        "datasets": [{"label": c, "data": [_num(v) for v in result[c]]} for c in y],
        "x_label": x,
        "y_label": y[0] if len(y) == 1 else None,
    }


def values_chart(req: ValuesChart) -> dict[str, Any]:
    return {
        "type": req.chart_type,
        "title": req.title,
        "labels": req.labels,
        "datasets": [{"label": s["name"], "data": [_num(v) for v in s["values"]]} for s in req.series],
        "x_label": req.x_label,
        "y_label": req.y_label,
    }


def _group_key_names(q: Query) -> list[str]:
    return [f"{g.column}_{g.time_unit}" if g.time_unit else g.column for g in q.group_by]


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #


def table(df: pd.DataFrame) -> dict[str, Any]:
    return {"columns": [str(c) for c in df.columns], "rows": [list(r.values()) for r in _records(df)]}


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): _json(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def _json(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.isoformat()
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return _num(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v if isinstance(v, (int, str)) else str(v)


def _num(v: Any) -> float | int | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return int(f) if f.is_integer() and abs(f) < 2**53 else round(f, 6)
