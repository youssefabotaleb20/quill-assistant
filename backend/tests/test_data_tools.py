from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from app import data_tools as dt


@pytest.fixture
def df(sales_csv: bytes) -> pd.DataFrame:
    return dt.coerce_dates(pd.read_csv(io.BytesIO(sales_csv)))


def q(**kwargs) -> dt.Query:
    return dt.Query.model_validate(kwargs)


def test_dates_are_parsed(df):
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
    assert df["region"].dtype != "datetime64[ns]"


def test_profile_describes_columns(df):
    p = dt.profile(df)
    assert p["rows"] == len(df)
    cols = {c["name"]: c for c in p["columns"]}
    assert cols["revenue"]["dtype"] == "number"
    assert cols["revenue"]["max"] == pytest.approx(df["revenue"].max())
    assert cols["order_date"]["dtype"] == "date"
    assert "Cairo" in cols["region"]["top_values"]
    assert cols["rating"]["nulls"] == int(df["rating"].isna().sum())
    assert len(p["sample"]) == 5


def test_group_sum_matches_pandas(df):
    result = dt.run_query(
        df, q(group_by=["region"], metrics=[{"agg": "sum", "column": "revenue"}], sort=[{"column": "sum_revenue"}])
    )
    expected = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    assert list(result["region"]) == list(expected.index)
    assert np.allclose(result["sum_revenue"], expected.values)


def test_filters_and_count(df):
    result = dt.run_query(
        df,
        q(
            filters=[{"column": "channel", "op": "==", "value": "Online"}, {"column": "units", "op": ">=", "value": 5}],
            metrics=[{"agg": "count"}],
        ),
    )
    assert result.loc[0, "count"] == len(df[(df.channel == "Online") & (df.units >= 5)])


def test_in_contains_and_null_filters(df):
    r = dt.run_query(
        df, q(filters=[{"column": "region", "op": "in", "value": ["Cairo", "Giza"]}], metrics=[{"agg": "count"}])
    )
    assert r.loc[0, "count"] == df.region.isin(["Cairo", "Giza"]).sum()
    r = dt.run_query(
        df, q(filters=[{"column": "product", "op": "contains", "value": "latte"}], metrics=[{"agg": "count"}])
    )
    assert r.loc[0, "count"] == df["product"].str.contains("Latte").sum()
    r = dt.run_query(df, q(filters=[{"column": "rating", "op": "is_null"}], metrics=[{"agg": "count"}]))
    assert r.loc[0, "count"] == df.rating.isna().sum()


def test_date_filter_and_monthly_buckets(df):
    result = dt.run_query(
        df,
        q(
            filters=[{"column": "order_date", "op": ">=", "value": "2025-07-01"}],
            group_by=[{"column": "order_date", "time_unit": "month"}],
            metrics=[{"agg": "sum", "column": "revenue", "alias": "revenue"}],
            sort=[{"column": "order_date_month", "descending": False}],
        ),
    )
    assert list(result["order_date_month"]) == [f"2025-{m:02d}" for m in range(7, 13)]
    july = df[df.order_date.dt.month == 7]["revenue"].sum()
    assert result.loc[0, "revenue"] == pytest.approx(july)


def test_percentiles_use_numpy(df):
    result = dt.run_query(df, q(metrics=[{"agg": "p90", "column": "revenue"}, {"agg": "median", "column": "revenue"}]))
    assert result.loc[0, "p90_revenue"] == pytest.approx(np.percentile(df.revenue, 90))
    assert result.loc[0, "median_revenue"] == pytest.approx(df.revenue.median())


def test_raw_columns_with_limit(df):
    result = dt.run_query(df, q(columns=["order_id", "revenue"], sort=[{"column": "revenue"}], limit=3))
    assert list(result.columns) == ["order_id", "revenue"]
    assert len(result) == 3
    assert result.loc[0, "revenue"] == df.revenue.max()


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"group_by": ["nope"]}, "unknown column"),
        ({"metrics": [{"agg": "sum", "column": "region"}]}, "needs a numeric column"),
        ({"metrics": [{"agg": "mean"}]}, "needs a column"),
        ({"group_by": [{"column": "region", "time_unit": "month"}]}, "is not a date"),
        ({"filters": [{"column": "region", "op": "in", "value": "Cairo"}]}, "needs a list"),
        ({"filters": [{"column": "units", "op": ">", "value": "lots"}]}, "is not a number"),
        ({"group_by": ["region"], "sort": [{"column": "revenue"}]}, "cannot sort"),
    ],
)
def test_bad_queries_explain_themselves(df, spec, message):
    with pytest.raises(dt.QueryError, match=message):
        dt.run_query(df, q(**spec))


def test_bar_chart_spec(df):
    req = dt.ChartRequest.model_validate(
        {
            "chart_type": "bar",
            "title": "Revenue by region",
            "query": {"group_by": ["region"], "metrics": [{"agg": "sum", "column": "revenue"}]},
        }
    )
    spec = dt.build_chart(df, req)
    assert spec["type"] == "bar"
    assert set(spec["labels"]) == set(df.region.unique())
    assert spec["datasets"][0]["label"] == "sum_revenue"
    assert sum(spec["datasets"][0]["data"]) == pytest.approx(df.revenue.sum(), rel=1e-6)


def test_scatter_chart_spec(df):
    req = dt.ChartRequest.model_validate(
        {"chart_type": "scatter", "title": "Units vs revenue", "x": "units", "y": ["revenue"]}
    )
    spec = dt.build_chart(df, req)
    assert spec["type"] == "scatter"
    assert 0 < len(spec["datasets"][0]["data"]) <= dt.MAX_ROWS
    assert set(spec["datasets"][0]["data"][0]) == {"x", "y"}


def test_pie_needs_one_series(df):
    req = dt.ChartRequest.model_validate(
        {
            "chart_type": "pie",
            "title": "x",
            "query": {"group_by": ["region"], "metrics": [{"agg": "sum", "column": "revenue"}, {"agg": "count"}]},
        }
    )
    with pytest.raises(dt.QueryError, match="exactly one"):
        dt.build_chart(df, req)


def test_values_chart_validates_lengths():
    ok = dt.ValuesChart.model_validate(
        {"chart_type": "bar", "title": "t", "labels": ["a", "b"], "series": [{"name": "s", "values": [1, 2.5]}]}
    )
    assert dt.values_chart(ok)["datasets"][0]["data"] == [1, 2.5]
    with pytest.raises(ValueError, match="2 labels"):
        dt.ValuesChart.model_validate(
            {"chart_type": "bar", "title": "t", "labels": ["a", "b"], "series": [{"name": "s", "values": [1]}]}
        )


def test_table_is_json_safe():
    frame = pd.DataFrame({"a": [1, None], "b": [pd.Timestamp("2025-01-01"), pd.NaT], "c": [np.int64(3), np.int64(4)]})
    t = dt.table(frame)
    assert t["rows"][0] == [1, "2025-01-01T00:00:00", 3]
    assert t["rows"][1] == [None, None, 4]
