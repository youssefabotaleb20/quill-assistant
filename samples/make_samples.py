"""Generate the demo files in this folder.

    python samples/make_samples.py

Both files describe a fictional company, Brightline Coffee Co. The CSV is
synthetic sales data from a fixed random seed; the PDF is a short annual
report whose figures match the CSV's yearly totals by region.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

REGIONS = {"Cairo": 1.25, "Alexandria": 0.9, "Giza": 1.0, "Mansoura": 0.6}
PRODUCTS = [
    ("Espresso Beans 1kg", "Beans", 21.0),
    ("House Blend 500g", "Beans", 12.5),
    ("Decaf 500g", "Beans", 13.5),
    ("Cold Brew Bottle", "Drinks", 3.8),
    ("Oat Latte Can", "Drinks", 2.9),
    ("Pour-over Kit", "Equipment", 34.0),
    ("Burr Grinder", "Equipment", 89.0),
    ("Travel Mug", "Merch", 16.0),
]
CHANNELS = ["Online", "Store", "Wholesale"]


def make_sales(rows: int = 2400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    # More orders later in the year and in December.
    weights = np.linspace(0.8, 1.3, len(days)) * np.where(days.month == 12, 1.5, 1.0)
    dates = rng.choice(days, size=rows, p=weights / weights.sum())

    region_names = list(REGIONS)
    region_w = np.array(list(REGIONS.values()))
    regions = rng.choice(region_names, size=rows, p=region_w / region_w.sum())
    product_idx = rng.choice(len(PRODUCTS), size=rows, p=[0.2, 0.2, 0.06, 0.16, 0.16, 0.08, 0.04, 0.1])
    channels = rng.choice(CHANNELS, size=rows, p=[0.45, 0.4, 0.15])

    units = rng.poisson(3, size=rows) + 1
    units = np.where(channels == "Wholesale", units * 6, units)
    base_price = np.array([PRODUCTS[i][2] for i in product_idx])
    discount = np.where(channels == "Wholesale", 0.18, rng.choice([0, 0, 0, 0.1], size=rows))
    unit_price = np.round(base_price * (1 - discount), 2)

    df = pd.DataFrame(
        {
            "order_id": [f"BL-{10000 + i}" for i in range(rows)],
            "order_date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "region": regions,
            "channel": channels,
            "product": [PRODUCTS[i][0] for i in product_idx],
            "category": [PRODUCTS[i][1] for i in product_idx],
            "units": units,
            "unit_price": unit_price,
            "revenue": np.round(units * unit_price, 2),
            "rating": np.where(rng.random(rows) < 0.3, np.nan, rng.choice([3, 4, 4, 5, 5, 5], size=rows)),
        }
    )
    return df.sort_values(["order_date", "order_id"]).reset_index(drop=True)


def make_report(df: pd.DataFrame, path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    h1, h2 = styles["Title"], styles["Heading2"]

    dates = pd.to_datetime(df["order_date"])
    quarterly = df.groupby(dates.dt.quarter)["revenue"].sum()
    by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    by_channel = df.groupby("channel")["revenue"].sum()
    total = df["revenue"].sum()

    def money(v: float) -> str:
        return f"${v:,.0f}"

    def grid(rows: list[list[str]]) -> Table:
        t = Table(rows, hAlign="LEFT", colWidths=[6 * cm] + [3.5 * cm] * (len(rows[0]) - 1))
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3b2a20")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8b8a8")),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ]
            )
        )
        return t

    story = [
        Paragraph("Brightline Coffee Co. Annual Report 2025", h1),
        Paragraph("A fictional company, made up for the AI Document Assistant demo.", body),
        Spacer(1, 12),
        Paragraph("Letter from the founder", h2),
        Paragraph(
            f"2025 was the year Brightline grew past its home city. Total revenue reached {money(total)}, "
            "with every quarter ahead of the one before it. We opened our Mansoura store in March, moved "
            "wholesale customers onto a monthly ordering portal in June, and redesigned the Oat Latte Can in "
            "September, after which it became our fastest-selling drink.",
            body,
        ),
        Paragraph(
            "Our priorities for 2026 are to open a second Cairo store, bring online orders to half of revenue, "
            "and cut delivery times outside Cairo to two days. We will also replace single-use cups in all "
            "stores by the end of the second quarter.",
            body,
        ),
        Spacer(1, 12),
        Paragraph("Revenue by quarter", h2),
        grid([["Quarter", "Revenue", "Share"]]
             + [[f"Q{q} 2025", money(v), f"{v / total:.1%}"] for q, v in quarterly.items()]
             + [["Total", money(total), "100%"]]),
        Spacer(1, 12),
        Paragraph("Revenue by region", h2),
        grid([["Region", "Revenue", "Share"]] + [[r, money(v), f"{v / total:.1%}"] for r, v in by_region.items()]),
        Spacer(1, 12),
        Paragraph("Revenue by channel", h2),
        grid([["Channel", "Revenue", "Share"]] + [[c, money(v), f"{v / total:.1%}"] for c, v in by_channel.items()]),
        PageBreak(),
        Paragraph("Operations", h2),
        Paragraph(
            "Average delivery time fell from 3.4 days in Q1 to 2.6 days in Q4 after we added a second courier "
            "partner in Alexandria. Customer ratings averaged 4.4 out of 5 across all channels. Returns stayed "
            "below 2% of orders, most of them damaged grinders in transit, which led us to change packaging "
            "suppliers in October.",
            body,
        ),
        Paragraph("Risks", h2),
        Paragraph(
            "Green coffee prices rose 18% during the year and remain our largest cost risk. We hedge about a "
            "third of next year's bean purchases with fixed-price contracts. A second risk is concentration: "
            "Cairo accounts for the largest share of revenue, so a slowdown there would affect results more "
            "than any other region.",
            body,
        ),
        Paragraph("Team", h2),
        Paragraph(
            "We ended the year with 46 employees, up from 31, including 9 baristas hired for the Mansoura store. "
            "Staff turnover was 12%, down from 19% in 2024.",
            body,
        ),
    ]
    SimpleDocTemplate(str(path), pagesize=A4, title="Brightline Coffee Co. Annual Report 2025",
                      author="Brightline Coffee Co. (fictional)").build(story)


if __name__ == "__main__":
    sales = make_sales()
    sales.to_csv(HERE / "brightline-sales-2025.csv", index=False)
    make_report(sales, HERE / "brightline-annual-report-2025.pdf")
    print(f"Wrote {len(sales)} rows and the annual report to {HERE}")
