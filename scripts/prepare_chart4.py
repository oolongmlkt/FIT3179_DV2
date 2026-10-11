"""
scripts/prepare_chart4.py

FIT3179 DV2 — Chart 4
Genre × Decade Heatmap

Input:
    data/charts/genre_by_year.csv

Output:
    data/charts/genre_by_decade.csv

Uses full genre counting, consistent with Chart 2.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "charts" / "genre_by_year.csv"
OUTPUT = ROOT / "data" / "charts" / "genre_by_decade.csv"

# Read existing Chart 2 dataset
df = pd.read_csv(INPUT)

required = {"year", "genre", "movie_count"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Validate input
df["year"] = pd.to_numeric(df["year"], errors="raise")
df["movie_count"] = pd.to_numeric(
    df["movie_count"], errors="raise"
)

assert df["year"].between(1980, 2025).all()
assert (df["movie_count"] >= 0).all()
assert df["movie_count"].notna().all()
assert df["genre"].notna().all()
assert (df["genre"].astype(str).str.strip() != "").all()

# Assign each year to its corresponding period
def assign_decade(year):
    if year >= 2020:
        return "2020–2025"
    return f"{(year // 10) * 10}s"

df["decade"] = df["year"].apply(assign_decade)

# Aggregate movie counts
result = (
    df.groupby(["decade", "genre"], as_index=False)
      ["movie_count"]
      .sum()
)

# Number of years represented by each period
period_years = {
    "1980s": 10,
    "1990s": 10,
    "2000s": 10,
    "2010s": 10,
    "2020–2025": 6
}

result["period_years"] = result["decade"].map(period_years)

# Average annual production volume
result["annual_average"] = (
    result["movie_count"] / result["period_years"]
)

# Share of genre assignments within each period.
# This is NOT the percentage of unique films.
period_totals = result.groupby("decade")["movie_count"].transform("sum")

result["genre_assignment_share"] = (
    result["movie_count"] / period_totals * 100
)

# Chronological ordering
decade_order = {
    "1980s": 0,
    "1990s": 1,
    "2000s": 2,
    "2010s": 3,
    "2020–2025": 4
}

result["decade_order"] = result["decade"].map(decade_order)

result = result.sort_values(
    ["decade_order", "genre"]
).reset_index(drop=True)

# Save output
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print(f"Chart 4 dataset saved: {OUTPUT}")
print(f"Rows generated: {len(result)}")
print(result.head(10))
