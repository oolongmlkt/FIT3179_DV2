
"""
FIT3179 DV2 — Chart 9
Production Company Ranking Bump Chart

Input:
    data/processed/movie_studios.csv

Output:
    data/charts/studio_rankings.csv

Method:
    Select the top 10 companies by distinct credited
    movies across 2000–2025.

    Track those companies across five reporting periods.

    Rank active companies within the selected cohort.
    Zero-production periods retain a row but no rank.
"""

from pathlib import Path
import pandas as pd


# -----------------------------------------
# Configuration
# -----------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data" / "processed" / "movie_studios.csv"
OUTPUT = ROOT / "data" / "charts" / "studio_rankings.csv"

TOP_N = 10

PERIODS = [
    "2000–2004",
    "2005–2009",
    "2010–2014",
    "2015–2019",
    "2020–2025"
]

PERIOD_ORDER = {
    period: i + 1
    for i, period in enumerate(PERIODS)
}


def get_period(year):
    if 2000 <= year <= 2004:
        return PERIODS[0]
    if 2005 <= year <= 2009:
        return PERIODS[1]
    if 2010 <= year <= 2014:
        return PERIODS[2]
    if 2015 <= year <= 2019:
        return PERIODS[3]
    if 2020 <= year <= 2025:
        return PERIODS[4]

    return None


# -----------------------------------------
# 1. Load and clean source data
# -----------------------------------------

df = pd.read_csv(
    INPUT,
    usecols=["id", "year", "studio"]
)

df = df.dropna(
    subset=["id", "year", "studio"]
)

df = df[
    df["year"].between(2000, 2025)
].copy()

# Prevent duplicate movie-company counting.
df = df.drop_duplicates(
    subset=["id", "studio"]
)

df["period"] = df["year"].apply(get_period)

print("\nCHART 9 PREPROCESSING")
print("------------------------------")
print(f"Eligible associations: {len(df):,}")
print(f"Unique movies: {df['id'].nunique():,}")
print(f"Unique companies: {df['studio'].nunique():,}")


# -----------------------------------------
# 2. Select top 10 companies
# -----------------------------------------

overall = (
    df.groupby("studio")["id"]
    .nunique()
    .reset_index(name="total_movies")
)

overall = overall.sort_values(
    ["total_movies", "studio"],
    ascending=[False, True]
).reset_index(drop=True)

selected = overall.head(TOP_N).copy()

selected["selection_rank"] = range(
    1, len(selected) + 1
)

selected_names = selected["studio"].tolist()

print("\nSelected production companies:")
print(selected.to_string(index=False))


# -----------------------------------------
# 3. Calculate period production counts
# -----------------------------------------

period_counts = (
    df.groupby(["studio", "period"])["id"]
    .nunique()
    .reset_index(name="movie_count")
)

# Global ranking among all companies active
# in each period.
period_counts = period_counts.sort_values(
    ["period", "movie_count", "studio"],
    ascending=[True, False, True]
)

period_counts["overall_rank"] = (
    period_counts.groupby("period")
    .cumcount() + 1
)


# -----------------------------------------
# 4. Create complete company-period grid
# -----------------------------------------

grid = pd.MultiIndex.from_product(
    [selected_names, PERIODS],
    names=["studio", "period"]
).to_frame(index=False)

result = grid.merge(
    period_counts,
    on=["studio", "period"],
    how="left",
    validate="one_to_one"
)

result["movie_count"] = (
    result["movie_count"]
    .fillna(0)
    .astype(int)
)

result = result.merge(
    selected[
        ["studio", "total_movies", "selection_rank"]
    ],
    on="studio",
    how="left",
    validate="many_to_one"
)

result["period_order"] = (
    result["period"].map(PERIOD_ORDER)
)

result["active"] = result["movie_count"] > 0


# -----------------------------------------
# 5. Calculate cohort rankings
# -----------------------------------------

result = result.sort_values(
    ["period_order", "movie_count", "studio"],
    ascending=[True, False, True]
)

# Only companies with credited productions
# receive a plotted rank.
result["rank"] = pd.Series(
    pd.NA,
    index=result.index,
    dtype="Int64"
)

active_mask = result["active"]

result.loc[active_mask, "rank"] = (
    result.loc[active_mask]
    .groupby("period")
    .cumcount()
    .add(1)
    .to_numpy()
)

result["overall_rank"] = (
    result["overall_rank"].astype("Int64")
)


# -----------------------------------------
# 6. Prepare final output
# -----------------------------------------

result = result.rename(
    columns={"studio": "company_name"}
)

result = result[
    [
        "company_name",
        "period",
        "period_order",
        "movie_count",
        "rank",
        "overall_rank",
        "total_movies",
        "selection_rank",
        "active"
    ]
]

result = result.sort_values(
    ["selection_rank", "period_order"]
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

result.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8"
)


# -----------------------------------------
# 7. Summary
# -----------------------------------------

print("\nCHART 9 PREPROCESSING COMPLETE")
print("------------------------------")

print(f"Output: {OUTPUT}")
print(f"Records: {len(result):,}")
print(f"Companies: {result['company_name'].nunique()}")
print(f"Periods: {result['period'].nunique()}")

print(
    "Zero-production records:",
    (~result["active"]).sum()
)

print("\nProduction counts by period:")

print(
    result.pivot(
        index="company_name",
        columns="period",
        values="movie_count"
    ).reindex(columns=PERIODS).to_string()
)
