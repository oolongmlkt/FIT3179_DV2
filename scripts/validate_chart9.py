
"""
FIT3179 DV2 — Chart 9 Validation

Validates the 2000–2025 production-company
ranking dataset against the processed source.
"""

from pathlib import Path
import pandas as pd


# -----------------------------------------
# Configuration
# -----------------------------------------

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "data/processed/movie_studios.csv"
OUTPUT = ROOT / "data/charts/studio_rankings.csv"

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


def check(condition, description):
    if not condition:
        raise AssertionError(f"FAIL: {description}")

    print(f"PASS: {description}")


# -----------------------------------------
# 1. Load datasets
# -----------------------------------------

df = pd.read_csv(OUTPUT)

source = pd.read_csv(
    SOURCE,
    usecols=["id", "year", "studio"]
)

source = source.dropna(
    subset=["id", "year", "studio"]
)

source = source[
    source["year"].between(2000, 2025)
].drop_duplicates(
    subset=["id", "studio"]
).copy()

source["period"] = source["year"].apply(get_period)

print("\nCHART 9 VALIDATION")
print("------------------------------")


# -----------------------------------------
# 2. Structural validation
# -----------------------------------------

required = [
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

check(
    all(c in df.columns for c in required),
    "All required columns exist"
)

check(
    len(df) == TOP_N * len(PERIODS),
    "Exactly 50 records"
)

check(
    df["company_name"].nunique() == TOP_N,
    "Exactly 10 companies"
)

check(
    set(df["period"]) == set(PERIODS),
    "All five periods are present"
)

check(
    not df.duplicated(
        ["company_name", "period"]
    ).any(),
    "No duplicate company-period records"
)

check(
    df.groupby("company_name")["period"]
    .nunique().eq(5).all(),
    "Every company has all five periods"
)

essential = [
    "company_name",
    "period",
    "period_order",
    "movie_count",
    "total_movies",
    "selection_rank",
    "active"
]

check(
    df[essential].notna().all().all(),
    "No missing essential values"
)

check(
    (df["movie_count"] >= 0).all(),
    "No negative production counts"
)

check(
    df["period_order"].eq(
        df["period"].map(PERIOD_ORDER)
    ).all(),
    "Period ordering is correct"
)


# -----------------------------------------
# 3. Validate selected companies
# -----------------------------------------

overall = (
    source.groupby("studio")["id"]
    .nunique()
    .reset_index(name="total_movies")
    .sort_values(
        ["total_movies", "studio"],
        ascending=[False, True]
    )
    .reset_index(drop=True)
)

expected = overall.head(TOP_N).copy()

expected["selection_rank"] = range(
    1, TOP_N + 1
)

actual = (
    df[
        ["company_name", "total_movies", "selection_rank"]
    ]
    .drop_duplicates()
    .rename(columns={"company_name": "studio"})
    .sort_values("selection_rank")
    .reset_index(drop=True)
)

check(
    actual.equals(
        expected[
            ["studio", "total_movies", "selection_rank"]
        ].reset_index(drop=True)
    ),
    "Top 10 selection and totals are correct"
)


# -----------------------------------------
# 4. Validate period production counts
# -----------------------------------------

counts = (
    source.groupby(["studio", "period"])["id"]
    .nunique()
    .to_dict()
)

expected_counts = df.apply(
    lambda row: counts.get(
        (row["company_name"], row["period"]),
        0
    ),
    axis=1
)

check(
    df["movie_count"].eq(expected_counts).all(),
    "All period counts match source data"
)

check(
    df.groupby("company_name")["movie_count"]
    .sum()
    .eq(
        df.groupby("company_name")["total_movies"]
        .first()
    ).all(),
    "Period counts sum to company totals"
)


# -----------------------------------------
# 5. Validate cohort rankings
# -----------------------------------------

active = df[df["movie_count"] > 0].copy()

expected_ranks = active.sort_values(
    ["period_order", "movie_count", "company_name"],
    ascending=[True, False, True]
).copy()

expected_ranks["expected_rank"] = (
    expected_ranks.groupby("period")
    .cumcount() + 1
)

rank_check = active.merge(
    expected_ranks[
        ["company_name", "period", "expected_rank"]
    ],
    on=["company_name", "period"],
    validate="one_to_one"
)

check(
    rank_check["rank"].eq(
        rank_check["expected_rank"]
    ).all(),
    "Active company rankings are correct"
)

check(
    all(
        sorted(group["rank"].astype(int).tolist())
        == list(range(1, len(group) + 1))
        for _, group in active.groupby("period")
    ),
    "Active ranks are consecutive within each period"
)

check(
    df.loc[df["movie_count"].eq(0), "rank"]
    .isna().all(),
    "Zero-production records have no cohort rank"
)


# -----------------------------------------
# 6. Validate global rankings
# -----------------------------------------

global_counts = (
    source.groupby(["period", "studio"])["id"]
    .nunique()
    .reset_index(name="count")
    .sort_values(
        ["period", "count", "studio"],
        ascending=[True, False, True]
    )
)

global_counts["expected_overall_rank"] = (
    global_counts.groupby("period")
    .cumcount() + 1
)

global_check = df.merge(
    global_counts[
        ["period", "studio", "expected_overall_rank"]
    ],
    left_on=["period", "company_name"],
    right_on=["period", "studio"],
    how="left",
    validate="one_to_one"
)

check(
    global_check["overall_rank"].fillna(-1).eq(
        global_check["expected_overall_rank"].fillna(-1)
    ).all(),
    "Global rankings match source data"
)

check(
    df.loc[df["movie_count"].eq(0), "overall_rank"]
    .isna().all(),
    "Zero-production records have no global rank"
)


# -----------------------------------------
# 7. Validate active flags
# -----------------------------------------

actual_active = (
    df["active"].astype(str).str.lower() == "true"
)

expected_active = df["movie_count"] > 0

check(
    actual_active.eq(expected_active).all(),
    "Active flags match production counts"
)


# -----------------------------------------
# 8. Summary
# -----------------------------------------

print("\nALL CHART 9 VALIDATION CHECKS PASSED")
print("------------------------------")

print(f"Records: {len(df)}")
print(f"Companies: {df['company_name'].nunique()}")
print(f"Periods: {df['period'].nunique()}")

print(
    "Zero-production records:",
    (df["movie_count"] == 0).sum()
)
