
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

SOURCE = Path("data/charts/director_actor_all_edges.csv")
OUTPUT = Path("data/charts/director_actor_rankings.csv")

PERIODS = {
    "2000–2025": (2000, 2025),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2025),
}

TOP_N = 30
CHUNK_SIZE = 200_000

SOURCE_COLUMNS = [
    "period", "source", "target", "director", "actor",
    "collaboration_count", "first_year", "last_year"
]

OUTPUT_COLUMNS = [
    "period", "rank", "source", "target", "director",
    "actor", "pair_label", "collaboration_count",
    "first_year", "last_year"
]

SORT_COLUMNS = [
    "period", "collaboration_count", "director",
    "actor", "source", "target"
]

SORT_ASCENDING = [True, False, True, True, True, True]

checks = 0


def check(condition, message):
    global checks

    if not condition:
        raise AssertionError(f"FAILED: {message}")

    checks += 1
    print(f"[PASS {checks:02d}] {message}")


def sorted_rankings(data):
    return data.sort_values(
        SORT_COLUMNS,
        ascending=SORT_ASCENDING,
        kind="mergesort"
    )


# --------------------------------------------------
# Part 1: Validate output structure
# --------------------------------------------------

print("\nPART 1: OUTPUT STRUCTURE\n")

check(SOURCE.is_file(), "Original source file exists")
check(OUTPUT.is_file(), "Chart 10 output file exists")

df = pd.read_csv(OUTPUT)

check(
    set(OUTPUT_COLUMNS).issubset(df.columns),
    "All required output columns exist"
)

check(len(df) == 120, "Output contains exactly 120 rows")

check(
    set(df["period"]) == set(PERIODS),
    "All four expected periods are present"
)

check(
    all(
        (df["period"] == period).sum() == TOP_N
        for period in PERIODS
    ),
    "Each period contains exactly 30 partnerships"
)

check(
    not df[OUTPUT_COLUMNS].isna().any().any(),
    "No missing values in required columns"
)

check(
    not df.duplicated(
        ["period", "source", "target"]
    ).any(),
    "No duplicate partnership identifiers within periods"
)

check(
    not df.duplicated(
        ["period", "rank"]
    ).any(),
    "No duplicate rankings within periods"
)

# --------------------------------------------------
# Part 2: Validate counts and rankings
# --------------------------------------------------

print("\nPART 2: COUNTS AND RANKINGS\n")

numeric_columns = [
    "rank", "collaboration_count", "first_year", "last_year"
]

check(
    all(
        pd.api.types.is_numeric_dtype(df[col])
        for col in numeric_columns
    ),
    "Numeric columns have numeric data types"
)

check(
    all(
        (df[col] % 1 == 0).all()
        for col in numeric_columns
    ),
    "All numeric values are whole numbers"
)

check(
    (df["collaboration_count"] > 0).all(),
    "All collaboration counts are positive"
)

check(
    all(
        sorted(group["rank"].tolist())
        == list(range(1, TOP_N + 1))
        for _, group in df.groupby("period")
    ),
    "Every period has consecutive rankings from 1 to 30"
)

expected_order = (
    sorted_rankings(df)
    .groupby("period", sort=False)
    .cumcount() + 1
)

check(
    (expected_order.to_numpy()
     == sorted_rankings(df)["rank"].to_numpy()).all(),
    "Ranking order and alphabetical tie-breaking are correct"
)

# --------------------------------------------------
# Part 3: Validate years and labels
# --------------------------------------------------

print("\nPART 3: YEARS AND LABELS\n")

check(
    (df["first_year"] <= df["last_year"]).all(),
    "First collaboration year never exceeds last year"
)

years_valid = True

for period, (start, end) in PERIODS.items():
    subset = df[df["period"] == period]

    if not (
        subset["first_year"].between(start, end).all()
        and subset["last_year"].between(start, end).all()
    ):
        years_valid = False

check(
    years_valid,
    "Collaboration years fall within their selected periods"
)

check(
    (
        df["pair_label"]
        == df["director"] + " / " + df["actor"]
    ).all(),
    "Partnership labels are constructed correctly"
)

check(
    (df["source"] == "director:" + df["director"]).all()
    and (df["target"] == "actor:" + df["actor"]).all(),
    "Display names are consistent with source identifiers"
)

# --------------------------------------------------
# Part 4: Independently verify source rankings
# --------------------------------------------------

print("\nPART 4: ORIGINAL SOURCE VERIFICATION\n")

candidates = []
source_rows = 0

for chunk in pd.read_csv(
    SOURCE,
    usecols=SOURCE_COLUMNS,
    chunksize=CHUNK_SIZE
):
    source_rows += len(chunk)

    chunk = chunk[
        chunk["period"].isin(PERIODS)
    ].copy()

    # Apply the same eligibility conditions as Phase A.
    chunk = chunk.dropna(subset=SOURCE_COLUMNS)

    chunk = chunk[
        (chunk["collaboration_count"] > 0)
        & (chunk["first_year"] <= chunk["last_year"])
    ]

    # Independently find the strongest 30 records
    # from each source chunk.
    chunk = sorted_rankings(chunk)

    candidates.append(
        chunk.groupby("period", sort=False).head(TOP_N)
    )

print(f"Scanned {source_rows:,} original records.")

source_top = pd.concat(candidates, ignore_index=True)

source_top = (
    sorted_rankings(source_top)
    .groupby("period", sort=False)
    .head(TOP_N)
    .copy()
)

source_top["rank"] = (
    source_top.groupby("period", sort=False)
    .cumcount() + 1
)

source_top["pair_label"] = (
    source_top["director"] + " / " + source_top["actor"]
)

check(
    source_rows == 14_628_284,
    "Original source row count matches Phase A"
)

check(
    len(source_top) == 120,
    "Independent source ranking contains 120 records"
)

# Compare the complete output with independently
# calculated top-30 source records.
comparison_columns = [
    "period", "rank", "source", "target",
    "director", "actor", "pair_label",
    "collaboration_count", "first_year", "last_year"
]

actual = df[comparison_columns].sort_values(
    ["period", "rank"]
).reset_index(drop=True)

expected = source_top[comparison_columns].sort_values(
    ["period", "rank"]
).reset_index(drop=True)

check(
    actual[["period", "source", "target"]].equals(
        expected[["period", "source", "target"]]
    ),
    "All output partnerships match the source top 30"
)

check(
    actual["collaboration_count"].equals(
        expected["collaboration_count"]
    ),
    "All collaboration counts match the original source"
)

check(
    actual[["first_year", "last_year"]].equals(
        expected[["first_year", "last_year"]]
    ),
    "All collaboration years match the original source"
)

check(
    actual[["rank", "pair_label"]].equals(
        expected[["rank", "pair_label"]]
    ),
    "All ranks and partnership labels match the source"
)

# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 50)
print("CHART 10 VALIDATION COMPLETE")
print("=" * 50)

print(f"Checks passed: {checks}")
print(f"Source records scanned: {source_rows:,}")
print(f"Validated output records: {len(df):,}")

print("\nTop partnership in each period:")

for period in PERIODS:
    top = df[
        (df["period"] == period)
        & (df["rank"] == 1)
    ].iloc[0]

    print(
        f"{period}: {top['director']} / "
        f"{top['actor']} "
        f"({top['collaboration_count']} movies)"
    )

print("\nAll Chart 10 validation checks passed.")
