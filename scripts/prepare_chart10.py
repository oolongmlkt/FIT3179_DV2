
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT = Path("data/charts/director_actor_all_edges.csv")
OUTPUT = Path("data/charts/director_actor_rankings.csv")

TOP_N = 30
CHUNK_SIZE = 200_000

PERIODS = [
    "2000–2025",
    "2000s",
    "2010s",
    "2020s",
]

COLUMNS = [
    "period",
    "source",
    "target",
    "director",
    "actor",
    "collaboration_count",
    "first_year",
    "last_year",
]

# --------------------------------------------------
# Read the existing dataset in chunks
# --------------------------------------------------

print("Reading Chart 8 collaboration dataset...")

top_chunks = []
total_rows = 0

for chunk in pd.read_csv(
    INPUT,
    usecols=COLUMNS,
    chunksize=CHUNK_SIZE,
):
    total_rows += len(chunk)

    # Keep only recognised periods.
    chunk = chunk[chunk["period"].isin(PERIODS)].copy()

    # Remove invalid records.
    chunk = chunk.dropna(subset=COLUMNS)

    chunk = chunk[
        (chunk["collaboration_count"] > 0)
        & (chunk["first_year"] <= chunk["last_year"])
    ]

    # Retain the strongest candidates from each period.
    chunk = chunk.sort_values(
        by=[
            "period",
            "collaboration_count",
            "director",
            "actor",
            "source",
            "target",
        ],
        ascending=[True, False, True, True, True, True],
        kind="mergesort",
    )

    top_chunks.append(
        chunk.groupby("period", sort=False).head(TOP_N)
    )

    if total_rows % 1_000_000 < CHUNK_SIZE:
        print(f"Processed {total_rows:,} rows...")

# --------------------------------------------------
# Combine candidates and calculate final rankings
# --------------------------------------------------

print("Calculating final rankings...")

rankings = pd.concat(top_chunks, ignore_index=True)

rankings = rankings.sort_values(
    by=[
        "period",
        "collaboration_count",
        "director",
        "actor",
        "source",
        "target",
    ],
    ascending=[True, False, True, True, True, True],
    kind="mergesort",
)

# Retain the top 30 for each period.
rankings = (
    rankings.groupby("period", sort=False)
    .head(TOP_N)
    .copy()
)

# Add a deterministic ranking.
rankings["rank"] = (
    rankings.groupby("period", sort=False)
    .cumcount() + 1
)

# Create readable partnership labels.
rankings["pair_label"] = (
    rankings["director"]
    + " / "
    + rankings["actor"]
)

# Sort periods into the intended display order.
rankings["period"] = pd.Categorical(
    rankings["period"],
    categories=PERIODS,
    ordered=True,
)

rankings = rankings.sort_values(
    ["period", "rank"]
)

# --------------------------------------------------
# Export
# --------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

rankings = rankings[
    [
        "period",
        "rank",
        "source",
        "target",
        "director",
        "actor",
        "pair_label",
        "collaboration_count",
        "first_year",
        "last_year",
    ]
]

rankings.to_csv(OUTPUT, index=False)

print("\nChart 10 preprocessing complete.")
print(f"Input rows processed: {total_rows:,}")
print(f"Output rows: {len(rankings):,}")
print(f"Saved to: {OUTPUT}")

print("\nRows per period:")
print(rankings.groupby("period", observed=True).size())

print("\nTop 15 partnerships (2000–2025):")
print(
    rankings[rankings["period"] == "2000–2025"]
    .head(15)[
        [
            "rank",
            "director",
            "actor",
            "collaboration_count",
        ]
    ]
    .to_string(index=False)
)