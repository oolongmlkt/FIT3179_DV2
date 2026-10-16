
"""
FIT3179 DV2 — Chart 8 Final Validation

Run:
    python scripts/validate_chart8.py

Checks:
- Required columns and missing values
- Duplicate nodes and edges
- Period-specific collaboration counts
- Node/edge integrity
- Director and actor display limits
- Agreement with the complete edge dataset
"""

from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "charts"

PERIODS = {
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2025),
    "2000–2025": (2000, 2025),
    "All years": (1980, 2025)
}

EDGE_COLUMNS = [
    "period", "source", "target",
    "director", "actor",
    "collaboration_count",
    "first_year", "last_year"
]

NODE_COLUMNS = [
    "period", "id", "name", "type", "degree"
]

errors = []

def check(condition, message):
    if condition:
        print(f"PASS: {message}")
    else:
        print(f"FAIL: {message}")
        errors.append(message)


# --------------------------------------------------
# 1. Load display datasets
# --------------------------------------------------

edges = pd.read_csv(
    DATA / "director_actor_edges.csv"
)

nodes = pd.read_csv(
    DATA / "director_actor_nodes.csv"
)

print("\n=== CHART 8 FINAL VALIDATION ===")

print(f"\nDisplay edges: {len(edges):,}")
print(f"Display nodes: {len(nodes):,}")

check(
    set(EDGE_COLUMNS).issubset(edges.columns),
    "Required edge columns exist"
)

check(
    set(NODE_COLUMNS).issubset(nodes.columns),
    "Required node columns exist"
)

if errors:
    raise SystemExit("Validation stopped: missing columns.")


# --------------------------------------------------
# 2. Missing and duplicate values
# --------------------------------------------------

check(
    not edges[EDGE_COLUMNS].isna().any().any(),
    "No missing edge values"
)

check(
    not nodes[NODE_COLUMNS].isna().any().any(),
    "No missing node values"
)

check(
    not edges.duplicated(
        ["period", "source", "target"]
    ).any(),
    "No duplicate period-edge combinations"
)

check(
    not nodes.duplicated(
        ["period", "id"]
    ).any(),
    "No duplicate period-node combinations"
)

check(
    set(edges["period"]) == set(PERIODS),
    "All seven edge periods exist"
)

check(
    set(nodes["period"]) == set(PERIODS),
    "All seven node periods exist"
)


# --------------------------------------------------
# 3. Validate each period
# --------------------------------------------------

print("\n=== PERIOD VALIDATION ===")

for period, (start, end) in PERIODS.items():

    e = edges[edges["period"] == period]
    n = nodes[nodes["period"] == period]

    director_nodes = n[n["type"] == "Director"]
    actor_nodes = n[n["type"] == "Actor"]

    print(f"\n{period}")
    print(
        f"Edges: {len(e)} | "
        f"Directors: {len(director_nodes)} | "
        f"Actors: {len(actor_nodes)}"
    )

    check(
        len(e) <= 75,
        f"{period}: Maximum 75 edges"
    )

    check(
        len(director_nodes) <= 15,
        f"{period}: Maximum 15 directors"
    )

    check(
        e.groupby("source").size().le(5).all(),
        f"{period}: Maximum five actors per director"
    )

    check(
        e["collaboration_count"].ge(5).all(),
        f"{period}: Minimum five collaborations"
    )

    check(
        (
            e["first_year"].between(start, end)
            & e["last_year"].between(start, end)
            & e["first_year"].le(e["last_year"])
        ).all(),
        f"{period}: Valid collaboration years"
    )

    node_ids = set(n["id"])

    check(
        set(e["source"]).issubset(node_ids)
        and set(e["target"]).issubset(node_ids),
        f"{period}: Every edge references existing nodes"
    )

    check(
        e["source"].str.startswith("director:").all()
        and e["target"].str.startswith("actor:").all(),
        f"{period}: Correct bipartite edge structure"
    )

    check(
        set(n["type"]).issubset({"Director", "Actor"}),
        f"{period}: Valid node types"
    )

    check(
        (
            director_nodes["id"].str.startswith("director:").all()
            and actor_nodes["id"].str.startswith("actor:").all()
        ),
        f"{period}: Node IDs match their types"
    )

    check(
        (
            (e["source"] == "director:" + e["director"])
            & (e["target"] == "actor:" + e["actor"])
        ).all(),
        f"{period}: Edge IDs match person names"
    )

    check(
        set(n["id"]) == set(e["source"]) | set(e["target"]),
        f"{period}: No disconnected or missing nodes"
    )

    degree = Counter()

    for row in e.itertuples(index=False):
        degree[row.source] += 1
        degree[row.target] += 1

    check(
        all(
            row.degree == degree[row.id]
            for row in n.itertuples(index=False)
        ),
        f"{period}: Node degrees are correct"
    )


# --------------------------------------------------
# 4. Verify display edges against complete dataset
# --------------------------------------------------

print("\n=== COMPLETE DATASET CROSS-CHECK ===")

# Only retain the small set of displayed keys in memory.
expected = {
    (r.period, r.source, r.target): (
        int(r.collaboration_count),
        int(r.first_year),
        int(r.last_year)
    )
    for r in edges.itertuples(index=False)
}

found = {}
duplicate_matches = set()
all_rows = 0
all_periods = set()

for chunk in pd.read_csv(
    DATA / "director_actor_all_edges.csv",
    chunksize=100_000
):
    all_rows += len(chunk)
    all_periods.update(chunk["period"].unique())

    check_columns = chunk[
        chunk["period"].isin(PERIODS)
    ]

    for r in check_columns.itertuples(index=False):

        key = (r.period, r.source, r.target)

        if key in expected:

            if key in found:
                duplicate_matches.add(key)

            found[key] = (
                int(r.collaboration_count),
                int(r.first_year),
                int(r.last_year)
            )

print(f"Complete dataset rows: {all_rows:,}")

check(
    all_periods == set(PERIODS),
    "Complete dataset contains all seven periods"
)

check(
    len(duplicate_matches) == 0,
    "No duplicate matches for displayed edges"
)

check(
    set(found) == set(expected),
    "Every displayed edge exists in complete dataset"
)

check(
    found == expected,
    "Displayed collaboration counts and years match source"
)


# --------------------------------------------------
# 5. Final result
# --------------------------------------------------

print("\n=== FINAL RESULT ===")

if errors:
    print(f"FAILED: {len(errors)} validation checks")
    for error in errors:
        print("-", error)
    raise SystemExit(1)

print("ALL CHART 8 VALIDATION CHECKS PASSED")
