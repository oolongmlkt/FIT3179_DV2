
"""
FIT3179 DV2 — Chart 8 Director Collaboration Explorer

Uses the existing complete aggregated edge dataset.
Does not modify the original data or Chart 8 baseline files.

Run:
    python scripts/prepare_chart8_explorer.py
"""

from pathlib import Path
from collections import defaultdict
import csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "charts"

SOURCE = DATA / "director_actor_all_edges.csv"

OUT_EDGES = DATA / "director_actor_explorer_edges.csv"
OUT_NODES = DATA / "director_actor_explorer_nodes.csv"
OUT_DIRECTORS = DATA / "director_actor_explorer_directors.csv"

MIN_COLLABORATIONS = 5
TOP_DIRECTORS = 15
TOP_ACTORS = 10

PERIODS = [
    "1980s",
    "1990s",
    "2000s",
    "2010s",
    "2020s",
    "2000–2025",
    "All years"
]

# Store only qualifying partnerships, not millions
# of one-off relationships.
by_period = {
    period: defaultdict(list)
    for period in PERIODS
}

print("\n=== CHART 8 EXPLORER PREPROCESSING ===")

print("\nReading complete collaboration dataset...")

with SOURCE.open(
    "r",
    newline="",
    encoding="utf-8-sig"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        count = int(row["collaboration_count"])
        period = row["period"]

        if (
            period not in by_period
            or count < MIN_COLLABORATIONS
        ):
            continue

        row["collaboration_count"] = count
        row["first_year"] = int(row["first_year"])
        row["last_year"] = int(row["last_year"])

        by_period[period][row["director"]].append(row)


edge_fields = [
    "period",
    "source",
    "target",
    "director",
    "actor",
    "collaboration_count",
    "first_year",
    "last_year"
]

node_fields = [
    "period",
    "director",
    "id",
    "name",
    "type",
    "degree"
]

director_fields = [
    "period",
    "director",
    "rank",
    "qualifying_actors",
    "total_collaborations"
]

with (
    OUT_EDGES.open(
        "w", newline="", encoding="utf-8-sig"
    ) as edge_file,

    OUT_NODES.open(
        "w", newline="", encoding="utf-8-sig"
    ) as node_file,

    OUT_DIRECTORS.open(
        "w", newline="", encoding="utf-8-sig"
    ) as director_file
):

    edge_writer = csv.DictWriter(
        edge_file,
        fieldnames=edge_fields
    )

    node_writer = csv.DictWriter(
        node_file,
        fieldnames=node_fields
    )

    director_writer = csv.DictWriter(
        director_file,
        fieldnames=director_fields
    )

    edge_writer.writeheader()
    node_writer.writeheader()
    director_writer.writeheader()

    for period in PERIODS:

        directors = by_period[period]

        # Rank directors by the number of actors
        # with at least five shared productions.
        # Break ties using total collaboration weight.
        ranked = sorted(
            directors,
            key=lambda name: (
                -len(directors[name]),
                -sum(
                    e["collaboration_count"]
                    for e in directors[name]
                ),
                name
            )
        )

        selected = ranked[:TOP_DIRECTORS]

        print(f"\n{period}")
        print(f"Selected directors: {len(selected)}")

        for rank, director in enumerate(
            selected,
            start=1
        ):

            candidates = sorted(
                directors[director],
                key=lambda e: (
                    -e["collaboration_count"],
                    e["actor"]
                )
            )

            strongest = candidates[:TOP_ACTORS]

            director_writer.writerow({
                "period": period,
                "director": director,
                "rank": rank,
                "qualifying_actors": len(candidates),
                "total_collaborations": sum(
                    e["collaboration_count"]
                    for e in candidates
                )
            })

            for edge in strongest:
                edge_writer.writerow(edge)

            # Nodes are scoped by BOTH period and director.
            # An actor shared by two directors therefore
            # appears in each director's selectable view.

            director_id = f"director:{director}"

            node_writer.writerow({
                "period": period,
                "director": director,
                "id": director_id,
                "name": director,
                "type": "Director",
                "degree": len(strongest)
            })

            for edge in strongest:

                node_writer.writerow({
                    "period": period,
                    "director": director,
                    "id": edge["target"],
                    "name": edge["actor"],
                    "type": "Actor",
                    "degree": 1
                })

            if period == "2000–2025" and rank <= 5:
                print(
                    f"  {rank}. {director}: "
                    f"{len(candidates)} qualifying actors"
                )

print("\n=== COMPLETE ===")
print(OUT_EDGES)
print(OUT_NODES)
print(OUT_DIRECTORS)
