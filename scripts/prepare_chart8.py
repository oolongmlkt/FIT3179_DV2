
"""
FIT3179 DV2 — Chart 8
Director–Actor Collaboration Network

Run from the project root:
    python scripts/prepare_chart8.py

Inputs:
    data/processed/movies.csv
    data/processed/movie_directors.csv
    data/processed/movie_cast.csv

Outputs:
    data/charts/director_actor_all_edges.csv
    data/charts/director_actor_edges.csv
    data/charts/director_actor_nodes.csv

The original processed datasets are not modified.
"""

from pathlib import Path
from collections import defaultdict
from tempfile import TemporaryDirectory
import sqlite3
import csv

import pandas as pd


# ==================================================
# 1. Configuration
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "processed"
OUTPUT = ROOT / "data" / "charts"

OUTPUT.mkdir(parents=True, exist_ok=True)

START_YEAR = 1980
END_YEAR = 2025

MIN_COLLABORATIONS = 5
TOP_DIRECTORS = 15
TOP_ACTORS_PER_DIRECTOR = 5

CHUNK_SIZE = 100_000

PERIODS = {
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2025),
    "2000–2025": (2000, 2025),
    "All years": (1980, 2025),
}

DEFAULT_PERIOD = "2000–2025"

# Conservative, explicitly documented title exclusion.
EXCLUDE_PATTERN = r"(?i)^\s*(?:WWE|WWF)\b"


# ==================================================
# 2. Load eligible movies
# ==================================================

print("\n=== CHART 8 PREPROCESSING ===")
print("\nLoading movies...")

movies = pd.read_csv(
    SOURCE / "movies.csv",
    usecols=["id", "title", "year"],
    low_memory=False
)

movies = movies[
    movies["year"].between(START_YEAR, END_YEAR)
].copy()

movies = movies.drop_duplicates(subset=["id"])

excluded = movies["title"].str.contains(
    EXCLUDE_PATTERN,
    regex=True,
    na=False
)

print(f"Original eligible movies: {len(movies):,}")
print(f"Excluded WWE/WWF titles: {excluded.sum():,}")

movies = movies.loc[
    ~excluded,
    ["id", "year"]
].copy()

print(f"Remaining movies: {len(movies):,}")

eligible_ids = set(movies["id"])


# ==================================================
# 3. Load directors
# ==================================================

print("\nLoading directors...")

directors = pd.read_csv(
    SOURCE / "movie_directors.csv",
    usecols=["id", "director"]
)

directors["director"] = (
    directors["director"]
    .astype("string")
    .str.strip()
)

directors = directors.dropna(subset=["director"])

directors = directors[
    directors["director"].ne("").fillna(False)
].copy()

directors = directors[
    directors["id"].isin(eligible_ids)
].copy()

directors = directors.drop_duplicates(
    subset=["id", "director"]
)

print(f"Eligible director records: {len(directors):,}")


# ==================================================
# 4. Create temporary relationship database
# ==================================================

# SQLite prevents duplicate movie-director-actor
# relationships, including duplicates across chunks.

with TemporaryDirectory() as temp_dir:

    db_path = Path(temp_dir) / "chart8_temp.sqlite"

    connection = sqlite3.connect(db_path)

    connection.execute("""
        CREATE TABLE relationships (
            movie_id INTEGER NOT NULL,
            director TEXT NOT NULL,
            actor TEXT NOT NULL,
            year INTEGER NOT NULL,

            PRIMARY KEY (movie_id, director, actor)
        )
    """)

    # Join director credits to movie years once.
    director_movies = directors.merge(
        movies,
        on="id",
        how="inner",
        validate="many_to_one"
    )

    director_movies = director_movies.rename(
        columns={"id": "movie_id"}
    )

    director_movies = director_movies[
        ["movie_id", "director", "year"]
    ]

    print("\nProcessing cast in chunks...")

    cast_chunks = pd.read_csv(
        SOURCE / "movie_cast.csv",
        usecols=["id", "actor"],
        chunksize=CHUNK_SIZE
    )

    for chunk_number, cast in enumerate(
        cast_chunks,
        start=1
    ):

        cast["actor"] = (
            cast["actor"]
            .astype("string")
            .str.strip()
        )

        cast = cast.dropna(subset=["actor"])

        cast = cast[
            cast["actor"].ne("").fillna(False)
        ].copy()

        cast = cast[
            cast["id"].isin(eligible_ids)
        ].copy()

        cast = cast.drop_duplicates(
            subset=["id", "actor"]
        )

        # Generate relationships only for matching
        # movie IDs in the current cast chunk.
        relationships = director_movies.merge(
            cast,
            left_on="movie_id",
            right_on="id",
            how="inner"
        )

        relationships = relationships[
            ["movie_id", "director", "actor", "year"]
        ]

        relationships = relationships.drop_duplicates(
            subset=["movie_id", "director", "actor"]
        )

        records = (
            (
                int(row.movie_id),
                str(row.director),
                str(row.actor),
                int(row.year)
            )
            for row in relationships.itertuples(index=False)
        )

        connection.executemany(
            """
            INSERT OR IGNORE INTO relationships
            (movie_id, director, actor, year)
            VALUES (?, ?, ?, ?)
            """,
            records
        )

        connection.commit()

        print(
            f"Processed cast chunk {chunk_number}",
            flush=True
        )

        del relationships

    del director_movies
    del directors

    # ==================================================
    # 5. Index and inspect relationships
    # ==================================================

    print("\nCreating database indexes...")

    connection.execute("""
        CREATE INDEX idx_relationship_year
        ON relationships (year)
    """)

    connection.execute("""
        CREATE INDEX idx_relationship_pair
        ON relationships (director, actor)
    """)

    connection.commit()

    relationship_count = connection.execute(
        "SELECT COUNT(*) FROM relationships"
    ).fetchone()[0]

    print(
        f"Unique movie-level relationships: "
        f"{relationship_count:,}"
    )

    # ==================================================
    # 6. Prepare output files
    # ==================================================

    ALL_EDGES = OUTPUT / "director_actor_all_edges.csv"
    DISPLAY_EDGES = OUTPUT / "director_actor_edges.csv"
    DISPLAY_NODES = OUTPUT / "director_actor_nodes.csv"

    all_fields = [
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
        "id",
        "name",
        "type",
        "degree"
    ]

    # ==================================================
    # 7. Aggregate each period independently
    # ==================================================

    print("\nAggregating period-specific collaborations...")

    with (
        open(
            ALL_EDGES,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as all_file,

        open(
            DISPLAY_EDGES,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as display_file,

        open(
            DISPLAY_NODES,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as node_file
    ):

        all_writer = csv.DictWriter(
            all_file,
            fieldnames=all_fields
        )

        display_writer = csv.DictWriter(
            display_file,
            fieldnames=all_fields
        )

        node_writer = csv.DictWriter(
            node_file,
            fieldnames=node_fields
        )

        all_writer.writeheader()
        display_writer.writeheader()
        node_writer.writeheader()

        for period, (start, end) in PERIODS.items():

            print(f"\nProcessing period: {period}")

            # Count distinct shared films within
            # the selected period, not lifetime totals.
            query = """
                SELECT
                    director,
                    actor,
                    COUNT(*) AS collaboration_count,
                    MIN(year) AS first_year,
                    MAX(year) AS last_year

                FROM relationships

                WHERE year BETWEEN ? AND ?

                GROUP BY director, actor

                ORDER BY director, actor
            """

            cursor = connection.execute(
                query,
                (start, end)
            )

            qualifying_edges = []

            all_count = 0

            # ==================================================
            # 8. Preserve all aggregated relationships
            # ==================================================

            for (
                director,
                actor,
                count,
                first_year,
                last_year
            ) in cursor:

                edge = {
                    "period": period,
                    "source": f"director:{director}",
                    "target": f"actor:{actor}",
                    "director": director,
                    "actor": actor,
                    "collaboration_count": int(count),
                    "first_year": int(first_year),
                    "last_year": int(last_year)
                }

                all_writer.writerow(edge)
                all_count += 1

                if count >= MIN_COLLABORATIONS:
                    qualifying_edges.append(edge)

            print(
                f"All aggregated edges: {all_count:,}"
            )

            print(
                f"Edges with {MIN_COLLABORATIONS}+ "
                f"collaborations: {len(qualifying_edges):,}"
            )

            # ==================================================
            # 9. Rank directors
            # ==================================================

            director_stats = defaultdict(
                lambda: {
                    "actors": set(),
                    "total_weight": 0
                }
            )

            for edge in qualifying_edges:

                director = edge["director"]

                director_stats[director]["actors"].add(
                    edge["actor"]
                )

                director_stats[director]["total_weight"] += (
                    edge["collaboration_count"]
                )

            ranked_directors = sorted(
                director_stats,
                key=lambda d: (
                    -len(director_stats[d]["actors"]),
                    -director_stats[d]["total_weight"],
                    d
                )
            )

            selected_directors = set(
                ranked_directors[:TOP_DIRECTORS]
            )

            # ==================================================
            # 10. Select five actors per director
            # ==================================================

            by_director = defaultdict(list)

            for edge in qualifying_edges:

                if edge["director"] in selected_directors:

                    by_director[
                        edge["director"]
                    ].append(edge)

            selected_edges = []

            for director in ranked_directors[:TOP_DIRECTORS]:

                candidates = sorted(
                    by_director[director],
                    key=lambda e: (
                        -e["collaboration_count"],
                        e["actor"]
                    )
                )

                selected_edges.extend(
                    candidates[:TOP_ACTORS_PER_DIRECTOR]
                )

            # Deterministic output ordering.
            selected_edges.sort(
                key=lambda e: (
                    -e["collaboration_count"],
                    e["director"],
                    e["actor"]
                )
            )

            for edge in selected_edges:
                display_writer.writerow(edge)

            # ==================================================
            # 11. Generate nodes from displayed edges
            # ==================================================

            nodes = {}

            connections = defaultdict(set)

            for edge in selected_edges:

                source = edge["source"]
                target = edge["target"]

                nodes[source] = {
                    "period": period,
                    "id": source,
                    "name": edge["director"],
                    "type": "Director"
                }

                nodes[target] = {
                    "period": period,
                    "id": target,
                    "name": edge["actor"],
                    "type": "Actor"
                }

                connections[source].add(target)
                connections[target].add(source)

            for node_id in sorted(nodes):

                node = nodes[node_id]

                node["degree"] = len(
                    connections[node_id]
                )

                node_writer.writerow(node)

            director_total = sum(
                node["type"] == "Director"
                for node in nodes.values()
            )

            actor_total = sum(
                node["type"] == "Actor"
                for node in nodes.values()
            )

            print(
                f"Display network: "
                f"{len(selected_edges)} edges, "
                f"{director_total} directors, "
                f"{actor_total} actors"
            )

            if period == DEFAULT_PERIOD:

                print("\nDEFAULT NETWORK — TOP 10 EDGES")

                for edge in selected_edges[:10]:

                    print(
                        f"{edge['director']} + "
                        f"{edge['actor']}: "
                        f"{edge['collaboration_count']}"
                    )

    connection.close()


# ==================================================
# 12. Completion
# ==================================================

print("\n=== CHART 8 PREPROCESSING COMPLETE ===")

print("\nGenerated files:")

print(ALL_EDGES)
print(DISPLAY_EDGES)
print(DISPLAY_NODES)

print("\nDefault period:", DEFAULT_PERIOD)
print("Minimum collaborations:", MIN_COLLABORATIONS)
print("Maximum directors:", TOP_DIRECTORS)

print(
    "Maximum display edges:",
    TOP_DIRECTORS * TOP_ACTORS_PER_DIRECTOR
)
