
"""
FIT3179 DV2 — Chart 12
Australian vs Global Movie Genre Representation

Inputs:
    data/processed/movies.csv
    data/processed/movie_genres.csv
    data/processed/movie_countries.csv

Output:
    data/charts/australia_global_genres.csv

Method:
    - 1980–2025 inclusive
    - Australian productions identified by credited production country
    - Global includes all eligible movies, including Australian movies
    - Full genre counting
    - Percentages calculated using distinct genre-classified movies
"""

from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "data" / "charts"

OUTPUT.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT / "australia_global_genres.csv"


# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

START_YEAR = 1980
END_YEAR = 2025

PERIODS = {
    "All years": (1980, 2025),
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2025),
}


# --------------------------------------------------
# 3. Load source data
# --------------------------------------------------

print("Loading processed datasets...")

movies = pd.read_csv(
    PROCESSED / "movies.csv",
    usecols=["id", "year"]
)

genres = pd.read_csv(
    PROCESSED / "movie_genres.csv",
    usecols=["id", "genre"]
)

countries = pd.read_csv(
    PROCESSED / "movie_countries.csv",
    usecols=["id", "country"]
)


# --------------------------------------------------
# 4. Prepare eligible movies
# --------------------------------------------------

movies["year"] = pd.to_numeric(
    movies["year"],
    errors="coerce"
)

movies = movies[
    movies["year"].between(START_YEAR, END_YEAR)
].copy()

# A movie should only appear once in the movie table.
if movies["id"].duplicated().any():
    raise ValueError("Duplicate movie IDs found in movies.csv")

eligible_ids = set(movies["id"])


# --------------------------------------------------
# 5. Prepare movie-genre relationships
# --------------------------------------------------

genres = genres[
    genres["id"].isin(eligible_ids)
].copy()

genres["genre"] = genres["genre"].astype("string").str.strip()

genres = genres[
    genres["genre"].notna()
    & genres["genre"].ne("")
].copy()

# Full genre counting, without duplicate associations.
genres = genres.drop_duplicates(
    subset=["id", "genre"]
)

# Only movies with at least one genre enter the denominator.
genre_movie_ids = set(genres["id"])

movies = movies[
    movies["id"].isin(genre_movie_ids)
].copy()

# Attach authoritative movie years.
genres = genres.merge(
    movies[["id", "year"]],
    on="id",
    how="inner",
    validate="many_to_one"
)

genre_names = sorted(genres["genre"].unique().tolist())


# --------------------------------------------------
# 6. Identify Australian productions
# --------------------------------------------------

countries = countries[
    countries["id"].isin(genre_movie_ids)
].copy()

countries["country"] = (
    countries["country"].astype("string").str.strip()
)

australian_ids = set(
    countries.loc[
        countries["country"].eq("Australia").fillna(False),
        "id"
    ]
)

print(f"Eligible global movies: {len(movies):,}")
print(f"Eligible Australian movies: {len(australian_ids):,}")
print(f"Genres: {len(genre_names)}")


# --------------------------------------------------
# 7. Calculate genre representation
# --------------------------------------------------

records = []

for period, (start, end) in PERIODS.items():

    period_movies = movies[
        movies["year"].between(start, end)
    ]

    period_genres = genres[
        genres["year"].between(start, end)
    ]

    for region in ["Australia", "Global"]:

        if region == "Australia":
            region_movies = period_movies[
                period_movies["id"].isin(australian_ids)
            ]

            region_genres = period_genres[
                period_genres["id"].isin(australian_ids)
            ]

        else:
            region_movies = period_movies
            region_genres = period_genres

        total_movies = region_movies["id"].nunique()

        genre_counts = (
            region_genres
            .groupby("genre")["id"]
            .nunique()
            .to_dict()
        )

        for genre in genre_names:

            movie_count = int(genre_counts.get(genre, 0))

            percentage = (
                movie_count / total_movies * 100
                if total_movies > 0
                else 0.0
            )

            records.append({
                "period": period,
                "genre": genre,
                "region": region,
                "movie_count": movie_count,
                "total_movies": int(total_movies),
                "percentage": round(percentage, 4),
            })


# --------------------------------------------------
# 8. Export
# --------------------------------------------------

result = pd.DataFrame(records)

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nChart 12 preprocessing complete.")
print(f"Output: {OUTPUT_FILE}")
print(f"Rows: {len(result):,}")
print(f"Periods: {result['period'].nunique()}")
print(f"Regions: {result['region'].nunique()}")
print(f"Genres: {result['genre'].nunique()}")

print("\nFirst 10 records:")
print(result.head(10).to_string(index=False))
