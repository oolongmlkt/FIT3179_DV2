
"""
scripts/prepare_chart7.py

FIT3179 DV2 — Chart 7
IMDb vs TMDB Audience Ratings by Genre.

Produces:
    data/charts/genre_ratings.csv

Methodology:
- Released-year range: 1980–2025
- IMDb and TMDB ratings must both be valid (0 < rating <= 10)
- At least 100 votes on BOTH platforms
- Paired sample: both platform means use identical movies
- Full genre counting
- One row per genre and period
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

MOVIES_PATH = PROCESSED / "movies.csv"
GENRES_PATH = PROCESSED / "movie_genres.csv"

OUTPUT_PATH = OUTPUT / "genre_ratings.csv"

START_YEAR = 1980
END_YEAR = 2025
MIN_VOTES = 100

# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

movies = pd.read_csv(MOVIES_PATH)
genres = pd.read_csv(GENRES_PATH)

print("=== CHART 7 PREPROCESSING ===")

print(f"Original movies: {len(movies):,}")
print(f"Original genre assignments: {len(genres):,}")

# --------------------------------------------------
# 3. Validate input structure
# --------------------------------------------------

required_movies = [
    "id",
    "year",
    "decade",
    "imdb_rating",
    "imdb_votes",
    "vote_average",
    "vote_count",
]

required_genres = [
    "id",
    "year",
    "decade",
    "genre",
]

for column in required_movies:
    if column not in movies.columns:
        raise ValueError(f"Missing movies column: {column}")

for column in required_genres:
    if column not in genres.columns:
        raise ValueError(f"Missing genres column: {column}")

if movies["id"].duplicated().any():
    raise ValueError("Duplicate movie IDs found in movies.csv")

if genres.duplicated(subset=["id", "genre"]).any():
    raise ValueError("Duplicate movie-genre assignments found")

# --------------------------------------------------
# 4. Filter eligible movies
# --------------------------------------------------

eligible = movies[
    movies["year"].between(START_YEAR, END_YEAR)
    & movies["imdb_rating"].between(0.01, 10)
    & movies["vote_average"].between(0.01, 10)
    & (movies["imdb_votes"] >= MIN_VOTES)
    & (movies["vote_count"] >= MIN_VOTES)
].copy()

print(f"\nEligible movies: {len(eligible):,}")

if eligible.empty:
    raise ValueError("No eligible movies found")

# --------------------------------------------------
# 5. Join movies to genres
# --------------------------------------------------

# Only retain ID and genre from movie_genres.csv.
# Movie metadata comes from movies.csv.

genre_lookup = genres[["id", "genre"]].copy()

genre_lookup["genre"] = genre_lookup["genre"].str.strip()

genre_lookup = genre_lookup[
    genre_lookup["genre"].notna()
    & genre_lookup["genre"].ne("")
]

joined = eligible.merge(
    genre_lookup,
    on="id",
    how="inner",
    validate="one_to_many",
)

print(f"Eligible genre assignments: {len(joined):,}")
print(f"Unique movies with genres: {joined['id'].nunique():,}")

if joined.empty:
    raise ValueError("No eligible movie-genre assignments found")

# --------------------------------------------------
# 6. Assign time periods
# --------------------------------------------------

period_map = {
    1980: "1980s",
    1990: "1990s",
    2000: "2000s",
    2010: "2010s",
    2020: "2020s",
}

joined["period"] = joined["decade"].map(period_map)

if joined["period"].isna().any():
    raise ValueError("Unexpected decade value found")

# --------------------------------------------------
# 7. Aggregate ratings
# --------------------------------------------------

def aggregate_ratings(data, group_columns):

    result = (
        data.groupby(group_columns)
        .agg(
            imdb_mean=("imdb_rating", "mean"),
            tmdb_mean=("vote_average", "mean"),
            imdb_median=("imdb_rating", "median"),
            tmdb_median=("vote_average", "median"),
            movie_count=("id", "nunique"),
        )
        .reset_index()
    )

    result["difference"] = (
        result["imdb_mean"] - result["tmdb_mean"]
    )

    return result


# Individual decades
by_decade = aggregate_ratings(
    joined,
    ["period", "genre"],
)

# All years, calculated directly from movie-level records.
# Do not average the five decade averages.
all_years = aggregate_ratings(
    joined,
    ["genre"],
)

all_years.insert(0, "period", "All years")

# Combine both aggregations.
result = pd.concat(
    [by_decade, all_years],
    ignore_index=True,
)

# --------------------------------------------------
# 8. Format and sort output
# --------------------------------------------------

period_order = [
    "1980s",
    "1990s",
    "2000s",
    "2010s",
    "2020s",
    "All years",
]

result["period"] = pd.Categorical(
    result["period"],
    categories=period_order,
    ordered=True,
)

result = result.sort_values(
    ["period", "genre"]
).reset_index(drop=True)

# Preserve precision for the CSV.
# Display rounding will be handled by Vega-Lite.

result = result[
    [
        "period",
        "genre",
        "imdb_mean",
        "tmdb_mean",
        "difference",
        "movie_count",
        "imdb_median",
        "tmdb_median",
    ]
]

# --------------------------------------------------
# 9. Validate output
# --------------------------------------------------

if result.duplicated(["period", "genre"]).any():
    raise ValueError("Duplicate period-genre combinations found")

if result.isna().any().any():
    raise ValueError("Missing values found in output")

if not result["imdb_mean"].between(0, 10).all():
    raise ValueError("Invalid IMDb means")

if not result["tmdb_mean"].between(0, 10).all():
    raise ValueError("Invalid TMDB means")

# Confirm the difference calculation.
if not (
    (result["imdb_mean"] - result["tmdb_mean"] - result["difference"])
    .abs() < 1e-9
).all():
    raise ValueError("Rating difference calculation failed")

# --------------------------------------------------
# 10. Export
# --------------------------------------------------

result.to_csv(
    OUTPUT_PATH,
    index=False,
    float_format="%.6f",
)

print("\n=== OUTPUT SUMMARY ===")

print(f"Total rows: {len(result):,}")
print(f"Unique genres: {result['genre'].nunique()}")

print("\nRows per period:")
print(
    result.groupby("period", observed=True)
    .size()
    .to_string()
)

print("\nGenre sample sizes (All years):")

print(
    result[result["period"] == "All years"]
    [["genre", "movie_count"]]
    .sort_values("movie_count", ascending=False)
    .to_string(index=False)
)

print(f"\nSaved: {OUTPUT_PATH}")

print("\nChart 7 preprocessing complete.")
