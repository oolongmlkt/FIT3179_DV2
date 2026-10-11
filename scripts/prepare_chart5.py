
"""
FIT3179 DV2 — Chart 5
Budget vs Revenue Scatterplot

Creates data/charts/movie_financials.csv.

Run from the repository root:
    python scripts/prepare_chart5.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MOVIES = ROOT / "data" / "processed" / "movies.csv"
GENRES = ROOT / "data" / "processed" / "movie_genres.csv"

OUTPUT = ROOT / "data" / "charts" / "movie_financials.csv"


def main():
    # Load processed data
    movies = pd.read_csv(MOVIES, low_memory=False)
    genres = pd.read_csv(GENRES, low_memory=False)

    required_movies = [
        "id", "title", "year", "budget", "revenue"
    ]

    required_genres = ["id", "genre"]

    for col in required_movies:
        if col not in movies.columns:
            raise ValueError(f"Missing movies column: {col}")

    for col in required_genres:
        if col not in genres.columns:
            raise ValueError(f"Missing genre column: {col}")

    # Select financial information
    df = movies[required_movies].copy()

    for col in ["year", "budget", "revenue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Only use movies with valid financial records
    df = df[
        df["year"].between(1980, 2025)
        & (df["budget"] > 0)
        & (df["revenue"] > 0)
    ].copy()

    # Assign one deterministic primary genre per movie.
    # This avoids duplicating movie points in the scatterplot.
    primary_genres = (
        genres.dropna(subset=["genre"])
        .sort_values(["id", "genre"])
        .drop_duplicates(subset=["id"])
        [["id", "genre"]]
    )

    df = df.merge(
        primary_genres,
        on="id",
        how="left",
        validate="one_to_one"
    )

    df["genre"] = df["genre"].fillna("Unknown")

    # Financial metrics
    df["profit"] = df["revenue"] - df["budget"]

    df["roi"] = (
        (df["revenue"] - df["budget"])
        / df["budget"]
    ) * 100

    # Convert to USD millions for visualisation
    df["budget_m"] = df["budget"] / 1_000_000
    df["revenue_m"] = df["revenue"] / 1_000_000

    # One record per movie
    df = df.drop_duplicates(subset=["id"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT, index=False)

    print("Chart 5 dataset created successfully.")
    print(f"Movies: {len(df):,}")
    print(f"Output: {OUTPUT}")

    print("\nGenres:")
    print(df["genre"].value_counts().head(10))

    print("\nFinancial summary:")
    print(
        df[["budget_m", "revenue_m", "roi"]]
        .describe()
    )


if __name__ == "__main__":
    main()
