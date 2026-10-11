
"""
FIT3179 DV2 — Chart 6 Data Preparation

Chart: Median reported ROI by movie genre (1980–2025)

Input:
    data/processed/movies.csv
    data/processed/movie_genres.csv

Output:
    data/charts/genre_financials.csv

Run from the project root:
    python scripts/prepare_chart6.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MOVIES_PATH = ROOT / "data" / "processed" / "movies.csv"
GENRES_PATH = ROOT / "data" / "processed" / "movie_genres.csv"

OUTPUT_DIR = ROOT / "data" / "charts"
OUTPUT_PATH = OUTPUT_DIR / "genre_financials.csv"

START_YEAR = 1980
END_YEAR = 2025


def prepare_chart6():

    print("\nPreparing Chart 6 dataset...\n")

    # -----------------------------------------
    # 1. Load processed datasets
    # -----------------------------------------

    movies = pd.read_csv(
        MOVIES_PATH,
        usecols=[
            "id",
            "year",
            "budget",
            "revenue"
        ]
    )

    genres = pd.read_csv(
        GENRES_PATH,
        usecols=[
            "id",
            "genre"
        ]
    )

    print(f"Movies loaded: {len(movies):,}")
    print(f"Genre records loaded: {len(genres):,}")

    # -----------------------------------------
    # 2. Validate movie identifiers
    # -----------------------------------------

    if movies["id"].isna().any():
        raise ValueError("Missing movie IDs found.")

    if movies["id"].duplicated().any():
        raise ValueError("Duplicate movie IDs found.")

    # -----------------------------------------
    # 3. Clean financial data
    # -----------------------------------------

    for column in ["year", "budget", "revenue"]:
        movies[column] = pd.to_numeric(
            movies[column],
            errors="coerce"
        )

    movies = movies.loc[
        movies["year"].between(START_YEAR, END_YEAR)
        & movies["budget"].gt(0)
        & movies["revenue"].gt(0)
    ].copy()

    print(f"Financially eligible movies: {len(movies):,}")

    # -----------------------------------------
    # 4. Calculate individual movie ROI
    # -----------------------------------------

    movies["roi"] = (
        (movies["revenue"] - movies["budget"])
        / movies["budget"]
    ) * 100

    movies["decade"] = (
        movies["year"] // 10 * 10
    ).astype(int)

    # -----------------------------------------
    # 5. Join movies with genres
    # -----------------------------------------

    genres["genre"] = genres["genre"].astype("string").str.strip()

    genres = genres.dropna(subset=["id", "genre"])

    genres = genres.loc[
        genres["genre"].ne("")
    ].copy()

    # Prevent duplicate movie-genre combinations.
    genres = genres.drop_duplicates(
        subset=["id", "genre"]
    )

    combined = movies.merge(
        genres,
        on="id",
        how="inner",
        validate="one_to_many"
    )

    print(f"Eligible movie-genre records: {len(combined):,}")
    print(f"Unique eligible genres: {combined['genre'].nunique()}")

    if combined.empty:
        raise ValueError("No eligible movie-genre records found.")

    # -----------------------------------------
    # 6. Aggregate financial statistics
    # -----------------------------------------

    def aggregate(data, group_columns):

        return (
            data.groupby(
                group_columns,
                as_index=False
            )
            .agg(
                movie_count=("id", "nunique"),
                median_roi=("roi", "median"),
                median_budget=("budget", "median"),
                median_revenue=("revenue", "median")
            )
        )

    # Individual decades
    by_decade = aggregate(
        combined,
        ["genre", "decade"]
    )

    by_decade["period"] = (
        by_decade["decade"].astype(str) + "s"
    )

    # Full period: 1980–2025
    all_years = aggregate(
        combined,
        ["genre"]
    )

    all_years["period"] = "All years"

    # -----------------------------------------
    # 7. Combine results
    # -----------------------------------------

    output_columns = [
        "genre",
        "period",
        "movie_count",
        "median_roi",
        "median_budget",
        "median_revenue"
    ]

    output = pd.concat(
        [
            all_years[output_columns],
            by_decade[output_columns]
        ],
        ignore_index=True
    )

    output = output.sort_values(
        ["period", "genre"]
    ).reset_index(drop=True)

    # -----------------------------------------
    # 8. Validate output
    # -----------------------------------------

    if output.duplicated(["genre", "period"]).any():
        raise ValueError("Duplicate genre-period combinations.")

    if output[output_columns].isna().any().any():
        raise ValueError("Missing output values found.")

    if (output["movie_count"] <= 0).any():
        raise ValueError("Invalid movie counts found.")

    if (output["median_budget"] <= 0).any():
        raise ValueError("Invalid median budgets found.")

    if (output["median_revenue"] <= 0).any():
        raise ValueError("Invalid median revenues found.")

    # -----------------------------------------
    # 9. Export
    # -----------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
        float_format="%.4f"
    )

    print("\nChart 6 dataset generated successfully!")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows: {len(output):,}")

    print("\nPeriods:")
    print(output["period"].value_counts().sort_index())

    print("\nAll-years financial comparison:")

    preview = (
        output.loc[output["period"] == "All years"]
        .sort_values(
            "median_roi",
            ascending=False
        )
    )

    print(
        preview[
            ["genre", "movie_count", "median_roi"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    prepare_chart6()
