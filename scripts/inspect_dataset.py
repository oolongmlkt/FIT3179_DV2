
"""
scripts/inspect_dataset.py
FIT3179 DV2 - TMDB Dataset Inspection
=====================================

Analyses the suitability of the TMDB dataset for the
Movie Collections data visualisation project.

Input:
    data/raw/tmdb_movies.csv

Outputs:
    reports/dataset_report.txt
    reports/missing_values.csv
    reports/decade_distribution.csv
    reports/genre_distribution.csv
    reports/country_distribution.csv
    reports/financial_by_decade.csv
    reports/chart_feasibility.csv

Requirements:
    pip install pandas

Run:
    python3 scripts/inspect_dataset.py

The original dataset is never modified.
"""

from pathlib import Path
from collections import Counter
from contextlib import redirect_stdout
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/tmdb_movies.csv")
REPORT_DIR = Path("reports")

CHUNK_SIZE = 100_000

# Thresholds are preliminary analysis assumptions.
# They can be adjusted after inspecting the results.
MIN_TMDB_VOTES = 100
MIN_IMDB_VOTES = 100

CURRENT_YEAR = 2026

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def heading(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def percentage(count, total):
    if total == 0:
        return 0.0
    return count / total * 100


def split_values(value):
    """
    Split comma-separated text fields such as:
    genres, production_countries, cast, director.

    Note: splitting names on commas is only a heuristic.
    A person's name containing a comma may be split incorrectly.
    """
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def print_counter(counter, limit=20):
    for name, count in counter.most_common(limit):
        print(f"{name:<38} {count:>10,}")


def save_counter(counter, filename, column_name):
    df = pd.DataFrame(
        counter.most_common(),
        columns=[column_name, "movie_count"]
    )
    df.to_csv(REPORT_DIR / filename, index=False)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyse_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    heading("FIT3179 DV2 - TMDB DATASET FEASIBILITY REPORT")

    total_rows = 0
    columns = None

    missing_counts = None
    blank_counts = None

    status_counts = Counter()
    decade_counts = Counter()
    year_counts = Counter()
    genre_counts = Counter()
    country_counts = Counter()
    studio_counts = Counter()

    genre_count_per_movie = Counter()
    country_count_per_movie = Counter()

    financial_decade = {}

    seen_ids = set()
    duplicate_ids = 0
    missing_ids = 0

    released_count = 0
    valid_date_count = 0
    future_date_count = 0

    valid_genre_count = 0
    valid_country_count = 0

    positive_budget_count = 0
    positive_revenue_count = 0
    valid_financial_count = 0

    valid_tmdb_rating_count = 0
    valid_imdb_rating_count = 0
    valid_both_ratings_count = 0

    valid_director_count = 0
    valid_cast_count = 0
    valid_collaboration_count = 0

    valid_studio_count = 0

    australian_count = 0
    australian_genre_count = 0
    australian_financial_count = 0
    australian_rating_count = 0

    # Chart-specific counts.
    chart_counts = Counter()

    # Collect valid values for descriptive statistics.
    # These lists contain only selected numeric fields,
    # not the complete dataset.
    budgets = []
    revenues = []
    roi_values = []

    tmdb_ratings = []
    imdb_ratings = []

    # ========================================================
    # READ DATASET IN CHUNKS
    # ========================================================

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        print(
            f"Processing chunk {chunk_number}...",
            flush=True
        )

        if columns is None:
            columns = chunk.columns.tolist()
            missing_counts = pd.Series(
                0, index=columns, dtype="int64"
            )
            blank_counts = pd.Series(
                0, index=columns, dtype="int64"
            )

        total_rows += len(chunk)

        missing_counts += chunk.isna().sum()

        for col in columns:
            if pd.api.types.is_object_dtype(chunk[col]):
                blank_counts[col] += (
                    chunk[col].fillna("")
                    .astype(str)
                    .str.strip()
                    .eq("")
                    .sum()
                    - chunk[col].isna().sum()
                )

        # ----------------------------------------------------
        # 1. DUPLICATE IDS
        # ----------------------------------------------------

        for movie_id in chunk["id"]:
            if pd.isna(movie_id):
                missing_ids += 1
            elif movie_id in seen_ids:
                duplicate_ids += 1
            else:
                seen_ids.add(movie_id)

        # ----------------------------------------------------
        # 2. STATUS
        # ----------------------------------------------------

        status_counts.update(
            chunk["status"]
            .fillna("Unknown")
            .astype(str)
            .tolist()
        )

        released = chunk["status"].eq("Released")
        released_count += int(released.sum())

        # ----------------------------------------------------
        # 3. RELEASE DATES
        # ----------------------------------------------------

        dates = pd.to_datetime(
            chunk["release_date"],
            errors="coerce"
        )

        years = dates.dt.year

        valid_dates = dates.notna()
        valid_date_count += int(valid_dates.sum())

        future_date_count += int(
            (years > CURRENT_YEAR).sum()
        )

        valid_years = years[
            years.between(1880, CURRENT_YEAR)
        ]

        year_counts.update(
            valid_years.astype(int).tolist()
        )

        decades = (years // 10 * 10)

        decade_counts.update(
            decades[
                years.between(1880, CURRENT_YEAR)
            ].astype(int).tolist()
        )

        # ----------------------------------------------------
        # 4. GENRES
        # ----------------------------------------------------

        genres_list = chunk["genres"].apply(split_values)

        has_genre = genres_list.apply(bool)
        valid_genre_count += int(has_genre.sum())

        for genres in genres_list:
            genre_counts.update(set(genres))
            genre_count_per_movie[len(genres)] += 1

        # ----------------------------------------------------
        # 5. PRODUCTION COUNTRIES
        # ----------------------------------------------------

        countries_list = chunk[
            "production_countries"
        ].apply(split_values)

        has_country = countries_list.apply(bool)
        valid_country_count += int(has_country.sum())

        for countries in countries_list:
            country_counts.update(set(countries))
            country_count_per_movie[len(countries)] += 1

        australia = countries_list.apply(
            lambda countries: "Australia" in countries
        )

        australian_count += int(australia.sum())

        australian_genre_count += int(
            (australia & has_genre).sum()
        )

        # ----------------------------------------------------
        # 6. FINANCIAL DATA
        # ----------------------------------------------------

        budget = numeric(chunk["budget"])
        revenue = numeric(chunk["revenue"])

        positive_budget = budget > 0
        positive_revenue = revenue > 0

        valid_financial = (
            positive_budget
            & positive_revenue
            & released
        )

        positive_budget_count += int(
            positive_budget.sum()
        )

        positive_revenue_count += int(
            positive_revenue.sum()
        )

        valid_financial_count += int(
            valid_financial.sum()
        )

        australian_financial_count += int(
            (australia & valid_financial).sum()
        )

        budgets.extend(
            budget[positive_budget].tolist()
        )

        revenues.extend(
            revenue[positive_revenue].tolist()
        )

        roi = (
            revenue[valid_financial]
            - budget[valid_financial]
        ) / budget[valid_financial]

        roi_values.extend(roi.tolist())

        for decade in decades[
            years.between(1880, CURRENT_YEAR)
        ].dropna().unique():

            decade = int(decade)

            mask = decades.eq(decade)

            if decade not in financial_decade:
                financial_decade[decade] = {
                    "total": 0,
                    "usable": 0
                }

            financial_decade[decade]["total"] += int(
                mask.sum()
            )

            financial_decade[decade]["usable"] += int(
                (mask & valid_financial).sum()
            )

        # ----------------------------------------------------
        # 7. RATINGS
        # ----------------------------------------------------

        tmdb_rating = numeric(chunk["vote_average"])
        tmdb_votes = numeric(chunk["vote_count"])

        imdb_rating = numeric(chunk["imdb_rating"])
        imdb_votes = numeric(chunk["imdb_votes"])

        valid_tmdb = (
            tmdb_rating.between(0, 10)
            & (tmdb_votes >= MIN_TMDB_VOTES)
            & released
        )

        valid_imdb = (
            imdb_rating.between(0, 10)
            & (imdb_votes >= MIN_IMDB_VOTES)
            & released
        )

        both_ratings = valid_tmdb & valid_imdb

        valid_tmdb_rating_count += int(
            valid_tmdb.sum()
        )

        valid_imdb_rating_count += int(
            valid_imdb.sum()
        )

        valid_both_ratings_count += int(
            both_ratings.sum()
        )

        tmdb_ratings.extend(
            tmdb_rating[valid_tmdb].tolist()
        )

        imdb_ratings.extend(
            imdb_rating[valid_imdb].tolist()
        )

        australian_rating_count += int(
            (australia & both_ratings).sum()
        )

        # ----------------------------------------------------
        # 8. DIRECTORS AND CAST
        # ----------------------------------------------------

        directors_list = chunk[
            "director"
        ].apply(split_values)

        cast_list = chunk[
            "cast"
        ].apply(split_values)

        has_director = directors_list.apply(bool)
        has_cast = cast_list.apply(bool)

        valid_director_count += int(
            has_director.sum()
        )

        valid_cast_count += int(
            has_cast.sum()
        )

        collaboration = (
            has_director
            & has_cast
            & released
        )

        valid_collaboration_count += int(
            collaboration.sum()
        )

        # ----------------------------------------------------
        # 9. STUDIOS
        # ----------------------------------------------------

        studios_list = chunk[
            "production_companies"
        ].apply(split_values)

        has_studio = studios_list.apply(bool)

        valid_studio_count += int(
            has_studio.sum()
        )

        for studios in studios_list:
            studio_counts.update(set(studios))

        # ----------------------------------------------------
        # 10. CHART FEASIBILITY
        # ----------------------------------------------------

        valid_period = years.between(
            1880, CURRENT_YEAR
        )

        chart_counts["1_world_choropleth"] += int(
            (released & valid_period & has_country).sum()
        )

        chart_counts["2_genre_streamgraph"] += int(
            (released & valid_period & has_genre).sum()
        )

        chart_counts["3_genre_symbol_map"] += int(
            (
                released
                & valid_period
                & has_genre
                & has_country
            ).sum()
        )

        chart_counts["4_genre_decade_heatmap"] += int(
            (released & valid_period & has_genre).sum()
        )

        chart_counts["5_budget_revenue_scatter"] += int(
            (valid_financial & valid_period).sum()
        )

        chart_counts["6_financial_performance"] += int(
            (
                valid_financial
                & valid_period
                & has_genre
            ).sum()
        )

        chart_counts["7_ratings_dot_plot"] += int(
            (both_ratings & has_genre & valid_period).sum()
        )

        chart_counts["8_collaboration_network"] += int(
            (collaboration & valid_period).sum()
        )

        chart_counts["9_studio_bump_chart"] += int(
            (
                released
                & valid_period
                & has_studio
            ).sum()
        )

        chart_counts["10_collaboration_lollipop"] += int(
            (collaboration & valid_period).sum()
        )

        chart_counts["11_australian_map"] += 0

        chart_counts["12_australia_global_genres"] += int(
            (
                released
                & valid_period
                & australia
                & has_genre
            ).sum()
        )

    # ========================================================
    # REPORT RESULTS
    # ========================================================

    heading("1. DATASET OVERVIEW")

    print(f"Total records: {total_rows:,}")
    print(f"Total columns: {len(columns)}")
    print(f"Unique movie IDs: {len(seen_ids):,}")
    print(f"Duplicate movie IDs: {duplicate_ids:,}")
    print(f"Missing movie IDs: {missing_ids:,}")

    heading("2. MISSING VALUES")

    missing_df = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_pct": (
            missing_counts / total_rows * 100
        ).round(2),
        "additional_blank_count": blank_counts
    }).sort_values(
        "missing_pct",
        ascending=False
    )

    print(missing_df.to_string())

    missing_df.to_csv(
        REPORT_DIR / "missing_values.csv"
    )

    heading("3. MOVIE STATUS")

    print_counter(status_counts)

    print(
        f"\nReleased movies: {released_count:,} "
        f"({percentage(released_count, total_rows):.2f}%)"
    )

    heading("4. RELEASE DATES")

    print(f"Valid dates: {valid_date_count:,}")
    print(f"Future release years: {future_date_count:,}")

    print("\nMovies by decade:")

    for decade in sorted(decade_counts):
        print(
            f"{decade}s: {decade_counts[decade]:,}"
        )

    pd.DataFrame(
        sorted(decade_counts.items()),
        columns=["decade", "movie_count"]
    ).to_csv(
        REPORT_DIR / "decade_distribution.csv",
        index=False
    )

    heading("5. GENRES")

    print(f"Movies with genres: {valid_genre_count:,}")

    print("\nTop genres:")
    print_counter(genre_counts)

    print("\nNumber of genres per movie:")
    for count in sorted(genre_count_per_movie):
        print(
            f"{count} genres: "
            f"{genre_count_per_movie[count]:,}"
        )

    save_counter(
        genre_counts,
        "genre_distribution.csv",
        "genre"
    )

    heading("6. PRODUCTION COUNTRIES")

    print(
        f"Movies with production countries: "
        f"{valid_country_count:,}"
    )

    print("\nTop production countries:")
    print_counter(country_counts, 30)

    print("\nNumber of countries per movie:")
    for count in sorted(country_count_per_movie):
        print(
            f"{count} countries: "
            f"{country_count_per_movie[count]:,}"
        )

    print(
        f"\nAustralian movies: "
        f"{australian_count:,}"
    )

    save_counter(
        country_counts,
        "country_distribution.csv",
        "country"
    )

    heading("7. FINANCIAL DATA")

    print(
        f"Positive budgets: "
        f"{positive_budget_count:,} "
        f"({percentage(positive_budget_count, total_rows):.2f}%)"
    )

    print(
        f"Positive revenues: "
        f"{positive_revenue_count:,} "
        f"({percentage(positive_revenue_count, total_rows):.2f}%)"
    )

    print(
        f"Released movies with both: "
        f"{valid_financial_count:,}"
    )

    if budgets:
        print(
            f"Median positive budget: "
            f"${pd.Series(budgets).median():,.0f}"
        )

    if revenues:
        print(
            f"Median positive revenue: "
            f"${pd.Series(revenues).median():,.0f}"
        )

    if roi_values:
        print(
            f"Median ROI: "
            f"{pd.Series(roi_values).median() * 100:.2f}%"
        )

    financial_rows = []

    for decade in sorted(financial_decade):
        values = financial_decade[decade]

        financial_rows.append({
            "decade": decade,
            "total_movies": values["total"],
            "usable_financial_movies": values["usable"],
            "usable_pct": round(
                percentage(
                    values["usable"],
                    values["total"]
                ), 2
            )
        })

    pd.DataFrame(financial_rows).to_csv(
        REPORT_DIR / "financial_by_decade.csv",
        index=False
    )

    heading("8. RATINGS")

    print(
        f"Usable TMDB ratings: "
        f"{valid_tmdb_rating_count:,}"
    )

    print(
        f"Usable IMDb ratings: "
        f"{valid_imdb_rating_count:,}"
    )

    print(
        f"Movies with both ratings: "
        f"{valid_both_ratings_count:,}"
    )

    if tmdb_ratings:
        print(
            f"Median TMDB rating: "
            f"{pd.Series(tmdb_ratings).median():.2f}"
        )

    if imdb_ratings:
        print(
            f"Median IMDb rating: "
            f"{pd.Series(imdb_ratings).median():.2f}"
        )

    heading("9. PEOPLE AND COLLABORATIONS")

    print(
        f"Movies with directors: "
        f"{valid_director_count:,}"
    )

    print(
        f"Movies with cast: "
        f"{valid_cast_count:,}"
    )

    print(
        f"Released movies with both: "
        f"{valid_collaboration_count:,}"
    )

    heading("10. PRODUCTION STUDIOS")

    print(
        f"Movies with studio information: "
        f"{valid_studio_count:,}"
    )

    print("\nTop studios:")
    print_counter(studio_counts, 20)

    heading("11. AUSTRALIAN MOVIES")

    print(
        f"Movies with Australia as a production country: "
        f"{australian_count:,}"
    )

    print(
        f"Australian movies with genres: "
        f"{australian_genre_count:,}"
    )

    print(
        f"Australian movies with financial data: "
        f"{australian_financial_count:,}"
    )

    print(
        f"Australian movies with both ratings: "
        f"{australian_rating_count:,}"
    )

    heading("12. VISUALISATION FEASIBILITY")

    feasibility_rows = []

    for chart, count in chart_counts.items():

        if chart == "11_australian_map":
            note = "Requires Screen Australia data"
        else:
            note = "Preliminary eligible record count"

        feasibility_rows.append({
            "chart": chart,
            "eligible_movies": count,
            "note": note
        })

        print(f"{chart:<40} {count:>10,}  {note}")

    pd.DataFrame(feasibility_rows).to_csv(
        REPORT_DIR / "chart_feasibility.csv",
        index=False
    )

    heading("INSPECTION COMPLETE")

    print(f"Reports saved to: {REPORT_DIR.resolve()}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    report_path = REPORT_DIR / "dataset_report.txt"

    with report_path.open(
        "w",
        encoding="utf-8"
    ) as report_file:

        with redirect_stdout(report_file):
            analyse_dataset()

    print("Dataset inspection completed.")
    print(f"Main report: {report_path.resolve()}")
    print(f"Additional CSV reports: {REPORT_DIR.resolve()}")
