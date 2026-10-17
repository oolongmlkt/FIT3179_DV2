
"""
FIT3179 DV2 — Chart 12 Data Validation

Validates:
    data/charts/australia_global_genres.csv

Against:
    data/processed/movies.csv
    data/processed/movie_genres.csv
    data/processed/movie_countries.csv
"""

from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "data" / "charts" / "australia_global_genres.csv"

PERIODS = {
    "All years": (1980, 2025),
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2025),
}

REGIONS = {"Australia", "Global"}

REQUIRED_COLUMNS = [
    "period",
    "genre",
    "region",
    "movie_count",
    "total_movies",
    "percentage",
]

passed = 0
failed = 0


def check(condition, description):
    global passed, failed

    if bool(condition):
        passed += 1
        print(f"[PASS] {description}")
    else:
        failed += 1
        print(f"[FAIL] {description}")


def integer_values(series):
    values = pd.to_numeric(series, errors="coerce")
    return bool(
        values.notna().all()
        and np.isfinite(values).all()
        and (values % 1 == 0).all()
    )


def main():
    print("\nCHART 12 — DATA VALIDATION")
    print("=" * 55)

    # ----------------------------------------------
    # A. File and structural checks
    # ----------------------------------------------

    exists = OUTPUT.is_file()
    check(exists, "01. Output CSV exists")

    if not exists:
        print("\nCannot continue: output file missing.")
        raise SystemExit(1)

    df = pd.read_csv(OUTPUT)

    check(len(df) > 0, "02. Output is not empty")

    check(
        list(df.columns) == REQUIRED_COLUMNS,
        "03. Output columns match expected schema"
    )

    if not set(REQUIRED_COLUMNS).issubset(df.columns):
        raise SystemExit("Missing required columns.")

    check(
        not df[REQUIRED_COLUMNS].isna().any().any(),
        "04. No missing values"
    )

    check(
        not df.duplicated(
            ["period", "genre", "region"]
        ).any(),
        "05. No duplicate period-genre-region records"
    )

    check(
        set(df["period"]) == set(PERIODS),
        "06. Exactly six expected periods"
    )

    check(
        set(df["region"]) == REGIONS,
        "07. Exactly two expected regions"
    )

    # ----------------------------------------------
    # B. Genre and completeness checks
    # ----------------------------------------------

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

    movies["year"] = pd.to_numeric(
        movies["year"], errors="coerce"
    )

    movies = movies[
        movies["year"].between(1980, 2025)
    ].copy()

    genres["genre"] = (
        genres["genre"].astype("string").str.strip()
    )

    genres = genres[
        genres["genre"].notna()
        & genres["genre"].ne("")
        & genres["id"].isin(movies["id"])
    ].drop_duplicates(["id", "genre"])

    expected_genres = set(genres["genre"].unique())

    check(
        set(df["genre"]) == expected_genres,
        "08. Genres match processed source data"
    )

    expected_rows = (
        len(PERIODS) * len(REGIONS) * len(expected_genres)
    )

    check(
        len(df) == expected_rows,
        "09. Correct number of output records"
    )

    expected_keys = pd.MultiIndex.from_product(
        [
            list(PERIODS),
            sorted(expected_genres),
            sorted(REGIONS),
        ],
        names=["period", "genre", "region"]
    )

    actual_keys = pd.MultiIndex.from_frame(
        df[["period", "genre", "region"]]
    )

    check(
        len(actual_keys) == len(expected_keys)
        and actual_keys.is_unique
        and actual_keys.symmetric_difference(
            expected_keys
        ).empty,
        "10. Every expected combination exists exactly once"
    )

    # ----------------------------------------------
    # C. Numeric integrity
    # ----------------------------------------------

    counts = pd.to_numeric(
        df["movie_count"], errors="coerce"
    )

    totals = pd.to_numeric(
        df["total_movies"], errors="coerce"
    )

    percentages = pd.to_numeric(
        df["percentage"], errors="coerce"
    )

    check(
        integer_values(df["movie_count"]),
        "11. Movie counts are integers"
    )

    check(
        integer_values(df["total_movies"]),
        "12. Total movie counts are integers"
    )

    check(
        counts.notna().all() and counts.ge(0).all(),
        "13. Movie counts are non-negative"
    )

    check(
        totals.notna().all() and totals.gt(0).all(),
        "14. All denominators are positive"
    )

    check(
        counts.le(totals).all(),
        "15. Genre counts never exceed denominators"
    )

    check(
        percentages.notna().all()
        and np.isfinite(percentages).all()
        and percentages.between(0, 100).all(),
        "16. Percentages are valid (0–100)"
    )

    calculated = (
        counts / totals.replace(0, np.nan) * 100
    )

    check(
        np.allclose(
            percentages,
            calculated,
            atol=0.000051,
            rtol=0,
            equal_nan=False
        ),
        "17. Percentages match counts and denominators"
    )

    check(
        df.groupby(
            ["period", "region"]
        )["total_movies"].nunique().eq(1).all(),
        "18. Denominators are consistent within groups"
    )

    # ----------------------------------------------
    # D. Independent source reconciliation
    # ----------------------------------------------

    # Only genre-classified movies are eligible.
    movies = movies[
        movies["id"].isin(genres["id"])
    ].drop_duplicates("id")

    countries["country"] = (
        countries["country"].astype("string").str.strip()
    )

    australian_ids = set(
        countries.loc[
            countries["country"].eq("Australia").fillna(False),
            "id"
        ]
    )

    genre_years = genres.merge(
        movies[["id", "year"]],
        on="id",
        how="inner",
        validate="many_to_one"
    )

    expected_records = []

    for period, (start, end) in PERIODS.items():

        period_movies = movies[
            movies["year"].between(start, end)
        ]

        period_genres = genre_years[
            genre_years["year"].between(start, end)
        ]

        for region in sorted(REGIONS):

            if region == "Australia":
                selected_ids = set(
                    period_movies.loc[
                        period_movies["id"].isin(australian_ids),
                        "id"
                    ]
                )
            else:
                selected_ids = set(period_movies["id"])

            selected_genres = period_genres[
                period_genres["id"].isin(selected_ids)
            ]

            genre_counts = (
                selected_genres.groupby("genre")["id"]
                .nunique()
                .to_dict()
            )

            for genre in sorted(expected_genres):
                expected_records.append({
                    "period": period,
                    "region": region,
                    "genre": genre,
                    "expected_count": int(
                        genre_counts.get(genre, 0)
                    ),
                    "expected_total": len(selected_ids),
                })

    expected = pd.DataFrame(expected_records)

    merged = df.merge(
        expected,
        on=["period", "region", "genre"],
        how="outer",
        validate="one_to_one",
        indicator=True
    )

    check(
        merged["_merge"].eq("both").all(),
        "19. Every output record matches a source-derived key"
    )

    check(
        merged["movie_count"].eq(
            merged["expected_count"]
        ).all(),
        "20. Genre counts reconcile with source data"
    )

    check(
        merged["total_movies"].eq(
            merged["expected_total"]
        ).all(),
        "21. Denominators reconcile with source data"
    )

    # ----------------------------------------------
    # E. Cross-period and region checks
    # ----------------------------------------------

    summary = (
        df.groupby(["period", "region"])["total_movies"]
        .first()
        .unstack()
    )

    check(
        summary["Australia"].le(summary["Global"]).all(),
        "22. Australia is a subset of Global"
    )

    decade_names = [
        "1980s", "1990s", "2000s", "2010s", "2020s"
    ]

    check(
        (
            summary.loc[decade_names].sum()
            == summary.loc["All years"]
        ).all(),
        "23. Decade denominators sum to All years"
    )

    all_years = df[
        df["period"] == "All years"
    ].set_index(["region", "genre"])["movie_count"]

    decade_counts = (
        df[df["period"].isin(decade_names)]
        .groupby(["region", "genre"])["movie_count"]
        .sum()
    )

    check(
        all_years.sort_index().equals(
            decade_counts.sort_index()
        ),
        "24. Genre counts reconcile across all periods"
    )

    # ----------------------------------------------
    # Final report
    # ----------------------------------------------

    print("\n" + "=" * 55)
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print(f"TOTAL:  {passed + failed}")

    print("\nPeriod summary:")
    print(summary.to_string())

    if failed:
        print("\nVALIDATION FAILED")
        raise SystemExit(1)

    print("\nALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
