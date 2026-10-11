
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

source = pd.read_csv(
    ROOT / "data/charts/genre_by_year.csv"
)

chart4 = pd.read_csv(
    ROOT / "data/charts/genre_by_decade.csv"
)

# 1. Overall counts must remain consistent
assert (
    source["movie_count"].sum()
    == chart4["movie_count"].sum()
)

print("PASS: Overall genre counts match Chart 2")

# 2. No duplicate genre-period combinations
assert not chart4.duplicated(
    subset=["decade", "genre"]
).any()

print("PASS: No duplicate genre-period combinations")

# 3. All periods must be valid
valid_decades = {
    "1980s", "1990s", "2000s",
    "2010s", "2020–2025"
}

assert set(chart4["decade"]) == valid_decades

print("PASS: All five periods are present")

# 4. No negative counts
assert (chart4["movie_count"] >= 0).all()

print("PASS: No negative movie counts")

# 5. Assignment shares should total 100% per period
shares = chart4.groupby(
    "decade"
)["genre_assignment_share"].sum()

assert ((shares - 100).abs() < 0.0001).all()

print("PASS: Genre assignment shares total 100%")

# 6. Period lengths must be correct
expected_years = {
    "1980s": 10,
    "1990s": 10,
    "2000s": 10,
    "2010s": 10,
    "2020–2025": 6
}

for decade, years in expected_years.items():
    actual = chart4.loc[
        chart4["decade"] == decade,
        "period_years"
    ]

    assert (actual == years).all()

print("PASS: Period lengths are correct")

print("\nAll Chart 4 validation checks passed!")
