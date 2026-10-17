
from pathlib import Path
import pandas as pd
import math

# --------------------------------------------------
# File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FILE = (
    BASE_DIR / "data" / "raw"
    / "screen_australia_spend_by_location.csv"
)

MAP_FILE = (
    BASE_DIR / "data" / "charts"
    / "australia_spend.csv"
)

SUMMARY_FILE = (
    BASE_DIR / "data" / "charts"
    / "australia_spend_summary.csv"
)

raw = pd.read_csv(RAW_FILE, dtype=str)
map_df = pd.read_csv(MAP_FILE)
summary = pd.read_csv(SUMMARY_FILE)

raw.columns = raw.columns.str.strip()
raw = raw.rename(columns={raw.columns[0]: "year"})

for column in raw.columns:
    raw[column] = raw[column].str.strip()

# --------------------------------------------------
# Validation helper
# --------------------------------------------------

passed = 0
failed = 0


def check(condition, description):
    global passed, failed

    if condition:
        passed += 1
        print(f"[PASS] {description}")
    else:
        failed += 1
        print(f"[FAIL] {description}")


# --------------------------------------------------
# Expected values
# --------------------------------------------------

STATES = {
    "NSW": ("New South Wales", 147.0, -32.0),
    "QLD": ("Queensland", 145.0, -22.5),
    "SA": ("South Australia", 135.5, -30.0),
    "VIC": ("Victoria", 144.0, -37.0)
}

EXPECTED_MAP_COLUMNS = {
    "year",
    "year_start",
    "state",
    "state_code",
    "spend_m",
    "longitude",
    "latitude",
    "total_m",
    "share_pct"
}

EXPECTED_SUMMARY_COLUMNS = {
    "year",
    "year_start",
    "other_reported",
    "other_m",
    "other_is_less_than_one",
    "total_m",
    "mapped_states_m",
    "reported_difference_m"
}

EXPECTED_YEARS = [
    f"{y}/{str(y + 1)[-2:]}"
    for y in range(1995, 2025)
]

# --------------------------------------------------
# A. Structural validation (1–8)
# --------------------------------------------------

print("\nA. STRUCTURAL VALIDATION")

check(
    len(raw) == 30,
    "01. Original dataset contains 30 years"
)

check(
    len(map_df) == 120,
    "02. Map dataset contains 120 records"
)

check(
    len(summary) == 30,
    "03. Summary dataset contains 30 records"
)

check(
    set(map_df.columns) == EXPECTED_MAP_COLUMNS,
    "04. Map columns match expected schema"
)

check(
    set(summary.columns) == EXPECTED_SUMMARY_COLUMNS,
    "05. Summary columns match expected schema"
)

check(
    not map_df.isna().any().any(),
    "06. No missing map values"
)

check(
    not map_df.duplicated(
        ["year", "state_code"]
    ).any(),
    "07. No duplicate year-state combinations"
)

check(
    not summary["year"].duplicated().any(),
    "08. No duplicate summary years"
)

# --------------------------------------------------
# B. Temporal validation (9–12)
# --------------------------------------------------

print("\nB. TEMPORAL VALIDATION")

check(
    set(map_df["year"]) == set(EXPECTED_YEARS),
    "09. All expected financial years are present"
)

check(
    set(summary["year"]) == set(EXPECTED_YEARS),
    "10. Summary contains all expected years"
)

check(
    (map_df.groupby("year").size() == 4).all(),
    "11. Exactly four state records per year"
)

check(
    all(
        row.year_start == int(row.year[:4])
        for row in map_df.itertuples()
    ),
    "12. Financial year start values are correct"
)

# --------------------------------------------------
# C. Geographic validation (13–17)
# --------------------------------------------------

print("\nC. GEOGRAPHIC VALIDATION")

check(
    set(map_df["state_code"]) == set(STATES),
    "13. Only the four reported states are mapped"
)

check(
    all(
        row.state == STATES[row.state_code][0]
        for row in map_df.itertuples()
    ),
    "14. State names match state codes"
)

check(
    all(
        math.isclose(
            row.longitude,
            STATES[row.state_code][1],
            abs_tol=0.000001
        )
        for row in map_df.itertuples()
    ),
    "15. Longitude coordinates are correct"
)

check(
    all(
        math.isclose(
            row.latitude,
            STATES[row.state_code][2],
            abs_tol=0.000001
        )
        for row in map_df.itertuples()
    ),
    "16. Latitude coordinates are correct"
)

check(
    map_df.groupby("year")["state_code"].apply(
        lambda x: set(x) == set(STATES)
    ).all(),
    "17. Every year has the correct state coverage"
)

# --------------------------------------------------
# D. Numeric validation (18–21)
# --------------------------------------------------

print("\nD. NUMERIC VALIDATION")

check(
    (map_df["spend_m"] >= 0).all(),
    "18. No negative expenditure values"
)

check(
    (map_df["total_m"] > 0).all()
    and (summary["total_m"] > 0).all(),
    "19. Published totals are positive"
)

expected_shares = (
    map_df["spend_m"]
    / map_df["total_m"] * 100
).round(2)

check(
    (
        abs(map_df["share_pct"] - expected_shares)
        < 0.001
    ).all(),
    "20. State percentage shares are calculated correctly"
)

check(
    map_df.groupby("year")["total_m"].nunique().eq(1).all()
    and
    map_df.groupby("year")["total_m"].first().sort_index().equals(
        summary.set_index("year")["total_m"].sort_index()
    ),
    "21. Map and summary totals agree"
)

# --------------------------------------------------
# E. Source consistency (22–24)
# --------------------------------------------------

print("\nE. SOURCE CONSISTENCY")

# Check every state expenditure against the original CSV.

raw_long = raw.melt(
    id_vars=["year"],
    value_vars=list(STATES),
    var_name="state_code",
    value_name="source_spend"
)

raw_long["source_spend"] = pd.to_numeric(
    raw_long["source_spend"],
    errors="raise"
)

comparison = map_df.merge(
    raw_long,
    on=["year", "state_code"],
    how="outer",
    indicator=True
)

check(
    len(comparison) == 120
    and (comparison["_merge"] == "both").all()
    and (
        comparison["spend_m"]
        == comparison["source_spend"]
    ).all(),
    "22. All mapped expenditure matches original source"
)

raw_totals = raw[["year", "Total"]].copy()
raw_totals["Total"] = pd.to_numeric(
    raw_totals["Total"],
    errors="raise"
)

total_comparison = summary.merge(
    raw_totals,
    on="year",
    how="outer",
    indicator=True
)

check(
    len(total_comparison) == 30
    and (total_comparison["_merge"] == "both").all()
    and (
        total_comparison["total_m"]
        == total_comparison["Total"]
    ).all(),
    "23. Published national totals are preserved"
)

other_comparison = summary.merge(
    raw[["year", "Other*"]],
    on="year",
    how="outer",
    indicator=True
)

other_valid = (
    len(other_comparison) == 30
    and (other_comparison["_merge"] == "both").all()
    and (
        other_comparison["other_reported"]
        == other_comparison["Other*"]
    ).all()
)

if other_valid:
    # Use column access to preserve the literal Other* name.
    for _, row in other_comparison.iterrows():
        if row["Other*"] == "<1":
            if (
                not bool(row["other_is_less_than_one"])
                or not pd.isna(row["other_m"])
            ):
                other_valid = False
                break
        else:
            if (
                bool(row["other_is_less_than_one"])
                or pd.isna(row["other_m"])
                or not math.isclose(
                    float(row["other_m"]),
                    float(row["Other*"]),
                    abs_tol=0.000001
                )
            ):
                other_valid = False
                break

check(
    other_valid,
    "24. Other* values and special reporting flags are preserved"
)

# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 50)
print("CHART 11 VALIDATION RESULTS")
print("=" * 50)

print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Total:  {passed + failed}")

if failed:
    raise SystemExit(
        "Chart 11 validation failed. Review the checks above."
    )

print("\nAll Chart 11 validation checks passed!")
