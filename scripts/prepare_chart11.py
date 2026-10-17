# https://github.com/rowanhogan/australian-states/blob/master/states.geojson for the geojson of Australia states

from pathlib import Path
import pandas as pd

# --------------------------------------------------
# File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR / "data" / "raw"
    / "screen_australia_spend_by_location.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_OUTPUT = OUTPUT_DIR / "australia_spend.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "australia_spend_summary.csv"


# --------------------------------------------------
# 1. Load original dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE, dtype=str)

df.columns = df.columns.str.strip()

# The first column contains financial years.
df = df.rename(columns={df.columns[0]: "year"})

# Remove trailing spaces from all string values.
for column in df.columns:
    df[column] = df[column].str.strip()

required = [
    "year", "NSW", "QLD", "VIC",
    "SA", "Other*", "Total"
]

missing = set(required) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df[required].isna().any().any():
    raise ValueError("Unexpected missing source values found.")


# --------------------------------------------------
# 2. Parse expenditure values
# --------------------------------------------------

state_columns = ["NSW", "QLD", "VIC", "SA"]

for column in state_columns + ["Total"]:
    df[column] = pd.to_numeric(
        df[column],
        errors="raise"
    )

# Preserve special reporting values such as <1.
df["other_reported"] = df["Other*"]

# <1 is not an exact numeric value.
# Do not silently replace it with 0 or 1.
df["other_m"] = pd.to_numeric(
    df["Other*"].where(df["Other*"] != "<1"),
    errors="coerce"
)

df["other_is_less_than_one"] = (
    df["Other*"] == "<1"
)

df["total_m"] = df["Total"]

# Chronological sorting.
df["year_start"] = df["year"].str[:4].astype(int)

df = df.sort_values("year_start").reset_index(drop=True)


# --------------------------------------------------
# 3. Create geographic records
# --------------------------------------------------

# Representative coordinates located within each state.
# These are symbol positions, not production locations.

locations = {
    "NSW": {
        "state": "New South Wales",
        "longitude": 147.0,
        "latitude": -32.0
    },
    "QLD": {
        "state": "Queensland",
        "longitude": 145.0,
        "latitude": -22.5
    },
    "VIC": {
        "state": "Victoria",
        "longitude": 144.0,
        "latitude": -37.0
    },
    "SA": {
        "state": "South Australia",
        "longitude": 135.5,
        "latitude": -30.0
    }
}

map_df = df.melt(
    id_vars=["year", "year_start", "total_m"],
    value_vars=state_columns,
    var_name="state_code",
    value_name="spend_m"
)

map_df["state"] = map_df["state_code"].map(
    lambda code: locations[code]["state"]
)

map_df["longitude"] = map_df["state_code"].map(
    lambda code: locations[code]["longitude"]
)

map_df["latitude"] = map_df["state_code"].map(
    lambda code: locations[code]["latitude"]
)

map_df["share_pct"] = (
    map_df["spend_m"] / map_df["total_m"] * 100
).round(2)

map_df = map_df[
    [
        "year",
        "year_start",
        "state",
        "state_code",
        "spend_m",
        "longitude",
        "latitude",
        "total_m",
        "share_pct"
    ]
]

map_df = map_df.sort_values(
    ["year_start", "state_code"]
).reset_index(drop=True)


# --------------------------------------------------
# 4. Create annual summary
# --------------------------------------------------

summary_df = df[
    [
        "year",
        "year_start",
        "other_reported",
        "other_m",
        "other_is_less_than_one",
        "total_m"
    ]
].copy()

summary_df["mapped_states_m"] = df[
    state_columns
].sum(axis=1)

summary_df["reported_difference_m"] = (
    summary_df["total_m"]
    - summary_df["mapped_states_m"]
    - summary_df["other_m"]
)

# Differences are retained for auditing.
# Do not force rounded figures to sum exactly.


# --------------------------------------------------
# 5. Export
# --------------------------------------------------

map_df.to_csv(MAP_OUTPUT, index=False)

summary_df.to_csv(SUMMARY_OUTPUT, index=False)

print("Chart 11 preprocessing complete.")
print(f"Financial years: {len(df)}")
print(f"Map records: {len(map_df)}")
print(f"Summary records: {len(summary_df)}")

print("\nMap preview:")
print(map_df.head(8).to_string(index=False))

print("\nSummary preview:")
print(summary_df.tail(5).to_string(index=False))

print("\nOutput files:")
print(MAP_OUTPUT)
print(SUMMARY_OUTPUT)
