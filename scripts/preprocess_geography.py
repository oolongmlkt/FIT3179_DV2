"""FIT3179 DV2 — prepare geographic reference data for Charts 1 and 3.

Run from project root:
    pip install pandas pycountry countryinfo
    python scripts/preprocess_geography.py

Inputs:
    data/raw/world_bank_population.csv
    data/processed/movie_countries.csv
Outputs:
    data/processed/population_by_year.csv
    data/processed/country_lookup.csv
    data/processed/country_coordinates.csv
    data/processed/geography_unmatched_countries.csv
    data/processed/geography_missing_population.csv
    data/processed/geography_missing_coordinates.csv

The country coordinates are representative reference points from countryinfo,
not geometric centroids or guaranteed points within each country's polygon.
Review placements on the rendered Vega-Lite flow map. No Natural Earth download needed.
"""
from __future__ import annotations

from pathlib import Path
import re
import unicodedata
import pandas as pd
import pycountry
from countryinfo import CountryInfo

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
START_YEAR, END_YEAR = 1980, 2025

# Exact TMDB label overrides. Historical/politically ambiguous entities are
# deliberately NOT silently assigned to a modern successor state.
ALIASES = {
    'United States of America': 'USA', 'United States': 'USA',
    'United Kingdom': 'GBR', 'South Korea': 'KOR', 'North Korea': 'PRK',
    'Russia': 'RUS', 'Russian Federation': 'RUS', 'Czech Republic': 'CZE',
    'Czechia': 'CZE', 'Vietnam': 'VNM', 'Viet Nam': 'VNM',
    'Iran': 'IRN', 'Iran, Islamic Republic of': 'IRN',
    'Syria': 'SYR', 'Syrian Arab Republic': 'SYR',
    'Taiwan': 'TWN', 'Taiwan, Province of China': 'TWN',
    'Hong Kong': 'HKG', 'Macao': 'MAC', 'Macau': 'MAC',
    'Palestine': 'PSE', 'Palestinian Territory': 'PSE',
    'Venezuela': 'VEN', 'Bolivia': 'BOL', 'Tanzania': 'TZA',
    'Moldova': 'MDA', 'Laos': 'LAO', 'Brunei': 'BRN',
    'Ivory Coast': 'CIV', "Côte d'Ivoire": 'CIV',
    'Democratic Republic of the Congo': 'COD', 'DR Congo': 'COD',
    'Republic of the Congo': 'COG', 'Congo': 'COG',
    'The Netherlands': 'NLD', 'The Bahamas': 'BHS', 'Bahamas': 'BHS',
    'The Gambia': 'GMB', 'Gambia': 'GMB',
    'Cape Verde': 'CPV', 'Swaziland': 'SWZ', 'East Timor': 'TLS',
    'Cocos  Islands': 'CCK', "Cote D'Ivoire": 'CIV',
    'Faeroe Islands': 'FRO', 'Falkland Islands': 'FLK',
    'Guadaloupe': 'GLP', 'Heard and McDonald Islands': 'HMD',
    'Holy See': 'VAT', 'Libyan Arab Jamahiriya': 'LBY',
    'Macedonia': 'MKD', 'Micronesia': 'FSM',
    'Pitcairn Island': 'PCN', 'Reunion': 'REU',
    'St. Helena': 'SHN', 'St. Kitts and Nevis': 'KNA',
    'St. Lucia': 'LCA', 'St. Pierre and Miquelon': 'SPM',
    'St. Vincent and the Grenadines': 'VCT',
    'Svalbard & Jan Mayen Islands': 'SJM', 'Turkey': 'TUR',
    'US Virgin Islands': 'VIR', 'Wallis and Futuna Islands': 'WLF',
    'Zaire': 'COD', 'Northern Ireland': 'GBR',
    'Kosovo': None, 'Soviet Union': None, 'Yugoslavia': None,
    'Czechoslovakia': None, 'East Germany': None, 'West Germany': None,
    'Serbia and Montenegro': None, 'Netherlands Antilles': None,
}

# Approximate representative points (latitude, longitude) for CountryInfo gaps.
# These are for flow-map endpoints, NOT country centroids or polygon geometry.
# Remote/uninhabited territories without a defensible single point remain blank.
COORDINATE_FALLBACKS = {
    'AND': (42.55, 1.58), 'BHS': (24.25, -76.00),
    'BOL': (-17.00, -65.00), 'BRN': (4.54, 114.73),
    'COG': (-1.00, 15.00), 'CPV': (16.00, -24.00),
    'CZE': (49.75, 15.50), 'GMB': (13.45, -15.40),
    'IRN': (32.00, 53.00), 'KOR': (36.50, 127.75),
    'MAC': (22.20, 113.55), 'MDA': (47.00, 28.50),
    'MMR': (21.00, 96.00), 'MNE': (42.75, 19.25),
    'PRK': (40.00, 127.00), 'PSE': (31.90, 35.20),
    'STP': (0.20, 6.60), 'SWZ': (-26.50, 31.50),
    'TCA': (21.75, -71.58), 'TLS': (-8.80, 125.75),
    'TWN': (23.70, 121.00), 'TZA': (-6.00, 35.00),
    'VEN': (8.00, -66.00), 'VGB': (18.42, -64.60),
    'VNM': (16.00, 107.50),
}

# Unresolved historical entities are intentionally not mapped to successor states.
# Northern Ireland is rolled into GBR; Zaire into COD as a historical name
# for the same geographic country. This is documented in match_status.
HISTORICAL_ALIASES = {'Zaire', 'Northern Ireland'}

# ISO numeric values must be kept as integers for Vega world-110m `id` lookup.
# A few entities have ISO3 codes but may not be present in the simplified map.
def iso_record(label: str):
    label = str(label).strip()
    if not label or label.casefold() in {'nan', 'none', 'unknown', 'n/a'}:
        return None
    if label in ALIASES:
        code = ALIASES[label]
        return pycountry.countries.get(alpha_3=code) if code else None
    try:
        return pycountry.countries.lookup(label)
    except LookupError:
        return None


def clean_population():
    path = RAW / 'world_bank_population.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing World Bank input: {path}')
    # World Bank download has two introductory metadata rows, then the header.
    df = pd.read_csv(path, skiprows=4, encoding='utf-8-sig', low_memory=False)
    required = {'Country Name', 'Country Code', 'Indicator Code'}
    if not required.issubset(df.columns):
        raise ValueError(f'Unexpected World Bank format; expected {required}, got {list(df.columns)[:8]}')
    df = df[df['Indicator Code'].eq('SP.POP.TOTL')].copy()
    years = [str(y) for y in range(START_YEAR, END_YEAR + 1)]
    missing = set(years) - set(df.columns)
    if missing:
        raise ValueError(f'Missing World Bank year columns: {sorted(missing)}')
    # pycountry membership excludes World Bank aggregates (WLD, regions, income groups).
    df['iso3'] = df['Country Code'].map(lambda c: c if pycountry.countries.get(alpha_3=str(c)) else None)
    df = df.dropna(subset=['iso3'])
    long = df.melt(id_vars=['iso3', 'Country Name'], value_vars=years,
                   var_name='year', value_name='population')
    long = long.rename(columns={'Country Name': 'world_bank_name'})
    long['year'] = long['year'].astype(int)
    long['population'] = pd.to_numeric(long['population'], errors='coerce')
    long.loc[long['population'] <= 0, 'population'] = float('nan')
    if long.duplicated(['iso3', 'year']).any():
        raise ValueError('Duplicate ISO3/year combinations in World Bank population file')
    long = long.sort_values(['iso3', 'year'])
    long.to_csv(PROCESSED / 'population_by_year.csv', index=False)
    return long


def tmdb_country_names():
    path = PROCESSED / 'movie_countries.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing processed input: {path}. Run TMDB preprocessing first.')
    names = set()
    for chunk in pd.read_csv(path, usecols=['country'], chunksize=100_000, keep_default_na=False):
        names.update(s.strip() for s in chunk['country'].astype(str) if s.strip())
    return sorted(names)


def make_lookup(names):
    rows = []
    for name in names:
        obj = iso_record(name)
        rows.append({'tmdb_country': name,
                     'iso3': obj.alpha_3 if obj else '',
                     'iso_numeric': int(obj.numeric) if obj else pd.NA,
                     'standard_name': obj.name if obj else '',
                     'match_status': ('matched_historical_or_subnational' if name in HISTORICAL_ALIASES else 'matched') if obj else 'REVIEW_UNMATCHED'})
    lookup = pd.DataFrame(rows).astype({'iso_numeric': 'Int64'})
    lookup.to_csv(PROCESSED / 'country_lookup.csv', index=False)
    unmatched = lookup[lookup['match_status'].eq('REVIEW_UNMATCHED')]
    unmatched.to_csv(PROCESSED / 'geography_unmatched_countries.csv', index=False)
    return lookup, unmatched


def make_coordinates(lookup):
    # CountryInfo's latlng is a country reference coordinate; unlike the World
    # Bank API's lat/long, it is not necessarily the capital's location.
    rows = []
    for iso3 in sorted(lookup['iso3'].dropna().loc[lambda s: s.ne('')].unique()):
        obj = pycountry.countries.get(alpha_3=iso3)
        lat, lon = None, None
        try:
            value = CountryInfo(obj.name).info().get('latlng', [])
            if len(value) == 2:
                lat, lon = float(value[0]), float(value[1])
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    lat, lon = None, None
        except (KeyError, ValueError, TypeError, AttributeError):
            pass
        source = 'countryinfo' if lat is not None else ''
        if lat is None and iso3 in COORDINATE_FALLBACKS:
            lat, lon = COORDINATE_FALLBACKS[iso3]
            source = 'curated_approximate'
        rows.append({'iso3': iso3, 'iso_numeric': int(obj.numeric),
                     'country': obj.name, 'latitude': lat, 'longitude': lon,
                     'coordinate_source': source})
    coords = pd.DataFrame(rows)
    coords.to_csv(PROCESSED / 'country_coordinates.csv', index=False)
    missing = coords[coords['latitude'].isna() | coords['longitude'].isna()]
    missing.to_csv(PROCESSED / 'geography_missing_coordinates.csv', index=False)
    return coords, missing


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    population = clean_population()
    names = tmdb_country_names()
    lookup, unmatched = make_lookup(names)
    coords, missing_coords = make_coordinates(lookup)
    used_iso = set(lookup['iso3'].dropna()) - {''}
    population_iso = set(population.loc[population.population.notna(), 'iso3'])
    missing_pop = lookup[lookup['iso3'].isin(used_iso - population_iso)].copy()
    missing_pop.to_csv(PROCESSED / 'geography_missing_population.csv', index=False)
    print(f'World Bank country-year records: {len(population):,}')
    print(f'TMDB country labels: {len(lookup):,}; unmatched: {len(unmatched):,}')
    print(f'Country coordinate records: {len(coords):,}; missing: {len(missing_coords):,}')
    print(f'Matched TMDB countries without any usable population: {len(missing_pop):,}')
    print('Review all geography_* report files before building Charts 1 and 3.')


if __name__ == '__main__':
    main()
