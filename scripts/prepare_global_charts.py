"""
scripts/prepare_global_charts.py

FIT3179 DV2 — build datasets for Charts 1–3.
Run from repository root: python scripts/prepare_global_charts.py
Requires pandas. Does not modify existing processed data.

Chart 1: Average annual TMDB-listed productions per million residents.
         Only country-years with valid World Bank population contribute.
Chart 2: Full genre counting, by year.
Chart 3: Each film contributes once to each unordered distinct ISO3 country pair.
         Only pairs with two known coordinate endpoints can be mapped.
"""
from pathlib import Path
from itertools import combinations
from collections import Counter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'data' / 'processed'
OUT = ROOT / 'data' / 'charts'
START, END = 1980, 2025


def read(name, **kwargs):
    path = P / name
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}; run preprocessing first.')
    return pd.read_csv(path, keep_default_na=False, **kwargs)


def require(df, fields, name):
    missing = set(fields) - set(df.columns)
    if missing:
        raise ValueError(f'{name} missing columns: {sorted(missing)}')


def numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def integer_iso_numeric(series):
    """Preserve missing ISO numeric codes without writing valid codes as 36.0."""
    values = pd.to_numeric(series, errors='coerce')
    nonmissing = values.dropna()
    if not nonmissing.eq(nonmissing.round()).all():
        raise ValueError('ISO numeric country identifiers must be whole numbers')
    return values.astype('Int64')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    countries = read('movie_countries.csv')
    genres = read('movie_genres.csv')
    lookup = read('country_lookup.csv')
    pop = read('population_by_year.csv')
    coords = read('country_coordinates.csv')
    require(countries, ['id', 'year', 'country'], 'movie_countries.csv')
    require(genres, ['id', 'year', 'genre'], 'movie_genres.csv')
    require(lookup, ['tmdb_country', 'iso3', 'iso_numeric'], 'country_lookup.csv')
    require(pop, ['iso3', 'year', 'population'], 'population_by_year.csv')
    require(coords, ['iso3', 'latitude', 'longitude'], 'country_coordinates.csv')
    numeric(countries, ['id', 'year'])
    numeric(genres, ['id', 'year'])
    numeric(pop, ['year', 'population'])
    numeric(coords, ['latitude', 'longitude'])
    lookup['iso_numeric'] = integer_iso_numeric(lookup['iso_numeric'])
    countries = countries[countries.year.between(START, END)].copy()
    genres = genres[genres.year.between(START, END)].copy()
    if countries[['id', 'year']].isna().any().any() or genres[['id', 'year']].isna().any().any():
        raise ValueError('Invalid movie IDs or years in relationship data')
    if lookup.tmdb_country.duplicated().any():
        raise ValueError('Duplicate TMDB labels in country lookup')
    if pop.duplicated(['iso3', 'year']).any() or coords.iso3.duplicated().any():
        raise ValueError('Duplicate ISO3-year population or ISO3 coordinate records')

    # Retain original labels for coverage accounting; no silent successor-state assignment.
    country = countries.merge(lookup[['tmdb_country', 'iso3', 'iso_numeric']],
                              left_on='country', right_on='tmdb_country', how='left', validate='many_to_one')
    country['iso3'] = country.iso3.fillna('').astype(str).str.strip()
    unresolved = country[country.iso3.eq('')]
    matched = country[country.iso3.ne('')].copy()
    # Distinct country credit per movie, even when two source labels map to one ISO3.
    matched = matched.drop_duplicates(['id', 'iso3'])
    if matched.groupby('id').year.nunique().gt(1).any():
        raise ValueError('A movie has conflicting release years')
    matched['year'] = matched.year.astype(int)
    country_counts = matched.groupby(['iso3', 'year']).id.nunique().rename('movie_count').reset_index()

    # Chart 1: keep missing population out of denominator and annual average.
    cp = country_counts.merge(pop[['iso3', 'year', 'population']], on=['iso3', 'year'],
                              how='left', validate='one_to_one')
    cp = cp.merge(lookup[lookup.iso3.ne('')][['iso3', 'iso_numeric']].drop_duplicates('iso3'),
                  on='iso3', how='left', validate='many_to_one')
    cp['iso_numeric'] = integer_iso_numeric(cp['iso_numeric'])
    cp['valid_population'] = cp.population.gt(0)
    cp['movies_per_million'] = cp.movie_count.div(cp.population).mul(1_000_000).where(cp.valid_population)
    cp.sort_values(['iso3', 'year']).to_csv(OUT / 'country_production_by_year.csv', index=False)
    # Include zero-movie years for each matched country where World Bank population is valid.
    # This avoids inflating the average by averaging only years with at least one movie.
    iso = matched[['iso3']].drop_duplicates()
    grid = iso.merge(pop[['iso3', 'year', 'population']], on='iso3', how='left')
    grid = grid[grid.year.between(START, END) & grid.population.gt(0)].copy()
    grid = grid.merge(country_counts, on=['iso3', 'year'], how='left', validate='one_to_one')
    grid['movie_count'] = grid.movie_count.fillna(0).astype(int)
    grid['movies_per_million'] = grid.movie_count / grid.population * 1_000_000
    aggregate = grid.groupby('iso3', as_index=False).agg(
        average_annual_movies_per_million=('movies_per_million', 'mean'),
        valid_population_years=('year', 'nunique'),
        movies_in_valid_years=('movie_count', 'sum'),
        first_valid_year=('year', 'min'), last_valid_year=('year', 'max'))
    aggregate = aggregate.merge(lookup[lookup.iso3.ne('')][['iso3', 'iso_numeric', 'standard_name']]
                                .drop_duplicates('iso3'), on='iso3', how='left', validate='one_to_one')
    aggregate['iso_numeric'] = integer_iso_numeric(aggregate['iso_numeric'])
    aggregate.sort_values('iso3').to_csv(OUT / 'country_production.csv', index=False)

    # Chart 2: count a movie once for each of its genres.
    genre = genres.copy()
    genre['genre'] = genre.genre.astype(str).str.strip()
    genre = genre[genre.genre.ne('')].drop_duplicates(['id', 'genre'])
    genre.groupby(['year', 'genre']).id.nunique().rename('movie_count').reset_index().sort_values(
        ['year', 'genre']).to_csv(OUT / 'genre_by_year.csv', index=False)

    # Chart 3: country pairs are undirected, so A–B and B–A are one relationship.
    coord = coords[['iso3', 'latitude', 'longitude']].copy()
    valid_coords = set(coord.loc[coord.latitude.between(-90, 90) &
                                 coord.longitude.between(-180, 180), 'iso3'])
    pairs = Counter()
    excluded_pairs = Counter()
    international_movies = 0
    for _, group in matched.groupby('id', sort=False):
        codes = sorted(set(group.iso3))
        if len(codes) < 2:
            continue
        international_movies += 1
        for a, b in combinations(codes, 2):
            if a in valid_coords and b in valid_coords:
                pairs[(a, b)] += 1
            else:
                excluded_pairs[(a, b)] += 1
    
    # Build Chart 3 country-pair dataset.
    # Each country pair remains undirected and counted once per film.

    rows = [
        {
            'source_iso3': a,
            'target_iso3': b,
            'movie_count': n
        }
        for (a, b), n in sorted(pairs.items())
    ]

    flow = pd.DataFrame(
        rows,
        columns=['source_iso3', 'target_iso3', 'movie_count']
    )

    # Add readable country names for Chart 3 tooltips.
    # Reuse the existing country lookup used by Chart 1.

    country_names = (
        lookup[lookup.iso3.ne('')]
        [['iso3', 'standard_name']]
        .drop_duplicates('iso3')
    )

    flow = flow.merge(
        country_names.rename(columns={
            'iso3': 'source_iso3',
            'standard_name': 'source_name'
        }),
        on='source_iso3',
        how='left',
        validate='many_to_one'
    )

    flow = flow.merge(
        country_names.rename(columns={
            'iso3': 'target_iso3',
            'standard_name': 'target_name'
        }),
        on='target_iso3',
        how='left',
        validate='many_to_one'
    )

    # Use ISO3 codes only if a readable name is unavailable.

    flow['source_name'] = flow['source_name'].fillna(flow['source_iso3'])
    flow['target_name'] = flow['target_name'].fillna(flow['target_iso3'])

    flow = flow.merge(coord.rename(columns={'iso3': 'source_iso3', 'latitude': 'source_lat',
                                            'longitude': 'source_lon'}), on='source_iso3', how='left')
    flow = flow.merge(coord.rename(columns={'iso3': 'target_iso3', 'latitude': 'target_lat',
                                            'longitude': 'target_lon'}), on='target_iso3', how='left')
    flow.sort_values(['movie_count', 'source_iso3', 'target_iso3'], ascending=[False, True, True]).to_csv(
        OUT / 'country_coproductions.csv', index=False)

    print('FIT3179 DV2 — GLOBAL CHART PREPARATION')
    print(f'Country credits (input): {len(countries):,}; unresolved credits: {len(unresolved):,}')
    print(f'Distinct matched movie-country credits: {len(matched):,}')
    print(f'Chart 1: {len(aggregate):,} countries with usable population; '
          f'{len(grid):,} valid country-years; {len(cp) - int(cp.valid_population.sum()):,} '
          'positive-production country-years missing population')
    print(f'Chart 2: {genre.id.nunique():,} movies with genres; '
          f'{len(genre):,} movie-genre credits')
    print(f'Chart 3: {international_movies:,} movies with 2+ matched countries; '
          f'{len(flow):,} mapped country pairs; {sum(pairs.values()):,} mapped pair credits; '
          f'{sum(excluded_pairs.values()):,} pair credits excluded for missing coordinates')
    print(f'Outputs saved to: {OUT}')
    print('Note: unresolved country credits and unmatched country pairs are not reassigned.')


if __name__ == '__main__':
    main()
