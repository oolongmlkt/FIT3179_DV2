"""
scripts/preprocess_tmdb.py

FIT3179 DV2 — Stage 1 preprocessing.

Run from the project root:
    python3 scripts/preprocess_tmdb.py

Requires pandas. Reads data/raw/ and recreates data/processed/ outputs.
Does not modify raw data. Compatible with Python 3.9.
"""
from pathlib import Path
import pandas as pd

RAW = Path('data/raw')
OUT = Path('data/processed')
CHUNK_SIZE = 50_000
START_YEAR, END_YEAR = 1980, 2025

FIELDS = [
    'id', 'title', 'status', 'release_date', 'genres',
    'production_countries', 'production_companies', 'director', 'cast',
    'budget', 'revenue', 'vote_average', 'vote_count', 'imdb_rating',
    'imdb_votes', 'imdb_id', 'runtime', 'original_language'
]

TABLES = {
    'movies.csv': ['id', 'title', 'year', 'decade', 'release_date',
                   'original_language', 'runtime', 'budget', 'revenue',
                   'vote_average', 'vote_count', 'imdb_rating', 'imdb_votes',
                   'imdb_id'],
    'movie_genres.csv': ['id', 'year', 'decade', 'genre'],
    'movie_countries.csv': ['id', 'year', 'decade', 'country'],
    'movie_studios.csv': ['id', 'year', 'decade', 'studio'],
    'movie_directors.csv': ['id', 'year', 'decade', 'director'],
    'movie_cast.csv': ['id', 'year', 'decade', 'actor'],
}

RELATIONSHIPS = {
    'movie_genres.csv': ('genres', 'genre'),
    'movie_countries.csv': ('production_countries', 'country'),
    'movie_studios.csv': ('production_companies', 'studio'),
    'movie_directors.csv': ('director', 'director'),
    'movie_cast.csv': ('cast', 'actor'),
}


def split_tokens(value):
    """Split the source's comma-delimited labels, preserving unique labels.

    Caveat: source has no escaping for commas inside personal/company names.
    Review ambiguous labels before final network/studio visualisations.
    """
    if pd.isna(value) or not str(value).strip():
        return []
    return list(dict.fromkeys(
        token.strip() for token in str(value).split(',') if token.strip()
    ))



def explode_field(chunk, source, label):
    """One row per movie–label pair; works when source == label."""
    frame = chunk[['id', 'year', 'decade', source]].copy()

    # Split comma-separated values into individual labels.
    frame['_values'] = frame[source].map(split_tokens)

    # Create one row per movie–label relationship.
    frame = frame[['id', 'year', 'decade', '_values']].explode('_values')
    frame = frame.dropna(subset=['_values'])

    # Rename temporary column.
    frame = frame.rename(columns={'_values': label})

    # Remove whitespace and invalid placeholder values.
    frame[label] = frame[label].astype(str).str.strip()

    invalid_labels = {'', 'nan', 'none', 'null', '<na>'}

    frame = frame[
        ~frame[label].str.lower().isin(invalid_labels)
    ]

    # Remove duplicate movie–label relationships.
    frame = frame.drop_duplicates(subset=['id', label])

    return frame[['id', 'year', 'decade', label]]



def process_tmdb():
    path = RAW / 'tmdb_movies.csv'
    if not path.is_file():
        raise FileNotFoundError('Missing {}'.format(path))

    # Remove old outputs, including any incomplete outputs from failed runs.
    for filename in TABLES:
        (OUT / filename).unlink(missing_ok=True)

    first = {filename: True for filename in TABLES}
    counts = {filename: 0 for filename in TABLES}
    seen_ids = set()
    read_rows = 0
    duplicate_ids = 0
    both_financial = 0
    australian_movies = 0

    reader = pd.read_csv(
        path, usecols=lambda col: col in FIELDS, chunksize=CHUNK_SIZE,
        low_memory=False, encoding='utf-8-sig'
    )
    for number, chunk in enumerate(reader, start=1):
        read_rows += len(chunk)
        missing = set(FIELDS).difference(chunk.columns)
        if missing:
            raise ValueError('Missing TMDB columns: {}'.format(sorted(missing)))

        chunk['release_date'] = pd.to_datetime(chunk['release_date'], errors='coerce')
        chunk['year'] = chunk['release_date'].dt.year
        chunk = chunk.loc[
            chunk['status'].eq('Released') &
            chunk['year'].between(START_YEAR, END_YEAR)
        ].copy()
        chunk['id'] = pd.to_numeric(chunk['id'], errors='coerce')
        chunk = chunk.dropna(subset=['id'])
        chunk = chunk.drop_duplicates(subset=['id'])
        is_new = ~chunk['id'].isin(seen_ids)
        duplicate_ids += int((~is_new).sum())
        chunk = chunk.loc[is_new].copy()
        seen_ids.update(chunk['id'].tolist())
        if chunk.empty:
            print('Chunk {}: no eligible rows'.format(number), flush=True)
            continue

        chunk['id'] = chunk['id'].astype('int64')
        chunk['year'] = chunk['year'].astype('int64')
        chunk['decade'] = (chunk['year'] // 10) * 10

        numeric = ['runtime', 'budget', 'revenue', 'vote_average',
                   'vote_count', 'imdb_rating', 'imdb_votes']
        for field in numeric:
            chunk[field] = pd.to_numeric(chunk[field], errors='coerce')
        for field in ['budget', 'revenue']:
            chunk.loc[chunk[field] <= 0, field] = float('nan')

        both_financial += int((chunk['budget'].notna() &
                               chunk['revenue'].notna()).sum())
        australian_movies += int(chunk['production_countries'].map(
            lambda x: 'Australia' in split_tokens(x)).sum())

        outputs = {'movies.csv': chunk[TABLES['movies.csv']]}
        for filename, (source, label) in RELATIONSHIPS.items():
            outputs[filename] = explode_field(chunk, source, label)

        for filename, frame in outputs.items():
            # Explicit columns even for an empty chunk.
            frame = frame.reindex(columns=TABLES[filename])
            frame.to_csv(OUT / filename, mode='w' if first[filename] else 'a',
                         header=first[filename], index=False, encoding='utf-8')
            first[filename] = False
            counts[filename] += len(frame)

        print('Chunk {}: {:,} source rows read; {:,} movies retained'.format(
            number, read_rows, counts['movies.csv']), flush=True)

    # Produce header-only files if no movie met the filters.
    for filename, columns in TABLES.items():
        if first[filename]:
            pd.DataFrame(columns=columns).to_csv(OUT / filename, index=False)

    print('\nTMDB output row counts:')
    for filename, count in counts.items():
        print('  {}: {:,}'.format(filename, count))
    print('  Duplicate IDs excluded across chunks: {:,}'.format(duplicate_ids))
    print('  Movies with both positive budget and revenue: {:,}'.format(both_financial))
    print('  Movies credited to Australia: {:,}'.format(australian_movies))

    # Fail visibly if source data has values but relationship outputs are empty.
    for filename in RELATIONSHIPS:
        if counts['movies.csv'] and counts[filename] == 0:
            raise RuntimeError('{} has zero rows; inspect raw columns'.format(filename))


def parse_spend(value):
    """Suppressed values such as '<1' remain missing, never become zero."""
    if pd.isna(value):
        return float('nan')
    value = str(value).strip().replace('$', '')
    if not value or value.lower() in {'nan', 'n/a', 'na', '-', '—'}:
        return float('nan')
    if value.startswith('<'):
        return float('nan')
    return float(value.replace(',', ''))


def process_screen_australia():
    spend_path = RAW / 'screen_australia_spend_by_location.csv'
    genre_path = RAW / 'screen_australia_genres.csv'
    for path in [spend_path, genre_path]:
        if not path.is_file():
            raise FileNotFoundError('Missing {}'.format(path))

    spend = pd.read_csv(spend_path, encoding='utf-8-sig', dtype=str)
    spend.columns = [str(c).strip() for c in spend.columns]
    spend = spend.rename(columns={spend.columns[0]: 'financial_year'})
    spend['financial_year'] = spend['financial_year'].str.strip()
    required = {'NSW', 'QLD', 'VIC', 'SA', 'Other*', 'Total'}
    if not required.issubset(spend.columns):
        raise ValueError('Missing Screen Australia spending columns: {}'.format(
            sorted(required - set(spend.columns))))

    states = {'NSW': 'New South Wales', 'QLD': 'Queensland',
              'VIC': 'Victoria', 'SA': 'South Australia'}
    rows = []
    for _, record in spend.iterrows():
        for code, name in states.items():
            rows.append({'financial_year': record['financial_year'],
                         'state_code': code, 'state': name,
                         'spend_m_aud': parse_spend(record[code])})
    pd.DataFrame(rows, columns=['financial_year', 'state_code', 'state',
                                'spend_m_aud']).to_csv(
        OUT / 'australian_state_spending.csv', index=False)

    totals = spend[['financial_year', 'Other*', 'Total']].rename(columns={
        'Other*': 'other_m_aud', 'Total': 'total_m_aud'
    }).copy()
    for col in ['other_m_aud', 'total_m_aud']:
        totals[col] = totals[col].map(parse_spend)
    totals.to_csv(OUT / 'australian_spending_totals.csv', index=False)

    genres = pd.read_csv(genre_path, encoding='utf-8-sig', dtype=str)
    genres.columns = [str(c).strip() for c in genres.columns]
    genres = genres.rename(columns={genres.columns[0]: 'genre'})
    long = genres.melt(id_vars='genre', var_name='cohort', value_name='share_pct')
    long['share_pct'] = pd.to_numeric(
        long['share_pct'].str.strip().str.rstrip('%'), errors='coerce')
    long.to_csv(OUT / 'australian_genre_shares.csv', index=False)
    print('\nScreen Australia outputs written (Other is not assigned to a state).')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    process_tmdb()
    process_screen_australia()
    print('\nFinished: {}'.format(OUT.resolve()))


if __name__ == '__main__':
    main()
