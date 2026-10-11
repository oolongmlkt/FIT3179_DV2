"""
scripts/validate_processed.py

FIT3179 DV2: validate Stage 1 processed CSV outputs.

Run from the project root: python3 scripts/validate_processed.py

Requires pandas. Read-only: never changes processed data.

"""

from pathlib import Path

import re

import sys

import pandas as pd



ROOT = Path('data/processed')

CHUNK = 100_000

START_YEAR, END_YEAR = 1980, 2025

RELATIONS = {

    'movie_genres.csv': 'genre',

    'movie_countries.csv': 'country',

    'movie_studios.csv': 'studio',

    'movie_directors.csv': 'director',

    'movie_cast.csv': 'actor',

}

issues = []

warnings = []





def check(condition, message):

    print(('PASS' if condition else 'FAIL') + ': ' + message)

    if not condition:

        issues.append(message)





def warn(message):

    print('WARN: ' + message)

    warnings.append(message)





def read_chunks(filename):

    return pd.read_csv(
        ROOT / filename,
        chunksize=CHUNK,
        low_memory=False,
        keep_default_na=False,
    )





def numeric(series):

    return pd.to_numeric(series, errors='coerce')





def valid_years(df, filename):

    year = numeric(df['year'])

    decade = numeric(df['decade'])

    check(year.notna().all() and year.between(START_YEAR, END_YEAR).all(),

          f'{filename}: all years within {START_YEAR}–{END_YEAR}')

    check((decade == (year // 10) * 10).all(),

          f'{filename}: decades match release years')





def main():

    expected = ['movies.csv', *RELATIONS,

                'australian_state_spending.csv',

                'australian_spending_totals.csv',

                'australian_genre_shares.csv']

    print('FIT3179 DV2 — PROCESSED DATA VALIDATION\n')

    missing = [f for f in expected if not (ROOT / f).is_file()]

    if missing:

        for name in missing:

            check(False, f'Missing required file: {name}')

        return 1



    # 1. Master movie table; collect IDs to validate foreign keys.

    movie_ids = set()

    movie_rows = 0

    duplicate_ids = 0

    australian_ids = set()

    print('\n=== MOVIES ===')

    for chunk in read_chunks('movies.csv'):

        required = {'id','year','decade','release_date','budget','revenue',

                    'vote_average','vote_count','imdb_rating','imdb_votes'}

        if not required.issubset(chunk.columns):

            check(False, f'movies.csv missing columns: {sorted(required - set(chunk.columns))}')

            return 1

        ids = numeric(chunk['id'])

        check(ids.notna().all(), 'Movie chunk has valid IDs')

        for ident in ids.dropna().astype('int64'):

            if ident in movie_ids:

                duplicate_ids += 1

            movie_ids.add(ident)

        movie_rows += len(chunk)

        valid_years(chunk, 'movies.csv')

        dates = pd.to_datetime(chunk['release_date'], errors='coerce')

        check(dates.notna().all() and (dates.dt.year == numeric(chunk['year'])).all(),

              'Movie release dates are valid and match year')

        for field in ('budget','revenue'):

            values = numeric(chunk[field])

            check((values.dropna() > 0).all(),

                  f'{field}: no zero/negative non-null values')

        for field in ('vote_average','imdb_rating'):

            values = numeric(chunk[field]).dropna()

            check(values.between(0, 10).all(), f'{field}: within 0–10')

        for field in ('vote_count','imdb_votes'):

            values = numeric(chunk[field]).dropna()

            check((values >= 0).all(), f'{field}: no negative vote counts')

    check(movie_rows > 0, 'Movies table is not empty')

    check(duplicate_ids == 0 and len(movie_ids) == movie_rows,

          f'Movie IDs unique ({movie_rows:,} rows; {duplicate_ids:,} duplicates)')



    # 2. Each exploded relationship should reference a movie and have no

    # duplicate (id, value) pair, including duplicates across chunk boundaries.

    print('\n=== RELATIONSHIP TABLES ===')

    for filename, field in RELATIONS.items():

        seen_pairs = set()

        rows = 0

        orphan_rows = 0

        duplicates = 0

        invalid_labels = 0

        invalid_years = 0

        for chunk in read_chunks(filename):

            required = {'id','year','decade',field}

            if not required.issubset(chunk.columns):

                check(False, f'{filename} missing columns: {sorted(required - set(chunk.columns))}')

                break

            rows += len(chunk)

            ids = numeric(chunk['id'])

            labels = chunk[field].astype('string').str.strip()

            invalid_labels += int((labels.isna() | labels.eq('').fillna(True)).sum())

            years = numeric(chunk['year'])

            decades = numeric(chunk['decade'])

            invalid_years += int((~years.between(START_YEAR, END_YEAR) |

                                  decades.ne((years // 10) * 10)).sum())

            for ident, label in zip(ids, labels):

                if pd.isna(ident) or int(ident) not in movie_ids:

                    orphan_rows += 1

                if pd.notna(ident) and pd.notna(label):

                    pair = (int(ident), str(label))

                    if pair in seen_pairs:

                        duplicates += 1

                    else:

                        seen_pairs.add(pair)

                if filename == 'movie_countries.csv' and label == 'Australia' and pd.notna(ident):

                    australian_ids.add(int(ident))

        check(rows > 0, f'{filename}: not empty ({rows:,} rows)')

        check(orphan_rows == 0, f'{filename}: no orphan movie IDs ({orphan_rows:,})')

        check(duplicates == 0, f'{filename}: no duplicate movie–{field} pairs ({duplicates:,})')

        check(invalid_labels == 0, f'{filename}: nonblank {field} labels')

        check(invalid_years == 0, f'{filename}: valid year/decade values')

    print(f'Australian movies found in movie_countries.csv: {len(australian_ids):,}')



    # 3. Australian spending: only the four individually reported states.

    print('\n=== SCREEN AUSTRALIA SPENDING ===')

    states = pd.read_csv(ROOT / 'australian_state_spending.csv')

    totals = pd.read_csv(ROOT / 'australian_spending_totals.csv')

    expected_states = {'NSW','QLD','VIC','SA'}

    required_state = {'financial_year','state_code','state','spend_m_aud'}

    required_total = {'financial_year','other_m_aud','total_m_aud'}

    if not required_state.issubset(states.columns) or not required_total.issubset(totals.columns):

        check(False, 'Australian spending tables have expected columns')

    else:

        check(set(states['state_code'].dropna()) == expected_states,

              'Only NSW, QLD, VIC, SA receive state symbols')

        check(not states.duplicated(['financial_year','state_code']).any(),

              'No duplicate state–financial-year rows')

        check(not totals.duplicated(['financial_year']).any(),

              'No duplicate financial-year totals')

        check(set(states['financial_year']) == set(totals['financial_year']),

              'State and national financial years match')

        pattern = re.compile(r'^\d{4}/\d{2}$')

        check(all(bool(pattern.fullmatch(str(y))) for y in totals['financial_year']),

              'Financial-year labels use YYYY/YY')

        for df, cols in [(states,['spend_m_aud']),

                         (totals,['other_m_aud','total_m_aud'])]:

            for col in cols:

                values = numeric(df[col])

                check((values.dropna() >= 0).all(), f'{col}: no negative expenditure')

        # Do not assume totals must exactly equal rounded state sums.

        merged = states.groupby('financial_year')['spend_m_aud'].sum(min_count=1).reset_index()

        merged = merged.merge(totals, on='financial_year')

        complete = merged[['spend_m_aud','other_m_aud','total_m_aud']].notna().all(axis=1)

        differences = (merged.loc[complete,'spend_m_aud'] +

                       merged.loc[complete,'other_m_aud'] -

                       merged.loc[complete,'total_m_aud']).abs()

        if (differences > 2).any():

            warn(f'{int((differences > 2).sum())} years differ by >$2m from state+Other totals; review source rounding/suppression')

        if states['spend_m_aud'].isna().any() or totals['other_m_aud'].isna().any():

            warn('Some expenditure values are suppressed or missing; do not interpret as zero')



    # 4. Screen Australia genres: percentages are cohort shares, not counts.

    print('\n=== SCREEN AUSTRALIA GENRES ===')

    genres = pd.read_csv(ROOT / 'australian_genre_shares.csv')

    required_genres = {'genre','cohort','share_pct'}

    if not required_genres.issubset(genres.columns):

        check(False, 'Australian genre shares have expected columns')

    else:

        check(not genres.duplicated(['genre','cohort']).any(),

              'No duplicate genre–cohort rows')

        shares = numeric(genres['share_pct'])

        check(shares.dropna().between(0,100).all(),

              'Genre percentages between 0 and 100')

        check(genres['genre'].notna().all() and genres['cohort'].notna().all(),

              'Genre/cohort labels present')

        for cohort, group in genres.groupby('cohort'):

            total = numeric(group['share_pct']).sum()

            print(f'  {cohort}: reported shares sum to {total:.1f}%')

            if not 95 <= total <= 105:

                warn(f'{cohort}: percentages do not sum to ~100%; verify multi-genre or source definitions')



    print('\n=== SUMMARY ===')

    print(f'Checks failed: {len(issues)}; warnings: {len(warnings)}')

    if issues:

        print('FAILURES:')

        for item in issues:

            print(' -', item)

        return 1

    print('VALIDATION PASSED (review warnings, if any).')

    print('Note: status=Released is enforced upstream but not stored in movies.csv; verify against preprocessing script.')

    return 0





if __name__ == '__main__':

    sys.exit(main())
