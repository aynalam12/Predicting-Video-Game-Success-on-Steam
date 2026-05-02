"""
ETL pipeline: load raw Steam CSV into a normalized SQLite database.

Pipeline stages:
    1. EXTRACT  - read CSV from disk
    2. TRANSFORM - clean nulls, parse owner ranges, derive temporal
                   features, compute rating ratio and target variable
    3. LOAD     - normalize categorical fields into reference tables,
                  populate junction tables, write to SQLite

Author: Ayaan Alam, Dawud Rana, Ashiyam Ahmed
Course: CS 210 - Data Management for Data Science
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / 'data' / 'steam.csv'
DB_PATH     = ROOT / 'db'   / 'steam.db'
SCHEMA_PATH = ROOT / 'db'   / 'schema.sql'

# Target-variable thresholds. Justified in the report:
#   - 80% positive matches Steam's own "Very Positive" badge cutoff
#   - 50,000 owners excludes hobby releases without filtering out all
#     successful indies (Steam Spy's third-lowest bucket)
SUCCESS_RATING_THRESHOLD = 0.80
SUCCESS_OWNERS_THRESHOLD = 50_000


# ---------------------------------------------------------------------
# EXTRACT
# ---------------------------------------------------------------------
def extract(csv_path: Path) -> pd.DataFrame:
    print(f'[EXTRACT] reading {csv_path.name}')
    df = pd.read_csv(csv_path)
    print(f'          {len(df):,} raw rows, {df.shape[1]} columns')
    return df


# ---------------------------------------------------------------------
# TRANSFORM
# ---------------------------------------------------------------------
def parse_owner_range(s):
    """'50000-100000' -> (50000, 100000). Returns (NaN, NaN) on bad input."""
    if not isinstance(s, str) or '-' not in s:
        return np.nan, np.nan
    lo, hi = s.split('-', 1)
    try:
        return int(lo), int(hi)
    except ValueError:
        return np.nan, np.nan


def transform(df: pd.DataFrame) -> pd.DataFrame:
    print('[TRANSFORM] cleaning + feature engineering')
    before = len(df)

    # Drop rows missing fields we cannot impute
    df = df.dropna(subset=['genres', 'price', 'release_date',
                           'positive_ratings', 'owners']).copy()
    print(f'            dropped {before - len(df):,} rows with critical nulls')

    # Parse owner range -> two integer columns
    parsed = df['owners'].apply(parse_owner_range)
    df['owners_min'] = parsed.apply(lambda x: x[0])
    df['owners_max'] = parsed.apply(lambda x: x[1])

    # Temporal features
    df['release_date']  = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['release_date'])
    df['release_year']  = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month

    # Rating ratio + target variable
    total = df['positive_ratings'] + df['negative_ratings']
    df['rating_ratio'] = (df['positive_ratings'] / total).fillna(0)
    df['is_successful'] = (
        (df['rating_ratio']  >= SUCCESS_RATING_THRESHOLD) &
        (df['owners_min']    >= SUCCESS_OWNERS_THRESHOLD)
    ).astype(int)

    pos = int(df['is_successful'].sum())
    print(f'            target: {pos:,} successful / {len(df):,} total '
          f'({100*pos/len(df):.1f}%)')
    return df


# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------
def _build_lookup(conn, table, name_col, values):
    """Insert unique values, return {name: id} mapping."""
    unique = sorted({v for v in values if isinstance(v, str) and v.strip()})
    conn.executemany(
        f'INSERT OR IGNORE INTO {table} ({name_col}) VALUES (?)',
        [(v,) for v in unique]
    )
    cur = conn.execute(f'SELECT {name_col}, {table[:-1]}_id FROM {table}')
    return {n: i for n, i in cur.fetchall()}


def load(df: pd.DataFrame, db_path: Path, schema_path: Path) -> None:
    print(f'[LOAD] writing to {db_path.name}')
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.executescript(schema_path.read_text())

    # 1. populate reference tables
    dev_map = _build_lookup(conn, 'developers', 'developer_name', df['developer'])
    pub_map = _build_lookup(conn, 'publishers', 'publisher_name', df['publisher'])

    all_genres = {g for row in df['genres'].dropna() for g in row.split(';') if g}
    all_tags   = {t for row in df['steamspy_tags'].dropna()
                  for t in row.split(';') if t}
    genre_map = _build_lookup(conn, 'genres', 'genre_name', all_genres)
    tag_map   = _build_lookup(conn, 'tags',   'tag_name',   all_tags)

    # 2. core games table
    games_rows = []
    for _, r in df.iterrows():
        games_rows.append((
            int(r['appid']), r['name'],
            r['release_date'].strftime('%Y-%m-%d'),
            int(r['release_year']), int(r['release_month']),
            dev_map.get(r['developer']), pub_map.get(r['publisher']),
            float(r['price']), int(r.get('required_age', 0)),
            int(r.get('achievements', 0)),
            int(r['positive_ratings']), int(r['negative_ratings']),
            int(r.get('average_playtime', 0)), int(r.get('median_playtime', 0)),
            int(r['owners_min']), int(r['owners_max']),
            float(r['rating_ratio']), int(r['is_successful']),
        ))
    conn.executemany("""
        INSERT INTO games VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, games_rows)

    # 3. junction tables
    gg_rows, gt_rows = [], []
    for _, r in df.iterrows():
        appid = int(r['appid'])
        for g in str(r['genres']).split(';'):
            if g in genre_map:
                gg_rows.append((appid, genre_map[g]))
        for t in str(r['steamspy_tags']).split(';'):
            if t in tag_map:
                gt_rows.append((appid, tag_map[t]))
    conn.executemany('INSERT OR IGNORE INTO game_genre VALUES (?, ?)', gg_rows)
    conn.executemany('INSERT OR IGNORE INTO game_tag   VALUES (?, ?)', gt_rows)

    conn.commit()

    # quick sanity check
    for tbl in ['games', 'developers', 'publishers', 'genres', 'tags',
                'game_genre', 'game_tag']:
        n = conn.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]
        print(f'        {tbl:14s} {n:>8,} rows')
    conn.close()


def main():
    df = extract(DATA_PATH)
    df = transform(df)
    load(df, DB_PATH, SCHEMA_PATH)
    print('[DONE]')


if __name__ == '__main__':
    main()
