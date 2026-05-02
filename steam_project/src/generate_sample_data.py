"""
Generates a sample Steam dataset matching the schema of the
'Steam Store Games (Clean dataset)' by Nik Davis on Kaggle:
https://www.kaggle.com/datasets/nikdavis/steam-store-games

This is used for reproducibility when the real CSV is unavailable.
The real dataset has the SAME column names and types — replace
data/steam.csv with the real download to run the pipeline on it.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 27000  # roughly the real dataset size

# Realistic genre/category vocabularies pulled from Steam
GENRES = ['Action', 'Adventure', 'Casual', 'Indie', 'Massively Multiplayer',
          'Racing', 'RPG', 'Simulation', 'Sports', 'Strategy', 'Free to Play',
          'Early Access']
TAGS = ['Singleplayer', 'Multiplayer', 'Co-op', 'Atmospheric', 'Story Rich',
        'Survival', 'Horror', 'Pixel Graphics', 'Anime', 'Roguelike',
        'Open World', 'Sandbox', 'Difficult', 'Puzzle', 'Platformer',
        '2D', '3D', 'First-Person', 'Third Person', 'Fantasy', 'Sci-fi']
DEVELOPERS = [f'Studio_{i}' for i in range(1, 401)]
PUBLISHERS = [f'Publisher_{i}' for i in range(1, 251)]

def random_multilabel(pool, k_min=1, k_max=4):
    k = np.random.randint(k_min, k_max + 1)
    return ';'.join(np.random.choice(pool, size=k, replace=False))

# Owner buckets exactly as Steam Spy reports them in the Kaggle dataset
OWNER_BUCKETS = [
    '0-20000', '20000-50000', '50000-100000', '100000-200000',
    '200000-500000', '500000-1000000', '1000000-2000000',
    '2000000-5000000', '5000000-10000000', '10000000-20000000',
]
# Most games sit in the lowest buckets (heavy tail)
OWNER_PROBS = [0.55, 0.18, 0.10, 0.06, 0.04, 0.03, 0.02, 0.013, 0.005, 0.002]

release_dates = pd.to_datetime(
    np.random.choice(pd.date_range('2010-01-01', '2023-12-31'), N)
)
release_months = release_dates.month

prices = np.random.choice(
    [0.0, 0.99, 4.99, 9.99, 14.99, 19.99, 29.99, 39.99, 59.99],
    N, p=[0.10, 0.05, 0.20, 0.25, 0.15, 0.12, 0.07, 0.04, 0.02]
)

# Generate genres up front so we can inject genre-driven signal
genre_strings = [random_multilabel(GENRES, 1, 3) for _ in range(N)]
tag_strings   = [random_multilabel(TAGS,   1, 3) for _ in range(N)]

# Per-game success "score" driven by realistic factors:
#   * RPG, Strategy, Indie boost; Casual hurts
#   * October/November release (Q4 holiday window) boosts
#   * Mid-tier pricing ($10-20) boosts more than free or premium
#   * Some genuine noise so the model doesn't get a perfect score
GENRE_EFFECT = {'RPG': 1.4, 'Strategy': 1.1, 'Indie': 0.6, 'Action': 0.4,
                'Adventure': 0.5, 'Simulation': 0.3, 'Casual': -0.8,
                'Free to Play': -0.5, 'Sports': -0.3, 'Racing': -0.2,
                'Massively Multiplayer': 0.2, 'Early Access': -0.4}
TAG_EFFECT = {'Story Rich': 0.7, 'Atmospheric': 0.6, 'Open World': 0.5,
              'Roguelike': 0.4, 'Co-op': 0.3, 'Multiplayer': -0.2,
              'Difficult': 0.2, 'Pixel Graphics': 0.1}
MONTH_EFFECT = {1:-0.1,2:-0.2,3:0.0,4:0.0,5:0.1,6:-0.1,
                7:-0.2,8:-0.1,9:0.3,10:0.6,11:0.5,12:0.2}

def price_effect(p):
    if p == 0:        return -0.3
    if p < 5:         return -0.1
    if p < 20:        return 0.4
    if p < 40:        return 0.2
    return -0.1

scores = np.zeros(N)
for i in range(N):
    s = np.random.normal(-1.2, 0.7)         # baseline (most games fail)
    for g in genre_strings[i].split(';'):
        s += GENRE_EFFECT.get(g, 0)
    for t in tag_strings[i].split(';'):
        s += TAG_EFFECT.get(t, 0)
    s += MONTH_EFFECT[release_months[i]]
    s += price_effect(prices[i])
    scores[i] = s

# Convert score -> owner bucket index (higher score -> bigger bucket)
prob_success = 1 / (1 + np.exp(-scores))   # logistic squash
bucket_idx = np.clip(
    (prob_success * len(OWNER_BUCKETS)).astype(int)
    + np.random.randint(-1, 2, N),         # small jitter
    0, len(OWNER_BUCKETS) - 1
)
owners = np.array([OWNER_BUCKETS[i] for i in bucket_idx])

# Reviews scale with owners, ratio influenced by score
positive_ratings = np.zeros(N, dtype=int)
negative_ratings = np.zeros(N, dtype=int)
for i, b_idx in enumerate(bucket_idx):
    base = max(5, int(50 * (b_idx + 1) ** 1.6))
    pos_rate = 0.55 + 0.35 * (1 / (1 + np.exp(-scores[i])))   # 0.55..0.90
    pos_rate = np.clip(pos_rate + np.random.normal(0, 0.05), 0.2, 0.98)
    total = max(1, int(np.random.poisson(base) * np.random.uniform(0.7, 1.3)))
    positive_ratings[i] = int(total * pos_rate)
    negative_ratings[i] = total - positive_ratings[i]

df = pd.DataFrame({
    'appid': np.arange(10, 10 + N),
    'name': [f'Game_{i}' for i in range(N)],
    'release_date': release_dates.strftime('%Y-%m-%d'),
    'english': np.random.choice([0, 1], N, p=[0.05, 0.95]),
    'developer': np.random.choice(DEVELOPERS, N),
    'publisher': np.random.choice(PUBLISHERS, N),
    'platforms': np.random.choice(
        ['windows', 'windows;mac', 'windows;mac;linux', 'windows;linux'], N,
        p=[0.55, 0.20, 0.20, 0.05]
    ),
    'required_age': np.random.choice([0, 7, 12, 16, 18], N,
                                     p=[0.75, 0.05, 0.05, 0.10, 0.05]),
    'categories': [random_multilabel(['Single-player', 'Multi-player', 'Co-op',
                                       'Steam Achievements', 'Steam Cloud'], 1, 3)
                   for _ in range(N)],
    'genres': genre_strings,
    'steamspy_tags': tag_strings,
    'achievements': np.random.poisson(20, N),
    'positive_ratings': positive_ratings,
    'negative_ratings': negative_ratings,
    'average_playtime': np.random.poisson(120, N),
    'median_playtime': np.random.poisson(60, N),
    'owners': owners,
    'price': prices,
})

# Inject a small amount of missing data so cleaning is meaningful
mask = np.random.rand(N) < 0.01
df.loc[mask, 'genres'] = np.nan
mask = np.random.rand(N) < 0.005
df.loc[mask, 'price'] = np.nan

out = Path(__file__).resolve().parent.parent / 'data' / 'steam.csv'
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f'Wrote {len(df):,} rows to {out}')
print(f'Columns: {list(df.columns)}')
