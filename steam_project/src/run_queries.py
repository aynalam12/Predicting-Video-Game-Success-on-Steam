"""Run the analytical SQL queries and dump the results as both pretty
text and a JSON file the report can reference."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
conn = sqlite3.connect(ROOT / 'db' / 'steam.db')

QUERIES = {
    'class_balance': """
        SELECT is_successful,
               COUNT(*) AS n_games,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM games), 2) AS pct
        FROM games GROUP BY is_successful
    """,
    'top_genres': """
        SELECT g.genre_name,
               COUNT(*) AS total_games,
               SUM(games.is_successful) AS successful_games,
               ROUND(100.0 * SUM(games.is_successful) / COUNT(*), 2) AS success_rate_pct
        FROM games
        JOIN game_genre gg ON games.appid = gg.appid
        JOIN genres     g  ON gg.genre_id = g.genre_id
        GROUP BY g.genre_name
        HAVING COUNT(*) >= 100
        ORDER BY success_rate_pct DESC
        LIMIT 10
    """,
    'success_by_month': """
        SELECT release_month,
               COUNT(*) AS n_games,
               ROUND(100.0 * AVG(is_successful), 2) AS success_rate_pct
        FROM games GROUP BY release_month ORDER BY release_month
    """,
    'success_by_price': """
        SELECT
            CASE
                WHEN price = 0 THEN 'Free'
                WHEN price < 5 THEN '<$5'
                WHEN price < 10 THEN '$5-10'
                WHEN price < 20 THEN '$10-20'
                WHEN price < 40 THEN '$20-40'
                ELSE '$40+' END AS price_bucket,
            COUNT(*) AS n_games,
            ROUND(100.0 * AVG(is_successful), 2) AS success_rate_pct
        FROM games GROUP BY price_bucket
    """,
    'top_tags': """
        SELECT t.tag_name,
               COUNT(*) AS appearances,
               ROUND(100.0 * AVG(games.is_successful), 2) AS success_rate_pct
        FROM games
        JOIN game_tag gt ON games.appid = gt.appid
        JOIN tags     t  ON gt.tag_id   = t.tag_id
        GROUP BY t.tag_name
        HAVING COUNT(*) >= 100
        ORDER BY success_rate_pct DESC
        LIMIT 15
    """,
}

results = {}
for name, sql in QUERIES.items():
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    results[name] = [dict(zip(cols, r)) for r in rows]
    print(f'\n==== {name} ====')
    print(' | '.join(cols))
    for r in rows:
        print(' | '.join(str(x) for x in r))

with open(ROOT / 'outputs' / 'sql_results.json', 'w') as f:
    json.dump(results, f, indent=2)

conn.close()
