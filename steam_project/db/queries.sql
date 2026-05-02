-- =====================================================================
-- Analytical SQL queries used in the report.
-- Each query answers a specific research question.
-- =====================================================================

-- Q1: Class balance — what fraction of games are "successful"?
-- ---------------------------------------------------------------------
SELECT
    is_successful,
    COUNT(*) AS n_games,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM games), 2) AS pct
FROM games
GROUP BY is_successful;


-- Q2: Top 10 genres by success rate (joins games <-> game_genre <-> genres)
-- ---------------------------------------------------------------------
SELECT
    g.genre_name,
    COUNT(*) AS total_games,
    SUM(games.is_successful) AS successful_games,
    ROUND(100.0 * SUM(games.is_successful) / COUNT(*), 2) AS success_rate_pct
FROM games
JOIN game_genre gg ON games.appid    = gg.appid
JOIN genres     g  ON gg.genre_id    = g.genre_id
GROUP BY g.genre_name
HAVING COUNT(*) >= 100
ORDER BY success_rate_pct DESC
LIMIT 10;


-- Q3: Success rate by release month (seasonal effect?)
-- ---------------------------------------------------------------------
SELECT
    release_month,
    COUNT(*) AS n_games,
    ROUND(100.0 * AVG(is_successful), 2) AS success_rate_pct
FROM games
GROUP BY release_month
ORDER BY release_month;


-- Q4: Success rate by price bucket
-- ---------------------------------------------------------------------
SELECT
    CASE
        WHEN price = 0          THEN '0_free'
        WHEN price < 5          THEN '1_under_5'
        WHEN price < 10         THEN '2_5_to_10'
        WHEN price < 20         THEN '3_10_to_20'
        WHEN price < 40         THEN '4_20_to_40'
        ELSE                         '5_40_plus'
    END AS price_bucket,
    COUNT(*) AS n_games,
    ROUND(100.0 * AVG(is_successful), 2) AS success_rate_pct
FROM games
GROUP BY price_bucket
ORDER BY price_bucket;


-- Q5: Top 10 developers by number of successful games
-- ---------------------------------------------------------------------
SELECT
    d.developer_name,
    COUNT(*)                       AS games_released,
    SUM(games.is_successful)       AS successful_games,
    ROUND(AVG(games.rating_ratio), 3) AS avg_rating_ratio
FROM games
JOIN developers d ON games.developer_id = d.developer_id
GROUP BY d.developer_name
HAVING COUNT(*) >= 5
ORDER BY successful_games DESC, avg_rating_ratio DESC
LIMIT 10;


-- Q6: Most common tag combinations in successful games
-- ---------------------------------------------------------------------
SELECT
    t.tag_name,
    COUNT(*)                                          AS appearances,
    ROUND(100.0 * AVG(games.is_successful), 2)        AS success_rate_pct
FROM games
JOIN game_tag gt ON games.appid = gt.appid
JOIN tags     t  ON gt.tag_id   = t.tag_id
GROUP BY t.tag_name
HAVING COUNT(*) >= 100
ORDER BY success_rate_pct DESC
LIMIT 15;
