-- =====================================================================
-- Steam Games Database Schema
-- CS 210: Data Management for Data Science
--
-- Design notes:
--   * Normalized to 3NF: developers, publishers, genres, and tags are
--     factored into their own tables to remove transitive dependencies.
--   * Many-to-many relationships (game <-> genre, game <-> tag) use
--     junction tables.
--   * owners_min and owners_max are stored as integers (parsed from
--     the Steam Spy "0-20000" string format) so we can query numerically.
--   * Indexes are placed on join columns and on columns used in our
--     analytical queries (release_year, owners_min).
-- =====================================================================

DROP TABLE IF EXISTS game_genre;
DROP TABLE IF EXISTS game_tag;
DROP TABLE IF EXISTS games;
DROP TABLE IF EXISTS genres;
DROP TABLE IF EXISTS tags;
DROP TABLE IF EXISTS developers;
DROP TABLE IF EXISTS publishers;

-- ---------------------------------------------------------------------
-- Reference tables
-- ---------------------------------------------------------------------
CREATE TABLE developers (
    developer_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    developer_name TEXT NOT NULL UNIQUE
);

CREATE TABLE publishers (
    publisher_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    publisher_name TEXT NOT NULL UNIQUE
);

CREATE TABLE genres (
    genre_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    genre_name TEXT NOT NULL UNIQUE
);

CREATE TABLE tags (
    tag_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT NOT NULL UNIQUE
);

-- ---------------------------------------------------------------------
-- Core entity table
-- ---------------------------------------------------------------------
CREATE TABLE games (
    appid             INTEGER PRIMARY KEY,
    name              TEXT NOT NULL,
    release_date      DATE,
    release_year      INTEGER,
    release_month     INTEGER,
    developer_id      INTEGER,
    publisher_id      INTEGER,
    price             REAL,
    required_age      INTEGER,
    achievements      INTEGER,
    positive_ratings  INTEGER,
    negative_ratings  INTEGER,
    average_playtime  INTEGER,
    median_playtime   INTEGER,
    owners_min        INTEGER,
    owners_max        INTEGER,
    -- denormalized convenience field for ML pipeline
    rating_ratio      REAL,
    is_successful     INTEGER,    -- 0 or 1, the ML target
    FOREIGN KEY (developer_id) REFERENCES developers(developer_id),
    FOREIGN KEY (publisher_id) REFERENCES publishers(publisher_id)
);

-- ---------------------------------------------------------------------
-- Junction tables (many-to-many)
-- ---------------------------------------------------------------------
CREATE TABLE game_genre (
    appid    INTEGER NOT NULL,
    genre_id INTEGER NOT NULL,
    PRIMARY KEY (appid, genre_id),
    FOREIGN KEY (appid)    REFERENCES games(appid),
    FOREIGN KEY (genre_id) REFERENCES genres(genre_id)
);

CREATE TABLE game_tag (
    appid  INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    PRIMARY KEY (appid, tag_id),
    FOREIGN KEY (appid)  REFERENCES games(appid),
    FOREIGN KEY (tag_id) REFERENCES tags(tag_id)
);

-- ---------------------------------------------------------------------
-- Indexes for analytical queries
-- ---------------------------------------------------------------------
CREATE INDEX idx_games_release_year ON games(release_year);
CREATE INDEX idx_games_owners_min   ON games(owners_min);
CREATE INDEX idx_games_is_success   ON games(is_successful);
CREATE INDEX idx_games_developer    ON games(developer_id);
CREATE INDEX idx_games_publisher    ON games(publisher_id);
CREATE INDEX idx_game_genre_genre   ON game_genre(genre_id);
CREATE INDEX idx_game_tag_tag       ON game_tag(tag_id);
