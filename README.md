# Predicting Video Game Success on Steam

**CS 210: Data Management for Data Science — Final Project**
Team: Ayaan Alam (aaa557), Dawud Rana (dar333), Ashiyam Ahmed (maa620)

## Overview

End-to-end pipeline that loads the Steam Store Games dataset into a
normalized SQLite database, derives a per-game success label, and trains
two classifiers (Logistic Regression baseline + Random Forest) to predict
whether a game will be successful from pre-launch features only.

## Repository Layout

```
steam_project/
├── data/                     # CSV inputs
│   └── steam.csv             # generated sample (replace with real Kaggle file)
├── db/
│   ├── schema.sql            # 3NF relational schema
│   ├── queries.sql           # analytical SQL used in the report
│   └── steam.db              # SQLite database produced by ETL
├── src/
│   ├── generate_sample_data.py   # creates a sample CSV with realistic schema
│   ├── etl.py                # extract -> transform -> load
│   ├── run_queries.py        # runs analytical SQL, writes results JSON
│   └── model.py              # trains both models, writes figures + metrics
├── outputs/                  # generated figures, metrics.json, sql_results.json
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. (optional) generate a sample CSV — skip if using real Kaggle data
python src/generate_sample_data.py

# 3. build the SQLite database
python src/etl.py

# 4. run the SQL analytics
python src/run_queries.py

# 5. train models + produce all report figures
python src/model.py
```

All figures and metrics are written to `outputs/`.

## Using the Real Dataset

Download the **Steam Store Games (Clean dataset)** by Nik Davis from
Kaggle: https://www.kaggle.com/datasets/nikdavis/steam-store-games

Place `steam.csv` in `data/` (overwriting the sample). The pipeline
expects the file to contain at least these columns: `appid`, `name`,
`release_date`, `developer`, `publisher`, `genres`, `steamspy_tags`,
`positive_ratings`, `negative_ratings`, `owners`, `price`,
`required_age`, `achievements`, `average_playtime`, `median_playtime`.

## Target Variable

A game is labelled `is_successful = 1` when **both** conditions hold:

* `positive_ratings / (positive_ratings + negative_ratings) >= 0.80`
  (matches Steam's own "Very Positive" badge)
* `owners_min >= 50,000` (parsed from the Steam Spy `"min-max"` range)

About 14-18% of games in the dataset meet this bar — a realistic class
imbalance that motivated the use of `class_weight='balanced'` and our
focus on F1 / ROC-AUC rather than accuracy.

## Contributions

| Member | Primary responsibility |
|---|---|
| Ayaan Alam   | Database schema, SQL queries, ETL pipeline |
| Dawud Rana   | Feature engineering, model training, evaluation |
| Ashiyam Ahmed | Visualizations, report writing, repository organization |

All members contributed to design discussions and final report editing.
