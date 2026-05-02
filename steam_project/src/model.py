"""
Modeling pipeline: predict whether a Steam game will be successful.

Loads features from the SQLite database built by etl.py, trains a
Random Forest classifier and a Logistic Regression baseline, evaluates
both with cross-validation, and writes all reporting figures to /outputs.

Author: Ayaan Alam, Dawud Rana, Ashiyam Ahmed
Course: CS 210 - Data Management for Data Science
"""
import json
import sqlite3
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parent.parent
DB_PATH  = ROOT / 'db' / 'steam.db'
OUT_DIR  = ROOT / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_features(db_path: Path):
    """
    Pulls per-game numeric features + one-hot encoded genres from SQLite.
    Demonstrates the value of the relational design: genres come from a
    JOIN through the junction table, not a string split.
    """
    conn = sqlite3.connect(db_path)

    games = pd.read_sql_query("""
        SELECT appid, price, release_month, required_age, achievements,
               is_successful
        FROM games
    """, conn)

    genre_long = pd.read_sql_query("""
        SELECT gg.appid, g.genre_name
        FROM   game_genre gg
        JOIN   genres g ON gg.genre_id = g.genre_id
    """, conn)
    genre_wide = (genre_long.assign(v=1)
                  .pivot_table(index='appid', columns='genre_name',
                               values='v', fill_value=0)
                  .add_prefix('genre_'))

    tag_long = pd.read_sql_query("""
        SELECT gt.appid, t.tag_name
        FROM   game_tag gt
        JOIN   tags t ON gt.tag_id = t.tag_id
    """, conn)
    tag_wide = (tag_long.assign(v=1)
                .pivot_table(index='appid', columns='tag_name',
                             values='v', fill_value=0)
                .add_prefix('tag_'))

    conn.close()

    df = (games.set_index('appid')
                .join(genre_wide, how='left')
                .join(tag_wide,   how='left')
                .fillna(0))

    y = df.pop('is_successful').astype(int)
    X = df
    print(f'[FEATURES] X shape = {X.shape}, '
          f'positive rate = {y.mean():.1%}')
    return X, y


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, class_weight='balanced',
                               random_state=RANDOM_STATE),
            X_train_s, X_test_s,
        ),
        'Random Forest': (
            RandomForestClassifier(n_estimators=200, max_depth=15,
                                   class_weight='balanced',
                                   n_jobs=-1, random_state=RANDOM_STATE),
            X_train, X_test,
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, (model, X_tr, X_te) in models.items():
        print(f'\n[TRAIN] {name}')
        model.fit(X_tr, y_train)
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        cv_scores = cross_val_score(model, X_tr, y_train,
                                    cv=cv, scoring='f1', n_jobs=-1)

        report = classification_report(y_test, y_pred, output_dict=True,
                                       zero_division=0)
        results[name] = {
            'model':   model,
            'y_pred':  y_pred,
            'y_proba': y_proba,
            'report':  report,
            'cv_f1_mean': float(cv_scores.mean()),
            'cv_f1_std':  float(cv_scores.std()),
            'roc_auc':    float(roc_auc_score(y_test, y_proba)),
            'avg_prec':   float(average_precision_score(y_test, y_proba)),
        }
        print(f'        CV F1: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}')
        print(f'        Test F1 (success): '
              f'{report["1"]["f1-score"]:.3f}')
        print(f'        Test ROC-AUC:     {results[name]["roc_auc"]:.3f}')

    return results, X_train.columns, y_test


def plot_class_distribution(y, out):
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['Not Successful (0)', 'Successful (1)'],
                  counts.values, color=['#888', '#3a7'])
    for b, n in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f'{n:,}\n({100*n/len(y):.1f}%)',
                ha='center', va='bottom', fontsize=10)
    ax.set_title('Target Variable Distribution (Class Imbalance)')
    ax.set_ylabel('Number of games')
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_confusion_matrices(results, y_test, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, r['y_pred'])
        ConfusionMatrixDisplay(cm, display_labels=['Not Succ.', 'Succ.']
                               ).plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_roc_curves(results, y_test, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
        ax.plot(fpr, tpr, label=f'{name} (AUC = {r["roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Random')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves'); ax.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def plot_pr_curves(results, y_test, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(y_test, r['y_proba'])
        ax.plot(rec, prec, label=f'{name} (AP = {r["avg_prec"]:.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves'); ax.legend(loc='lower left')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def plot_feature_importance(rf_model, feature_names, out, top_n=20):
    imp = pd.Series(rf_model.feature_importances_,
                    index=feature_names).sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(imp.index, imp.values, color='#3a7')
    ax.set_title(f'Top {top_n} Random Forest Feature Importances')
    ax.set_xlabel('Importance')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def plot_success_by_month(db_path, out):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT release_month,
               100.0 * AVG(is_successful) AS success_rate_pct,
               COUNT(*) AS n
        FROM games GROUP BY release_month ORDER BY release_month
    """, conn)
    conn.close()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df['release_month'], df['success_rate_pct'], color='#5a8')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_ylabel('Success rate (%)')
    ax.set_title('Game Success Rate by Release Month')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def plot_success_by_price(db_path, out):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
            CASE
                WHEN price = 0   THEN 'Free'
                WHEN price < 5   THEN '<$5'
                WHEN price < 10  THEN '$5-10'
                WHEN price < 20  THEN '$10-20'
                WHEN price < 40  THEN '$20-40'
                ELSE                  '$40+'
            END AS bucket,
            100.0 * AVG(is_successful) AS rate
        FROM games GROUP BY bucket
    """, conn)
    conn.close()
    order = ['Free','<$5','$5-10','$10-20','$20-40','$40+']
    df['bucket'] = pd.Categorical(df['bucket'], categories=order, ordered=True)
    df = df.sort_values('bucket')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df['bucket'].astype(str), df['rate'], color='#48a')
    ax.set_ylabel('Success rate (%)')
    ax.set_title('Game Success Rate by Price Bucket')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def main():
    X, y = load_features(DB_PATH)

    plot_class_distribution(y, OUT_DIR / 'fig_class_distribution.png')
    plot_success_by_month(DB_PATH, OUT_DIR / 'fig_success_by_month.png')
    plot_success_by_price(DB_PATH, OUT_DIR / 'fig_success_by_price.png')

    results, feature_names, y_test = train_and_evaluate(X, y)

    plot_confusion_matrices(results, y_test, OUT_DIR / 'fig_confusion_matrices.png')
    plot_roc_curves(results, y_test, OUT_DIR / 'fig_roc_curves.png')
    plot_pr_curves(results, y_test, OUT_DIR / 'fig_pr_curves.png')
    plot_feature_importance(results['Random Forest']['model'], feature_names,
                            OUT_DIR / 'fig_feature_importance.png')

    summary = {
        name: {
            'cv_f1_mean':   r['cv_f1_mean'],
            'cv_f1_std':    r['cv_f1_std'],
            'roc_auc':      r['roc_auc'],
            'avg_precision':r['avg_prec'],
            'precision_pos':r['report']['1']['precision'],
            'recall_pos':   r['report']['1']['recall'],
            'f1_pos':       r['report']['1']['f1-score'],
            'accuracy':     r['report']['accuracy'],
        }
        for name, r in results.items()
    }
    with open(OUT_DIR / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n[SUMMARY]')
    print(json.dumps(summary, indent=2))
    print(f'\nAll figures saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
