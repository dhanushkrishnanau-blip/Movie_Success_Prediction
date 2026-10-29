from typing import Any

import numpy as np
import pandas as pd

from app.services.dataset_service import get_clean_dataset, get_raw_dataset


def score_distribution(bins: int = 20) -> dict[str, Any]:
    scores = get_clean_dataset()["imdb_score"].dropna()
    counts, edges = np.histogram(scores, bins=bins, range=(0, 10))
    return {
        "count": int(scores.size),
        "mean": round(float(scores.mean()), 3),
        "median": round(float(scores.median()), 3),
        "bins": [
            {
                "start": round(float(edges[index]), 2),
                "end": round(float(edges[index + 1]), 2),
                "count": int(count),
            }
            for index, count in enumerate(counts)
        ],
    }


def category_distribution() -> list[dict[str, Any]]:
    categories = get_clean_dataset()["Classify"].value_counts()
    total = int(categories.sum())
    return [
        {
            "category": category,
            "count": int(categories.get(category, 0)),
            "percent": round(float(categories.get(category, 0) / total * 100), 2),
        }
        for category in ("Flop", "Average", "Hit")
    ]


def genre_statistics(limit: int = 20) -> list[dict[str, Any]]:
    frame = get_clean_dataset()[["genres", "imdb_score", "Classify"]].dropna(
        subset=["genres"]
    )
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        for genre in str(row["genres"]).split("|"):
            records.append(
                {
                    "genre": genre.strip(),
                    "imdb_score": float(row["imdb_score"]),
                    "category": str(row["Classify"]),
                }
            )
    expanded = pd.DataFrame(records)
    if expanded.empty:
        return []
    grouped = expanded.groupby("genre").agg(
        movies=("imdb_score", "size"), mean_score=("imdb_score", "mean")
    )
    grouped = grouped.sort_values(["movies", "mean_score"], ascending=[False, False])
    return [
        {
            "genre": str(index),
            "movies": int(row["movies"]),
            "mean_score": round(float(row["mean_score"]), 3),
        }
        for index, row in grouped.head(limit).iterrows()
    ]


def correlation_matrix() -> dict[str, Any]:
    frame = get_clean_dataset().select_dtypes(include="number")
    matrix = frame.corr(numeric_only=True).round(3)
    return {
        "columns": list(matrix.columns),
        "values": [
            [None if pd.isna(value) else float(value) for value in row]
            for row in matrix.to_numpy()
        ],
    }


def category_score_summary() -> list[dict[str, Any]]:
    frame = get_clean_dataset()
    result = []
    for category in ("Flop", "Average", "Hit"):
        scores = frame.loc[frame["Classify"] == category, "imdb_score"]
        result.append(
            {
                "category": category,
                "count": int(scores.size),
                "mean_score": round(float(scores.mean()), 3) if not scores.empty else None,
            }
        )
    return result


def country_distribution(limit: int = 12) -> list[dict[str, Any]]:
    counts = get_raw_dataset()["country"].dropna().value_counts().head(limit)
    return [
        {"country": str(country), "movies": int(count)}
        for country, count in counts.items()
    ]


def relationships() -> dict[str, Any]:
    frame = get_clean_dataset()
    budget_rows = frame[["budget", "imdb_score"]].dropna()
    if len(budget_rows) > 600:
        budget_rows = budget_rows.sample(n=600, random_state=42)
    year_rows = frame[["title_year", "imdb_score"]].dropna()
    by_year = (
        year_rows.groupby("title_year")["imdb_score"]
        .agg(["mean", "size"])
        .query("size >= 5")
        .reset_index()
        .sort_values("title_year")
    )
    return {
        "budget_vs_score": [
            {"budget": float(row.budget), "imdb_score": float(row.imdb_score)}
            for row in budget_rows.itertuples(index=False)
        ],
        "budget_sample_size": int(len(budget_rows)),
        "budget_non_missing": int(frame["budget"].notna().sum()),
        "year_trends": [
            {
                "year": int(row.title_year),
                "mean_score": round(float(row.mean), 3),
                "movies": int(row.size),
            }
            for row in by_year.itertuples(index=False)
        ],
    }
