from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from app.config import DATASET_PATH

TARGET = "Classify"
FEATURE_COLUMNS = [
    "director_name",
    "duration",
    "genres",
    "actor_1_name",
    "actor_2_name",
    "actor_3_name",
    "budget",
    "title_year",
    "language",
    "country",
]
PERSON_NAME_COLUMNS = [
    "director_name",
    "actor_1_name",
    "actor_2_name",
    "actor_3_name",
]
CLASS_LABELS = ["Flop", "Average", "Hit"]


@lru_cache(maxsize=1)
def get_raw_dataset() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


@lru_cache(maxsize=1)
def get_clean_dataset() -> pd.DataFrame:
    raw = get_raw_dataset()
    cleaned = raw.drop_duplicates().dropna(subset=["imdb_score"]).copy()
    cleaned[TARGET] = pd.cut(
        cleaned["imdb_score"],
        bins=[0, 3, 6, 10],
        labels=CLASS_LABELS,
        right=False,
        include_lowest=True,
    )
    return cleaned.dropna(subset=[TARGET]).copy()


def json_safe(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def column_schema() -> list[dict[str, Any]]:
    raw = get_raw_dataset()
    result = []
    for column in raw.columns:
        series = raw[column]
        result.append(
            {
                "name": column,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "missing_percent": round(float(series.isna().mean() * 100), 2),
                "unique": int(series.nunique(dropna=True)),
            }
        )
    return result


def suggest_person_names(query: str, limit: int = 10) -> list[str]:
    raw = get_raw_dataset()
    normalized_query = query.casefold()
    matches: dict[str, str] = {}
    for column in PERSON_NAME_COLUMNS:
        for value in raw[column].dropna().astype(str):
            normalized_value = value.casefold()
            if normalized_query in normalized_value:
                matches.setdefault(normalized_value, value.strip())
    return sorted(
        matches.values(),
        key=lambda name: (
            not name.casefold().startswith(normalized_query),
            name.casefold(),
        ),
    )[:limit]


def overview() -> dict[str, Any]:
    raw = get_raw_dataset()
    clean = get_clean_dataset()
    numeric = raw.select_dtypes(include="number").columns
    categorical = raw.select_dtypes(exclude="number").columns
    genre_count = (
        clean["genres"].dropna().astype(str).str.split("|").explode().nunique()
        if "genres" in clean
        else 0
    )
    return {
        "filename": DATASET_PATH.name,
        "raw_rows": int(len(raw)),
        "raw_columns": int(len(raw.columns)),
        "clean_rows": int(len(clean)),
        "numeric_columns": int(len(numeric)),
        "categorical_columns": int(len(categorical)),
        "missing_values": int(raw.isna().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "unique_genres": int(genre_count),
        "raw_population_note": "Raw CSV, before notebook cleaning.",
        "clean_population_note": (
            "Notebook cleaning: remove duplicate rows and records without imdb_score; "
            "derive Classify using [0,3), [3,6), and [6,10]."
        ),
    }


def table_rows(
    page: int = 1,
    page_size: int = 25,
    search: str = "",
    sort_by: str = "movie_title",
    direction: str = "asc",
    filters: dict[str, str] | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
) -> dict[str, Any]:
    frame = get_raw_dataset()
    frame = frame.copy()
    if search:
        searchable = [
            column
            for column in (
                "movie_title",
                "director_name",
                "genres",
                "country",
                "language",
            )
            if column in frame
        ]
        mask = pd.Series(False, index=frame.index)
        for column in searchable:
            mask |= frame[column].astype("string").str.contains(
                search, case=False, na=False, regex=False
            )
        frame = frame[mask]
    for column, value in (filters or {}).items():
        if column in frame.columns and value:
            frame = frame[
                frame[column].astype("string").str.contains(
                    value, case=False, na=False, regex=False
                )
            ]
    if min_score is not None and "imdb_score" in frame:
        frame = frame[frame["imdb_score"] >= min_score]
    if max_score is not None and "imdb_score" in frame:
        frame = frame[frame["imdb_score"] <= max_score]
    if min_year is not None and "title_year" in frame:
        frame = frame[frame["title_year"] >= min_year]
    if max_year is not None and "title_year" in frame:
        frame = frame[frame["title_year"] <= max_year]
    if sort_by in frame.columns:
        frame = frame.sort_values(
            sort_by, ascending=direction != "desc", na_position="last"
        )
    total = len(frame)
    start = (page - 1) * page_size
    items = []
    for index, row in frame.iloc[start : start + page_size].iterrows():
        item = {key: json_safe(value) for key, value in row.items()}
        item["row_id"] = int(index)
        items.append(item)
    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": int(total),
        "pages": int((total + page_size - 1) // page_size),
    }


def get_column_stats(column: str) -> dict[str, Any]:
    frame = get_raw_dataset()
    if column not in frame.columns:
        raise KeyError(column)
    series = frame[column]
    result: dict[str, Any] = {
        "column": column,
        "dtype": str(series.dtype),
        "missing": int(series.isna().sum()),
        "unique": int(series.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(series):
        result.update(
            {
                "mean": json_safe(series.mean()),
                "median": json_safe(series.median()),
                "min": json_safe(series.min()),
                "max": json_safe(series.max()),
                "std": json_safe(series.std()),
                "histogram": _numeric_histogram(series.dropna()),
            }
        )
    else:
        counts = series.fillna("(Missing)").astype(str).value_counts().head(20)
        result["top_values"] = [
            {"value": str(value), "count": int(count)}
            for value, count in counts.items()
        ]
    return result


def _numeric_histogram(values: pd.Series, bins: int = 16) -> list[dict[str, Any]]:
    if values.empty:
        return []
    counts, edges = np.histogram(values.to_numpy(dtype=float), bins=bins)
    return [
        {
            "start": float(edges[index]),
            "end": float(edges[index + 1]),
            "count": int(count),
        }
        for index, count in enumerate(counts)
    ]
