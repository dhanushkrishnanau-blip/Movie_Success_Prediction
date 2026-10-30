import re
from typing import Any

import pandas as pd

from app.config import MIN_GENRE_MOVIES
from app.services import analytics_service, dataset_service, model_service

FEATURE_AVAILABILITY = [
    {
        "feature": "director_name",
        "status": "review",
        "note": "Use only a director publicly attached to the project before release.",
    },
    {
        "feature": "actor_1_name, actor_2_name, actor_3_name",
        "status": "review",
        "note": "Cast can change; use only cast announced by the prediction date.",
    },
    {
        "feature": "genres",
        "status": "review",
        "note": "Genre labels may be assigned retrospectively; confirm the intended pre-release source.",
    },
    {
        "feature": "duration",
        "status": "review",
        "note": "Final runtime may not be set or publicly known before release.",
    },
    {
        "feature": "budget",
        "status": "review",
        "note": "Production budgets are often estimated or not publicly available before release.",
    },
    {
        "feature": "title_year",
        "status": "review",
        "note": "Use the planned release year known at the prediction date; release plans can change.",
    },
    {
        "feature": "language, country",
        "status": "review",
        "note": "Confirm the relevant language and country definitions and that they are known at prediction time.",
    },
]
TOP_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

def explain_prediction(features: dict[str, Any]) -> dict[str, Any]:
    prediction = model_service.predict(features)
    effects = _feature_sensitivity(features, prediction)
    historical = historical_comparison(features)
    return {
        "prediction": prediction,
        "feature_effects": effects,
        "historical_comparison": historical,
        "feature_availability": FEATURE_AVAILABILITY,
        "explanation_method": (
            "One-feature-at-a-time sensitivity: each input is replaced by its "
            "historical dataset median or most common value and the model probability "
            "is compared. This is local to this prediction, is not additive SHAP, "
            "and does not show causation."
        ),
    }


def _feature_sensitivity(
    features: dict[str, Any], prediction: dict[str, Any]
) -> list[dict[str, Any]]:
    artifact = model_service.load_artifact()
    if artifact is None:
        raise RuntimeError("The Random Forest artifact is not available.")
    pipeline = artifact["pipeline"]
    if not hasattr(pipeline, "predict_proba"):
        raise RuntimeError("The saved Random Forest does not provide class probabilities.")

    columns = artifact["metadata"]["features"]
    original = pd.DataFrame([{column: features.get(column) for column in columns}])
    target_category = prediction["predicted_category"]
    target_probability = prediction["probabilities"].get(target_category)
    if target_probability is None:
        raise RuntimeError("The Random Forest did not return the predicted class probability.")

    historical = dataset_service.get_clean_dataset()
    effects = []
    for column in columns:
        values = historical[column].dropna()
        if values.empty:
            continue
        if pd.api.types.is_numeric_dtype(values):
            reference = float(values.median())
        else:
            reference = str(values.mode().iloc[0])

        changed = original.copy()
        changed.loc[0, column] = reference
        encoded_probabilities = pipeline.predict_proba(changed)[0]
        classes = pipeline.named_steps["classifier"].classes_
        mapping = artifact["metadata"]["target_mapping"]
        reference_probability = next(
            float(probability)
            for encoded, probability in zip(classes, encoded_probabilities, strict=True)
            if mapping[str(int(encoded))] == target_category
        )
        change = (target_probability - reference_probability) * 100
        effects.append(
            {
                "feature": column,
                "input_value": dataset_service.json_safe(features.get(column)),
                "reference_value": dataset_service.json_safe(reference),
                "predicted_probability_change_percentage_points": round(change, 2),
            }
        )
    return sorted(
        effects,
        key=lambda item: abs(
            item["predicted_probability_change_percentage_points"]
        ),
        reverse=True,
    )


def historical_comparison(features: dict[str, Any]) -> dict[str, Any]:
    frame = dataset_service.get_clean_dataset()
    criteria = []
    requested_genres = {
        genre.strip().casefold()
        for genre in str(features.get("genres") or "").split("|")
        if genre.strip()
    }
    if requested_genres:
        genre_match = frame["genres"].fillna("").map(
            lambda value: bool(
                requested_genres.intersection(
                    genre.strip().casefold() for genre in str(value).split("|")
                )
            )
        )
        frame = frame[genre_match]
        criteria.append("shares at least one supplied genre")

    for column in ("country", "language"):
        value = str(features.get(column) or "").strip()
        if value:
            frame = frame[
                frame[column]
                .fillna("")
                .astype(str)
                .str.casefold()
                .eq(value.casefold())
            ]
            criteria.append(f"same {column}")

    if not requested_genres:
        criteria.append("all eligible dataset movies (no genre supplied)")

    total = int(len(frame))
    if total < 5:
        return {
            "available": False,
            "matched_movies": total,
            "criteria": criteria,
            "message": "Fewer than five comparable dataset movies; no summary is shown.",
        }

    category_counts = frame["Classify"].value_counts()
    return {
        "available": True,
        "matched_movies": total,
        "criteria": criteria,
        "mean_imdb_score": round(float(frame["imdb_score"].mean()), 2),
        "median_imdb_score": round(float(frame["imdb_score"].median()), 2),
        "category_percentages": {
            label: round(float(category_counts.get(label, 0) / total * 100), 2)
            for label in dataset_service.CLASS_LABELS
        },
        "interpretation_note": (
            "These are historical dataset associations, not causes or a guarantee "
            "for a new movie."
        ),
    }


def analyze_dataset_question(question: str) -> dict[str, Any]:
    raw = dataset_service.get_raw_dataset()
    normalized = question.casefold()
    if "genre" in normalized:
        missing_columns = {"genres", "imdb_score"} - set(raw.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            return _not_enough(
                f"The dataset is missing required genre-analysis column(s): {missing}."
            )
        clean = dataset_service.get_clean_dataset()
        return _genre_analysis(clean, normalized)

    clean = dataset_service.get_clean_dataset()
    if "director" in normalized:
        return _group_analysis(clean, "director_name", normalized)

    for column in ("country", "language", "content_rating", "color"):
        if column.replace("_", " ") in normalized or column in normalized:
            return _group_analysis(clean, column, normalized)

    matched_numeric = next(
        (
            column
            for column in (
                "imdb_score",
                "budget",
                "duration",
                "title_year",
                "gross",
                "num_voted_users",
                "num_user_for_reviews",
                "num_critic_for_reviews",
            )
            if column.replace("_", " ") in normalized or column in normalized
        ),
        None,
    )
    if matched_numeric:
        return _numeric_analysis(clean, matched_numeric)

    if "score" in normalized or "rating" in normalized:
        return _numeric_analysis(clean, "imdb_score")

    if any(word in normalized for word in ("category", "categories", "flop", "hit")):
        return {
            "supported": True,
            "operation": "category_distribution",
            "statistics": analytics_service.category_distribution(),
            "source": "Cleaned dataset; IMDb-score-derived categories.",
        }

    for column in raw.columns:
        column_name = str(column).casefold().replace("_", " ")
        if not re.search(rf"\b{re.escape(column_name)}\b", normalized):
            continue
        if pd.api.types.is_numeric_dtype(raw[column]):
            return _numeric_analysis(clean, str(column))
        counts = raw[column].dropna().astype(str).value_counts().head(15)
        return {
            "supported": True,
            "operation": "categorical_value_counts",
            "column": str(column),
            "statistics": [
                {"value": str(value), "count": int(count)}
                for value, count in counts.items()
            ],
            "missing_values": int(raw[column].isna().sum()),
            "source": f"Raw dataset column `{column}`; top observed values only.",
        }

    if any(word in normalized for word in ("distribution", "overall", "dataset", "movies")):
        return {
            "supported": True,
            "operation": "score_distribution",
            "statistics": analytics_service.score_distribution(),
            "source": "Cleaned dataset; IMDb score is a post-release outcome.",
        }

    return {
        "supported": False,
        "operation": None,
        "statistics": None,
        "source": "Project movie_metadata.csv",
        "message": (
            "I could not map that question to a supported dataset analysis. "
            "Try asking about genres, IMDb score, budget, duration, release year, "
            "directors, countries, languages, or categories."
        ),
    }


def _genre_analysis(frame: pd.DataFrame, question: str) -> dict[str, Any]:
    missing_columns = {"genres", "imdb_score"} - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        return _not_enough(
            f"The dataset is missing required genre-analysis column(s): {missing}."
        )

    expanded = frame[["genres", "imdb_score"]].copy()
    expanded["imdb_score"] = pd.to_numeric(expanded["imdb_score"], errors="coerce")
    expanded = expanded.dropna(subset=["genres", "imdb_score"])
    expanded["genre"] = expanded["genres"].astype(str).str.split("|")
    expanded = expanded.explode("genre")
    expanded["genre"] = expanded["genre"].astype(str).str.strip()
    expanded = expanded[expanded["genre"].ne("")]
    available = sorted(expanded["genre"].unique())
    matched = [
        genre
        for genre in available
        if re.search(rf"\b{re.escape(genre.casefold())}\b", question)
    ]
    if matched:
        expanded = expanded[expanded["genre"].isin(matched)]

    grouped = expanded.groupby("genre").agg(
        movies=("imdb_score", "size"),
        mean_imdb_score=("imdb_score", "mean"),
        median_imdb_score=("imdb_score", "median"),
    )
    excluded_genres = int((grouped["movies"] < MIN_GENRE_MOVIES).sum())
    grouped = grouped[grouped["movies"] >= MIN_GENRE_MOVIES]

    if any(term in question for term in ("score", "rating", "average", "highest", "best")):
        grouped = grouped.sort_values(
            ["mean_imdb_score", "movies"], ascending=[False, False]
        )
    else:
        grouped = grouped.sort_values(
            ["movies", "mean_imdb_score"], ascending=[False, False]
        )

    requested_count = re.search(
        r"\btop\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        question,
    )
    limit = (
        min(
            int(requested_count.group(1))
            if requested_count.group(1).isdigit()
            else TOP_COUNT_WORDS[requested_count.group(1)],
            25,
        )
        if requested_count
        else 5
    )
    eligible_count = len(grouped)
    if eligible_count < limit:
        if eligible_count:
            result_message = (
                f"Only {eligible_count} genre(s) met the minimum of "
                f"{MIN_GENRE_MOVIES} movies; all eligible genres are shown."
            )
        else:
            result_message = (
                f"No genre met the minimum of {MIN_GENRE_MOVIES} movies; "
                "no ranked results are available. Lower "
                "CINESCOPE_MIN_GENRE_MOVIES to include smaller samples."
            )
    else:
        result_message = None

    stats = [
        {
            "rank": rank,
            "genre": str(genre),
            "movies": int(row["movies"]),
            "mean_imdb_score": round(float(row["mean_imdb_score"]), 2),
            "median_imdb_score": round(float(row["median_imdb_score"]), 2),
        }
        for rank, (genre, row) in enumerate(grouped.head(limit).iterrows(), start=1)
    ]
    return {
        "supported": True,
        "operation": "genre_performance",
        "statistics": stats,
        "minimum_movies_per_genre": MIN_GENRE_MOVIES,
        "excluded_genres_below_minimum": excluded_genres,
        "eligible_genres": eligible_count,
        "requested_count": limit,
        "message": result_message,
        "source": (
            "Cleaned dataset; missing genres and IMDb scores are excluded; a movie "
            "may count under each listed genre. Each reported genre has at least "
            f"{MIN_GENRE_MOVIES} scored movies."
        ),
    }


def _group_analysis(
    clean: pd.DataFrame, column: str, question: str
) -> dict[str, Any]:
    frame = clean[[column, "imdb_score", "Classify"]].dropna(subset=[column]).copy()
    if column == "genres":
        return _genre_analysis(clean, question)

    candidates = frame[column].astype(str).unique()
    named_match = next(
        (
            value
            for value in candidates
            if re.search(rf"\b{re.escape(value.casefold())}\b", question)
        ),
        None,
    )
    if named_match:
        frame = frame[frame[column].astype(str).eq(named_match)]
    grouped = frame.groupby(column).agg(
        movies=("imdb_score", "size"),
        mean_imdb_score=("imdb_score", "mean"),
    ).query("movies >= 3")
    if any(term in question for term in ("score", "rating", "average", "highest", "best")):
        grouped = grouped.sort_values(
            ["mean_imdb_score", "movies"], ascending=[False, False]
        )
    else:
        grouped = grouped.sort_values(
            ["movies", "mean_imdb_score"], ascending=[False, False]
        )
    grouped = grouped.head(15)
    if grouped.empty:
        return _not_enough(f"Not enough scored rows for {column} in this dataset.")
    stats = [
        {
            "value": str(value),
            "movies": int(row["movies"]),
            "mean_imdb_score": round(float(row["mean_imdb_score"]), 2),
        }
        for value, row in grouped.iterrows()
    ]
    return {
        "supported": True,
        "operation": "grouped_column_performance",
        "column": column,
        "statistics": stats,
        "source": "Cleaned dataset; groups need at least three scored movies.",
    }


def _numeric_analysis(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return _not_enough(f"No numeric values for {column} are available.")
    return {
        "supported": True,
        "operation": "numeric_summary",
        "column": column,
        "statistics": {
            "count": int(values.size),
            "mean": round(float(values.mean()), 2),
            "median": round(float(values.median()), 2),
            "minimum": round(float(values.min()), 2),
            "maximum": round(float(values.max()), 2),
        },
        "source": (
            f"Cleaned dataset column `{column}`. Historical descriptive statistics; "
            "not a causal result."
        ),
    }


def _not_enough(message: str) -> dict[str, Any]:
    return {
        "supported": False,
        "operation": None,
        "statistics": None,
        "source": "Project movie_metadata.csv",
        "message": message,
    }
