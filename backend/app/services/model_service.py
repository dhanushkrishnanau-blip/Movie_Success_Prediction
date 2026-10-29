import json
from functools import lru_cache
from typing import Any

import joblib

from app.config import MODEL_METADATA_PATH, MODEL_PATH

NOTEBOOK_COMPARISON = [
    {"model": "Oversampled HistGradientBoosting", "accuracy": 78.0, "macro_precision": 48.65, "balanced_accuracy": 49.66, "macro_f1": 49.12},
    {"model": "XGBoost", "accuracy": 79.3, "macro_precision": 50.11, "balanced_accuracy": 47.67, "macro_f1": 48.48},
    {"model": "HistGradientBoosting", "accuracy": 78.9, "macro_precision": 49.56, "balanced_accuracy": 47.85, "macro_f1": 48.46},
    {"model": "Balanced HistGradientBoosting", "accuracy": 76.4, "macro_precision": 47.54, "balanced_accuracy": 48.99, "macro_f1": 48.16},
    {"model": "Random Forest", "accuracy": 76.2, "macro_precision": 47.37, "balanced_accuracy": 48.68, "macro_f1": 47.94},
    {"model": "KNN", "accuracy": 72.9, "macro_precision": 44.40, "balanced_accuracy": 44.75, "macro_f1": 44.57},
    {"model": "Logistic Regression", "accuracy": 63.5, "macro_precision": 45.76, "balanced_accuracy": 45.19, "macro_f1": 43.12},
    {"model": "Decision Tree", "accuracy": 69.9, "macro_precision": 42.73, "balanced_accuracy": 43.49, "macro_f1": 43.04},
]


@lru_cache(maxsize=1)
def load_artifact() -> dict[str, Any] | None:
    if not MODEL_PATH.is_file() or not MODEL_METADATA_PATH.is_file():
        return None
    with MODEL_METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)
    return {"pipeline": joblib.load(MODEL_PATH), "metadata": metadata}


def artifact_status() -> dict[str, bool]:
    available = MODEL_PATH.is_file() and MODEL_METADATA_PATH.is_file()
    return {"available": available, "artifact": available}


def comparison() -> dict[str, Any]:
    artifact = load_artifact()
    deployed_experiment = None
    if artifact is not None:
        metadata = artifact["metadata"]
        deployed_experiment = {
            "model": "Random Forest · reduced inputs",
            "accuracy": metadata["metrics_percent"]["accuracy_percent"],
            "macro_precision": metadata["metrics_percent"]["macro_precision_percent"],
            "balanced_accuracy": metadata["metrics_percent"]["balanced_accuracy_percent"],
            "macro_f1": metadata["metrics_percent"]["macro_f1_percent"],
            "baseline_metrics": metadata["baseline_metrics_percent"],
            "baseline_source": metadata["baseline_source"],
            "metric_change_percentage_points": metadata[
                "metric_change_percentage_points"
            ],
            "flop_recall_percent": round(
                metadata["classification_report"]["Flop"]["recall"] * 100, 2
            ),
        }
    return {
        "source": "Saved notebook holdout comparison; metric values are percentages.",
        "split": "80/20 stratified train/test split, random_state=42.",
        "items": NOTEBOOK_COMPARISON,
        "deployed_experiment": deployed_experiment,
        "cross_validation": {
            "model": "HistGradientBoosting",
            "method": "5-fold stratified cross-validation on training data only",
            "accuracy_mean_percent": 79.34,
            "accuracy_std_percent": 0.91,
            "balanced_accuracy_mean_percent": 48.47,
            "balanced_accuracy_std_percent": 0.53,
            "macro_f1_mean_percent": 49.04,
            "macro_f1_std_percent": 0.57,
        },
    }


def confusion_matrix_data() -> dict[str, Any]:
    metadata = _require_metadata()
    return {
        "labels": metadata["labels"],
        "matrix": metadata["confusion_matrix"],
        "source": metadata["evaluation_source"],
    }


def classification_report_data() -> dict[str, Any]:
    metadata = _require_metadata()
    return {
        "labels": metadata["labels"],
        "report": metadata["classification_report"],
        "source": metadata["evaluation_source"],
    }


def feature_importance_data(limit: int = 20) -> dict[str, Any]:
    artifact = load_artifact()
    if artifact is None:
        return {"available": False, "items": []}
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    scores = classifier.feature_importances_
    items = sorted(
        (
            {"feature": str(name), "importance": float(score)}
            for name, score in zip(names, scores, strict=True)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return {"available": True, "items": items[:limit]}


def predict(features: dict[str, Any]) -> dict[str, Any]:
    artifact = load_artifact()
    if artifact is None:
        raise RuntimeError(
            "The Random Forest artifact is not available. Run "
            "python -m app.export_model from the backend directory."
        )
    pipeline = artifact["pipeline"]
    frame = __import__("pandas").DataFrame([features])
    encoded_category = int(pipeline.predict(frame)[0])
    mapping = artifact["metadata"]["target_mapping"]
    category = mapping[str(encoded_category)]
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        raw_probabilities = pipeline.predict_proba(frame)[0]
        classes = pipeline.named_steps["classifier"].classes_
        probabilities = {
            mapping[str(int(label))]: float(probability)
            for label, probability in zip(classes, raw_probabilities, strict=True)
        }
    return {
        "predicted_category": category,
        "model_name": "Random Forest · reduced inputs",
        "probabilities": probabilities,
        "note": (
            "Categories are derived from IMDb scores, not box-office revenue. "
            "Probabilities are model outputs and have not been calibrated. "
            "Unseen director or actor names may provide little individual signal."
        ),
    }


def _require_metadata() -> dict[str, Any]:
    artifact = load_artifact()
    if artifact is None:
        raise RuntimeError(
            "The Random Forest evaluation artifact is missing. "
            "Run python -m app.export_model from the backend directory."
        )
    return artifact["metadata"]
