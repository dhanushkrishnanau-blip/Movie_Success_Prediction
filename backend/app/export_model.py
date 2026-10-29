"""Reproduce the notebook's Random Forest pipeline as a deployable artifact."""

import json
from datetime import date

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from app.config import ARTIFACT_DIR, DATASET_PATH, MODEL_METADATA_PATH, MODEL_PATH
from app.services.dataset_service import (
    CLASS_LABELS,
    FEATURE_COLUMNS,
    PERSON_NAME_COLUMNS,
    TARGET,
)

NOTEBOOK_BASELINE_METRICS = {
    "accuracy_percent": 76.2,
    "macro_precision_percent": 47.37,
    "balanced_accuracy_percent": 48.68,
    "macro_f1_percent": 47.94,
}


def export_model() -> dict:
    raw = pd.read_csv(DATASET_PATH)
    frame = raw.drop_duplicates().dropna(subset=["imdb_score"]).copy()
    frame[TARGET] = pd.cut(
        frame["imdb_score"],
        bins=[0, 3, 6, 10],
        labels=CLASS_LABELS,
        right=False,
        include_lowest=True,
    )
    frame = frame.dropna(subset=[TARGET])
    x = frame[FEATURE_COLUMNS]
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame[TARGET])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    numeric_columns = x_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column
        for column in x_train.select_dtypes(exclude="number").columns
        if column not in PERSON_NAME_COLUMNS
    ]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    person_name_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
            ("person_names", person_name_pipeline, PERSON_NAME_COLUMNS),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=2,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    prediction = pipeline.predict(x_test)
    previous_metadata = None
    baseline_predictions = None
    if MODEL_PATH.is_file() and MODEL_METADATA_PATH.is_file():
        with MODEL_METADATA_PATH.open(encoding="utf-8") as file:
            previous_metadata = json.load(file)
        previous_features = previous_metadata.get("features", [])
        if previous_features and set(previous_features).issubset(frame.columns):
            previous_pipeline = joblib.load(MODEL_PATH)
            baseline_features = frame.loc[x_test.index, previous_features]
            baseline_predictions = previous_pipeline.predict(baseline_features)
    class_order = ["Flop", "Average", "Hit"]
    ordered_encoded_labels = encoder.transform(class_order).tolist()
    matrix = confusion_matrix(y_test, prediction, labels=ordered_encoded_labels)
    report = classification_report(
        y_test,
        prediction,
        labels=ordered_encoded_labels,
        target_names=class_order,
        output_dict=True,
        zero_division=0,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    comparison = {
        "accuracy_percent": round(accuracy_score(y_test, prediction) * 100, 2),
        "macro_precision_percent": round(
            precision_score(y_test, prediction, average="macro", zero_division=0)
            * 100,
            2,
        ),
        "balanced_accuracy_percent": round(
            balanced_accuracy_score(y_test, prediction) * 100, 2
        ),
        "macro_f1_percent": round(
            f1_score(y_test, prediction, average="macro", zero_division=0) * 100,
            2,
        ),
    }
    if baseline_predictions is not None:
        baseline_metrics = _metrics(y_test, baseline_predictions)
        baseline_source = (
            "Previously deployed Random Forest using Facebook-like counts and "
            "other metadata, evaluated on the same stratified holdout."
        )
    else:
        baseline_metrics = NOTEBOOK_BASELINE_METRICS
        baseline_source = (
            "Saved notebook Random Forest holdout metrics; the earlier fitted "
            "pipeline was unavailable for a fresh paired evaluation."
        )
    metadata = {
        "model_type": "RandomForestClassifier",
        "artifact_version": 3,
        "training_source": (
            "Random Forest trained on movie duration, genre, director and actor "
            "names, budget, release year, language, and country."
        ),
        "export_date": date.today().isoformat(),
        "dataset": DATASET_PATH.name,
        "features": FEATURE_COLUMNS,
        "person_name_features": PERSON_NAME_COLUMNS,
        "person_name_min_frequency": 2,
        "target": TARGET,
        "target_mapping": {
            str(index): str(label) for index, label in enumerate(encoder.classes_)
        },
        "labels": class_order,
        "evaluation_source": (
            "Fresh holdout evaluation from the notebook's cleaning, split, and "
            "Random Forest configuration reproduced by this export script."
        ),
        "test_size": 0.2,
        "random_state": 42,
        "metrics_percent": comparison,
        "baseline_metrics_percent": baseline_metrics,
        "baseline_source": baseline_source,
        "metric_change_percentage_points": {
            key: round(comparison[key] - baseline_metrics[key], 2)
            for key in comparison
        },
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "class_distribution": {
            str(label): int(count)
            for label, count in frame[TARGET].value_counts().items()
        },
        "training_class_distribution": _class_counts(y_train, encoder),
        "test_class_distribution": _class_counts(y_test, encoder),
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _class_counts(values, encoder: LabelEncoder) -> dict[str, int]:
    import numpy as np

    labels, counts = np.unique(values, return_counts=True)
    return {
        str(encoder.inverse_transform([int(label)])[0]): int(count)
        for label, count in zip(labels, counts, strict=True)
    }


def _metrics(y_true, prediction) -> dict[str, float]:
    return {
        "accuracy_percent": round(accuracy_score(y_true, prediction) * 100, 2),
        "macro_precision_percent": round(
            precision_score(y_true, prediction, average="macro", zero_division=0) * 100,
            2,
        ),
        "balanced_accuracy_percent": round(
            balanced_accuracy_score(y_true, prediction) * 100, 2
        ),
        "macro_f1_percent": round(
            f1_score(y_true, prediction, average="macro", zero_division=0) * 100,
            2,
        ),
    }


if __name__ == "__main__":
    result = export_model()
    print(json.dumps(result["metrics_percent"], indent=2))
