import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import DATASET_PATH
from app.schemas import PredictionInput, PredictionResponse, ResearchChatRequest
from app.services import (
    analytics_service,
    dataset_service,
    gemini_service,
    model_service,
    researcher_service,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cinescope")

app = FastAPI(title="CineScope API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_dataset():
    if not DATASET_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Dataset unavailable. Expected {DATASET_PATH.name} in ASSETS.",
        )


def _model_call(function, *args):
    try:
        return function(*args)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception:
        logger.exception("Model operation failed")
        raise HTTPException(status_code=500, detail="Model operation failed.")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": "available",
        "dataset": DATASET_PATH.is_file(),
        "model": model_service.artifact_status()["available"],
    }


@app.get("/api/overview")
def get_overview() -> dict[str, Any]:
    _require_dataset()
    summary = dataset_service.overview()
    summary["categories"] = analytics_service.category_distribution()
    models = model_service.comparison()
    summary["model_comparison"] = models["items"]
    summary["model_experiment"] = models["deployed_experiment"]
    return summary


@app.get("/api/dataset/schema")
def get_schema() -> dict[str, Any]:
    _require_dataset()
    return {"columns": dataset_service.column_schema()}


@app.get("/api/dataset/name-suggestions")
def get_name_suggestions(
    query: str = Query(..., min_length=2, max_length=100),
    limit: int = Query(10, ge=1, le=20),
) -> dict[str, list[str]]:
    _require_dataset()
    return {"items": dataset_service.suggest_person_names(query, limit)}


@app.get("/api/dataset/export")
def export_dataset() -> FileResponse:
    _require_dataset()
    return FileResponse(
        DATASET_PATH,
        media_type="text/csv",
        filename="movie_metadata.csv",
    )


@app.get("/api/dataset/rows")
def get_rows(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    search: str = "",
    sort_by: str = "movie_title",
    direction: str = Query("asc", pattern="^(asc|desc)$"),
    genre: str = "",
    country: str = "",
    language: str = "",
    content_rating: str = "",
    color: str = "",
    min_score: float | None = Query(None, ge=0, le=10),
    max_score: float | None = Query(None, ge=0, le=10),
    min_year: int | None = Query(None, ge=1800, le=2200),
    max_year: int | None = Query(None, ge=1800, le=2200),
) -> dict[str, Any]:
    _require_dataset()
    return dataset_service.table_rows(
        page,
        page_size,
        search,
        sort_by,
        direction,
        {
            "genres": genre,
            "country": country,
            "language": language,
            "content_rating": content_rating,
            "color": color,
        },
        min_score,
        max_score,
        min_year,
        max_year,
    )


@app.get("/api/dataset/column-stats/{column_name}")
def get_column_stats(column_name: str) -> dict[str, Any]:
    _require_dataset()
    try:
        return dataset_service.get_column_stats(column_name)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Column not found.") from error


@app.get("/api/analytics/score-distribution")
def get_score_distribution(bins: int = Query(20, ge=5, le=50)) -> dict[str, Any]:
    _require_dataset()
    return analytics_service.score_distribution(bins)


@app.get("/api/analytics/categories")
def get_categories() -> dict[str, Any]:
    _require_dataset()
    return {
        "boundaries": [
            {"category": "Flop", "minimum": 0, "maximum_exclusive": 3},
            {"category": "Average", "minimum": 3, "maximum_exclusive": 6},
            {"category": "Hit", "minimum": 6, "maximum_exclusive": 10},
        ],
        "items": analytics_service.category_distribution(),
        "score_summary": analytics_service.category_score_summary(),
    }


@app.get("/api/analytics/genres")
def get_genres(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    _require_dataset()
    return {"items": analytics_service.genre_statistics(limit)}


@app.get("/api/analytics/correlation")
def get_correlation() -> dict[str, Any]:
    _require_dataset()
    return analytics_service.correlation_matrix()


@app.get("/api/analytics/countries")
def get_countries(limit: int = Query(12, ge=1, le=100)) -> dict[str, Any]:
    _require_dataset()
    return {"items": analytics_service.country_distribution(limit)}


@app.get("/api/analytics/relationships")
def get_relationships() -> dict[str, Any]:
    _require_dataset()
    return analytics_service.relationships()


@app.get("/api/models/comparison")
def get_model_comparison() -> dict[str, Any]:
    return model_service.comparison()


@app.get("/api/models/confusion-matrix")
def get_confusion_matrix() -> dict[str, Any]:
    return _model_call(model_service.confusion_matrix_data)


@app.get("/api/models/classification-report")
def get_classification_report() -> dict[str, Any]:
    return _model_call(model_service.classification_report_data)


@app.get("/api/models/feature-importance")
def get_feature_importance(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    return _model_call(model_service.feature_importance_data, limit)


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: PredictionInput) -> dict[str, Any]:
    _require_dataset()
    return _model_call(model_service.predict, payload.model_dump())


@app.post("/api/researcher/explain")
def explain_prediction(payload: PredictionInput) -> dict[str, Any]:
    _require_dataset()
    result = _model_call(researcher_service.explain_prediction, payload.model_dump())
    verified_evidence = {
        "predicted_category": result["prediction"]["predicted_category"],
        "probabilities": result["prediction"]["probabilities"],
        "feature_effects": result["feature_effects"],
        "historical_comparison": result["historical_comparison"],
        "explanation_method": result["explanation_method"],
    }
    try:
        explanation = gemini_service.explain_prediction(verified_evidence)
    except gemini_service.GeminiRateLimitError:
        explanation = None
        gemini_notice = "Gemini rate limit reached. The model result is still available."
    except gemini_service.GeminiUnavailableError as error:
        explanation = None
        gemini_notice = str(error)
    else:
        gemini_notice = None

    return {
        **result,
        "gemini_explanation": explanation,
        "gemini_notice": gemini_notice,
    }


@app.post("/api/researcher/chat")
def researcher_chat(payload: ResearchChatRequest) -> dict[str, Any]:
    history = [message.model_dump() for message in payload.history]
    try:
        if payload.mode == "general":
            answer = gemini_service.answer_general_question(
                payload.question, history
            )
            return {
                "mode": payload.mode,
                "answer": answer,
                "evidence": None,
                "notice": "General movie information is not live-verified.",
            }

        _require_dataset()
        analysis = researcher_service.analyze_dataset_question(payload.question)
        try:
            answer = gemini_service.explain_dataset_results(
                payload.question, analysis, history
            )
            notice = None
        except gemini_service.GeminiRateLimitError as error:
            answer = "Gemini could not explain the results; the Python evidence is available below."
            notice = str(error)
        except gemini_service.GeminiUnavailableError as error:
            answer = "Gemini could not explain the results; the Python evidence is available below."
            notice = str(error)
        return {
            "mode": payload.mode,
            "answer": answer,
            "evidence": analysis,
            "notice": notice,
        }
    except gemini_service.GeminiRateLimitError as error:
        raise HTTPException(status_code=429, detail=str(error)) from error
    except gemini_service.GeminiUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/movies")
def get_movies(
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
    search: str = "",
    genre: str = "",
    country: str = "",
    language: str = "",
    content_rating: str = "",
    min_score: float | None = Query(None, ge=0, le=10),
    max_score: float | None = Query(None, ge=0, le=10),
    min_year: int | None = Query(None, ge=1800, le=2200),
    max_year: int | None = Query(None, ge=1800, le=2200),
    sort_by: str = Query("imdb_score"),
    direction: str = Query("desc", pattern="^(asc|desc)$"),
) -> dict[str, Any]:
    _require_dataset()
    return dataset_service.table_rows(
        page,
        page_size,
        search,
        sort_by,
        direction,
        {
            "genres": genre,
            "country": country,
            "language": language,
            "content_rating": content_rating,
        },
        min_score,
        max_score,
        min_year,
        max_year,
    )


@app.get("/api/movies/{movie_id}")
def get_movie(movie_id: int) -> dict[str, Any]:
    _require_dataset()
    raw = dataset_service.get_raw_dataset()
    if movie_id not in raw.index:
        raise HTTPException(status_code=404, detail="Movie not found.")
    record = {
        key: dataset_service.json_safe(value)
        for key, value in raw.loc[movie_id].items()
    }
    record["row_id"] = movie_id
    related = raw[
        (raw.index != movie_id)
        & raw["genres"].fillna("").eq(record.get("genres", ""))
    ]
    record["related"] = [
        {
            "row_id": int(index),
            "movie_title": dataset_service.json_safe(row.get("movie_title")),
            "imdb_score": dataset_service.json_safe(row.get("imdb_score")),
        }
        for index, row in related.head(4).iterrows()
    ]
    return record
