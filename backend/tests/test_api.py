from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import dataset_service, gemini_service, model_service, researcher_service
import app.main as main_module

client = TestClient(app)


def test_health_and_dataset_schema():
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["dataset"] is True
    schema = client.get("/api/dataset/schema")
    assert schema.status_code == 200
    assert any(column["name"] == "imdb_score" for column in schema.json()["columns"])


def test_dataset_pagination_search_and_filter():
    response = client.get("/api/dataset/rows?page=1&page_size=5")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 5
    assert response.json()["total"] == 5043
    filtered = client.get("/api/movies?search=Avatar&page_size=5")
    assert filtered.status_code == 200
    assert any("Avatar" in (item["movie_title"] or "") for item in filtered.json()["items"])


def test_dataset_search_matches_director_name():
    director = dataset_service.get_raw_dataset()["director_name"].dropna().iloc[0]
    response = client.get(
        "/api/dataset/rows",
        params={"search": director, "page_size": 100},
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert items
    assert all(
        item["director_name"]
        and director.casefold() in item["director_name"].casefold()
        for item in items
    )


def test_person_name_suggestions_match_partial_names():
    response = client.get("/api/dataset/name-suggestions", params={"query": "nol"})
    assert response.status_code == 200
    assert "Christopher Nolan" in response.json()["items"]

    invalid = client.get("/api/dataset/name-suggestions", params={"query": "n"})
    assert invalid.status_code == 422


def test_actual_analytics_responses():
    categories = client.get("/api/analytics/categories").json()
    assert {item["category"]: item["count"] for item in categories["items"]} == {
        "Flop": 41,
        "Average": 1401,
        "Hit": 3556,
    }
    scores = client.get("/api/analytics/score-distribution").json()
    assert scores["count"] == 4998
    assert sum(item["count"] for item in scores["bins"]) == 4998
    assert client.get("/api/analytics/genres").status_code == 200
    assert categories["boundaries"][-1] == {
        "category": "Hit",
        "minimum": 6,
        "maximum_exclusive": 10,
    }


def test_models_comparison_is_not_mocked():
    comparison = client.get("/api/models/comparison").json()
    random_forest = next(
        item for item in comparison["items"] if item["model"] == "Random Forest"
    )
    assert random_forest["accuracy"] == 74.6
    assert random_forest["macro_precision"] == 45.69
    assert random_forest["balanced_accuracy"] == 45.83
    assert random_forest["macro_f1"] == 45.76
    assert comparison["source"].startswith("Saved notebook")
    assert all(
        "HistGradientBoosting" not in item["model"] for item in comparison["items"]
    )
    assert "cross_validation" not in comparison


def test_model_artifact_evaluation_and_prediction():
    health = client.get("/api/health").json()
    assert health["model"] is True
    matrix = client.get("/api/models/confusion-matrix")
    assert matrix.status_code == 200
    assert matrix.json()["labels"] == ["Flop", "Average", "Hit"]
    assert [sum(row) for row in matrix.json()["matrix"]] == [8, 280, 712]
    report = client.get("/api/models/classification-report").json()
    experiment = client.get("/api/models/comparison").json()["deployed_experiment"]
    assert experiment["flop_recall_percent"] == round(
        report["report"]["Flop"]["recall"] * 100, 2
    )
    prediction = client.post("/api/predict", json={})
    assert prediction.status_code == 200
    result = prediction.json()
    assert result["predicted_category"] in {"Flop", "Average", "Hit"}
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-8
    assert "names" in result["note"]
    assert client.get("/api/models/feature-importance").json()["available"] is True
    overview = client.get("/api/overview").json()
    assert overview["model_experiment"]["accuracy"] > 0
    assert overview["model_experiment"]["flop_recall_percent"] >= 0


def test_model_schema_uses_reduced_movie_inputs():
    from app.services.dataset_service import FEATURE_COLUMNS

    metadata = model_service.load_artifact()["metadata"]
    expected = {
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
    }
    excluded = {
        "director_facebook_likes",
        "actor_1_facebook_likes",
        "actor_2_facebook_likes",
        "actor_3_facebook_likes",
        "cast_total_facebook_likes",
        "movie_facebook_likes",
        "color",
    }
    assert set(FEATURE_COLUMNS) == expected
    assert set(metadata["features"]) == expected
    assert excluded.isdisjoint(metadata["features"])
    assert metadata["artifact_version"] == 3
    assert metadata["baseline_source"].startswith("Previously deployed")
    response = client.post(
        "/api/predict",
        json={
            "director_name": "Christopher Nolan",
            "actor_1_name": "Christian Bale",
            "actor_2_name": "Heath Ledger",
            "actor_3_name": "Aaron Eckhart",
        },
    )
    assert response.status_code == 200
    assert client.post("/api/predict", json={"movie_facebook_likes": 100}).status_code == 422


def test_dataset_score_and_year_filters():
    response = client.get(
        "/api/dataset/rows?min_score=8&min_year=2000&page_size=100"
    )
    assert response.status_code == 200
    assert all(
        row["imdb_score"] is None or row["imdb_score"] >= 8
        for row in response.json()["items"]
    )


def test_dataset_export_returns_original_csv():
    response = client.get("/api/dataset/export")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert response.content.startswith(b"color,director_name")


def test_missing_dataset_is_reported(monkeypatch, tmp_path):
    unavailable = tmp_path / "missing.csv"
    monkeypatch.setattr(main_module, "DATASET_PATH", unavailable)
    monkeypatch.setattr(dataset_service, "DATASET_PATH", unavailable)
    dataset_service.get_raw_dataset.cache_clear()
    response = client.get("/api/dataset/schema")
    assert response.status_code == 503
    dataset_service.get_raw_dataset.cache_clear()


def test_missing_model_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(model_service, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(model_service, "MODEL_METADATA_PATH", tmp_path / "missing.json")
    model_service.load_artifact.cache_clear()
    response = client.post("/api/predict", json={})
    assert response.status_code == 503
    model_service.load_artifact.cache_clear()


def test_prediction_rejects_unknown_feature():
    response = client.post(
        "/api/predict",
        json={"movie_title": "Not part of model schema"},
    )
    assert response.status_code == 422


def test_researcher_explanation_keeps_prediction_when_gemini_is_unavailable(monkeypatch):
    def unavailable(_evidence):
        raise gemini_service.GeminiUnavailableError("Gemini is not configured.")

    monkeypatch.setattr(gemini_service, "explain_prediction", unavailable)
    response = client.post(
        "/api/researcher/explain",
        json={
            "genres": "Drama|Thriller",
            "country": "USA",
            "language": "English",
            "duration": 110,
            "budget": 25_000_000,
            "title_year": 2027,
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"]["predicted_category"] in {"Flop", "Average", "Hit"}
    assert abs(sum(result["prediction"]["probabilities"].values()) - 1) < 1e-8
    assert len(result["feature_effects"]) == 10
    assert result["historical_comparison"]["matched_movies"] > 0
    assert result["gemini_explanation"] is None
    assert result["gemini_notice"] == "Gemini is not configured."
    assert all(item["status"] == "review" for item in result["feature_availability"])
    assert "does not show causation" in result["explanation_method"]


def test_dataset_research_calculates_genre_statistics_from_csv():
    analysis = researcher_service.analyze_dataset_question(
        "Which genres have the highest average IMDb score?"
    )
    assert analysis["supported"] is True
    assert analysis["operation"] == "genre_performance"
    assert analysis["statistics"]
    assert all(item["movies"] > 0 for item in analysis["statistics"])
    assert "Cleaned dataset" in analysis["source"]
    assert analysis["statistics"][0]["mean_imdb_score"] >= analysis["statistics"][-1][
        "mean_imdb_score"
    ]


def test_dataset_research_uses_python_evidence_for_gemini(monkeypatch):
    observed = {}

    def explain(question, analysis, history):
        observed.update(question=question, analysis=analysis, history=history)
        return "The dataset shows historical patterns, not causes."

    monkeypatch.setattr(gemini_service, "explain_dataset_results", explain)
    response = client.post(
        "/api/researcher/chat",
        json={
            "mode": "dataset",
            "question": "Summarize movie budgets.",
            "history": [{"role": "user", "content": "What is in this dataset?"}],
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["evidence"]["operation"] == "numeric_summary"
    assert result["evidence"]["column"] == "budget"
    assert result["evidence"]["statistics"]["count"] > 0
    assert observed["analysis"] == result["evidence"]
    assert observed["history"][0]["role"] == "user"
    assert result["answer"].startswith("The dataset shows")


def test_general_movie_chat_is_labeled_not_live_verified(monkeypatch):
    monkeypatch.setattr(
        gemini_service,
        "answer_general_question",
        lambda question, history: "A sample general answer.",
    )
    response = client.post(
        "/api/researcher/chat",
        json={"mode": "general", "question": "Recommend a mystery film."},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "A sample general answer."
    assert "not live-verified" in response.json()["notice"]
    assert response.json()["evidence"] is None


def test_researcher_chat_reports_missing_gemini_and_rate_limits(monkeypatch):
    def unavailable(_question, _history):
        raise gemini_service.GeminiUnavailableError("Gemini is not configured.")

    monkeypatch.setattr(gemini_service, "answer_general_question", unavailable)
    unavailable_response = client.post(
        "/api/researcher/chat",
        json={"mode": "general", "question": "Tell me about filmmaking."},
    )
    assert unavailable_response.status_code == 503
    assert unavailable_response.json()["detail"] == "Gemini is not configured."

    def rate_limited(_question, _history):
        raise gemini_service.GeminiRateLimitError("Gemini rate limit reached.")

    monkeypatch.setattr(gemini_service, "answer_general_question", rate_limited)
    limited_response = client.post(
        "/api/researcher/chat",
        json={"mode": "general", "question": "Tell me about filmmaking."},
    )
    assert limited_response.status_code == 429


def test_researcher_chat_rejects_invalid_mode_and_long_history():
    invalid_mode = client.post(
        "/api/researcher/chat",
        json={"mode": "execute", "question": "Run arbitrary code."},
    )
    assert invalid_mode.status_code == 422

    invalid_history = client.post(
        "/api/researcher/chat",
        json={
            "mode": "general",
            "question": "Question",
            "history": [
                {"role": "assistant", "content": "Earlier answer."}
            ] * 13,
        },
    )
    assert invalid_history.status_code == 422


def test_researcher_prediction_rejects_invalid_numeric_input():
    response = client.post(
        "/api/researcher/explain",
        json={"duration": -1},
    )
    assert response.status_code == 422


def test_gemini_client_uses_text_generation_api(monkeypatch):
    observed = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            observed.update(kwargs)
            return SimpleNamespace(text="Verified evidence, explained simply.")

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(gemini_service, "GEMINI_API_KEY", "test-key-not-used")
    monkeypatch.setattr(gemini_service, "_client", lambda _key: FakeClient())
    assert gemini_service._generate_text("Test prompt") == "Verified evidence, explained simply."
    assert observed["contents"] == "Test prompt"
    assert observed["config"] == {"max_output_tokens": 1024}
    assert observed["model"] == gemini_service.GEMINI_MODEL


def test_gemini_service_handles_missing_key_rate_limit_and_empty_response(monkeypatch):
    monkeypatch.setattr(gemini_service, "GEMINI_API_KEY", None)
    monkeypatch.setattr(
        gemini_service,
        "_client",
        lambda _key: (_ for _ in ()).throw(AssertionError("Client should not be created")),
    )
    with pytest.raises(gemini_service.GeminiUnavailableError) as error:
        gemini_service._generate_text("Test prompt")
    assert str(error.value) == "Gemini is not configured."

    from google.genai.errors import APIError

    monkeypatch.setattr(gemini_service, "GEMINI_API_KEY", "test-key-not-used")

    class RateLimitedModels:
        def generate_content(self, **_kwargs):
            raise APIError(429, {"error": {"message": "test"}})

    class RateLimitedClient:
        models = RateLimitedModels()

    monkeypatch.setattr(gemini_service, "_client", lambda _key: RateLimitedClient())
    with pytest.raises(gemini_service.GeminiRateLimitError):
        gemini_service._generate_text("Test prompt")

    class MissingModelModels:
        def generate_content(self, **_kwargs):
            raise APIError(404, {"error": {"message": "test"}})

    class MissingModelClient:
        models = MissingModelModels()

    monkeypatch.setattr(gemini_service, "_client", lambda _key: MissingModelClient())
    with pytest.raises(gemini_service.GeminiUnavailableError) as error:
        gemini_service._generate_text("Test prompt")
    assert str(error.value) == "Gemini is temporarily unavailable."

    class TimedOutModels:
        def generate_content(self, **_kwargs):
            raise TimeoutError("test")

    class TimedOutClient:
        models = TimedOutModels()

    monkeypatch.setattr(gemini_service, "_client", lambda _key: TimedOutClient())
    with pytest.raises(gemini_service.GeminiUnavailableError) as error:
        gemini_service._generate_text("Test prompt")
    assert str(error.value) == "Gemini request timed out or could not connect."

    class EmptyModels:
        def generate_content(self, **_kwargs):
            return SimpleNamespace(text="")

    class EmptyClient:
        models = EmptyModels()

    monkeypatch.setattr(gemini_service, "_client", lambda _key: EmptyClient())
    with pytest.raises(gemini_service.GeminiUnavailableError) as error:
        gemini_service._generate_text("Test prompt")
    assert str(error.value) == "Gemini returned an empty response."
