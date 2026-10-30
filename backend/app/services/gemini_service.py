import json
from functools import lru_cache
from typing import Any

from app.config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiUnavailableError(RuntimeError):
    pass


class GeminiRateLimitError(GeminiUnavailableError):
    pass


@lru_cache(maxsize=1)
def _client(api_key: str) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=25_000),
    )


def _generate_text(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise GeminiUnavailableError("Gemini is not configured.")

    from google.genai.errors import APIError
    from httpx import TimeoutException, TransportError

    try:
        response = _client(GEMINI_API_KEY).models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"max_output_tokens": 1024},
        )
    except (
        TimeoutException,
        TransportError,
        TimeoutError,
    ) as error:
        raise GeminiUnavailableError("Gemini request timed out or could not connect.") from error
    except APIError as error:
        status_code = getattr(error, "status_code", getattr(error, "code", None))
        if status_code == 429:
            raise GeminiRateLimitError("Gemini rate limit reached. Try again later.") from error
        raise GeminiUnavailableError("Gemini is temporarily unavailable.") from error

    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise GeminiUnavailableError("Gemini returned an empty response.")
    return text.strip()


def explain_prediction(evidence: dict[str, Any]) -> str:
    prompt = (
        "You explain a movie category prediction made by a fixed Random Forest. "
        "The JSON below is verified evidence from the model and Python dataset calculations. "
        "Write a short, simple explanation. Never alter the predicted category, invent or "
        "recalculate statistics, introduce new examples, or say a feature caused an outcome. "
        "Describe feature perturbations only as local changes in this model's output, not "
        "causal contributions. Do not add numeric values; the exact metrics are displayed "
        "separately from your narrative. Distinguish historical association from model output "
        "and mention uncertainty or insufficient sample size when present.\n\n"
        f"VERIFIED_EVIDENCE_JSON:\n{json.dumps(evidence, ensure_ascii=True, allow_nan=False)}"
    )
    return _generate_text(prompt)


def answer_general_question(
    question: str, history: list[dict[str, str]]
) -> str:
    context = _history_context(history)
    prompt = (
        "You are a helpful general movie and filmmaking assistant for a student capstone. "
        "Answer clearly and simply. You have no live web or database access: do not claim "
        "current or verified facts unless supplied in the conversation. Mark uncertainty, "
        "and do not pretend recommendations or opinions are objective facts.\n\n"
        f"{context}\nUSER QUESTION:\n{question}"
    )
    return _generate_text(prompt)


def explain_dataset_results(
    question: str,
    analysis: dict[str, Any],
    history: list[dict[str, str]],
) -> str:
    context = _history_context(history)
    prompt = (
        "You explain results calculated by Python from the project's movie dataset. "
        "Use only the supplied analysis JSON for dataset claims. Do not invent, estimate, "
        "recalculate, or add statistics, film examples, or rankings. Do not describe "
        "associations as causes. If supported is false or the sample is too small, say the "
        "dataset cannot answer reliably. Explain limitations in simple language. The exact "
        "statistics are shown separately in the application; do not add numeric values.\n\n"
        f"{context}\nUSER QUESTION:\n{question}\n"
        f"PYTHON_ANALYSIS_JSON:\n{json.dumps(analysis, ensure_ascii=True, allow_nan=False)}"
    )
    return _generate_text(prompt)


def _history_context(history: list[dict[str, str]]) -> str:
    if not history:
        return "There is no earlier conversation."
    safe_history = history[-12:]
    return "Recent conversation (context only):\n" + "\n".join(
        f"{message['role']}: {message['content'][:1200]}"
        for message in safe_history
    )
