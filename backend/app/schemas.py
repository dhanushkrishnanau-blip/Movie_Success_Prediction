from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration: Optional[float] = Field(default=None, ge=0, le=1000)
    genres: Optional[str] = Field(default=None, max_length=500)
    director_name: Optional[str] = Field(default=None, max_length=200)
    actor_1_name: Optional[str] = Field(default=None, max_length=200)
    actor_2_name: Optional[str] = Field(default=None, max_length=200)
    actor_3_name: Optional[str] = Field(default=None, max_length=200)
    budget: Optional[float] = Field(default=None, ge=0, le=1_000_000_000_000)
    title_year: Optional[float] = Field(default=None, ge=1800, le=2200)
    language: Optional[str] = Field(default=None, max_length=100)
    country: Optional[str] = Field(default=None, max_length=100)

    @field_validator(
        "genres",
        "director_name",
        "actor_1_name",
        "actor_2_name",
        "actor_3_name",
        "language",
        "country",
        mode="before",
    )
    @classmethod
    def normalize_empty_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip() or None
        return value


class PredictionResponse(BaseModel):
    predicted_category: str
    model_name: str
    probabilities: dict[str, float] | None
    note: str


class ResearchChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1200)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Chat messages cannot be blank.")
        return value


class ResearchChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["general", "dataset"]
    question: str = Field(min_length=1, max_length=1200)
    history: list[ResearchChatMessage] = Field(default_factory=list, max_length=12)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank.")
        return value
