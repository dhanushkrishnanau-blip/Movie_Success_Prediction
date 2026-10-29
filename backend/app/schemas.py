from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration: Optional[float] = Field(default=None, ge=0, le=1000)
    genres: Optional[str] = None
    director_name: Optional[str] = Field(default=None, max_length=200)
    actor_1_name: Optional[str] = Field(default=None, max_length=200)
    actor_2_name: Optional[str] = Field(default=None, max_length=200)
    actor_3_name: Optional[str] = Field(default=None, max_length=200)
    budget: Optional[float] = Field(default=None, ge=0)
    title_year: Optional[float] = Field(default=None, ge=1800, le=2200)
    language: Optional[str] = None
    country: Optional[str] = None


class PredictionResponse(BaseModel):
    predicted_category: str
    model_name: str
    probabilities: dict[str, float] | None
    note: str
