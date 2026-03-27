from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SemanticBand = Literal["low", "medium", "high"]


class AnalyzeRequest(BaseModel):
    text: str = Field(default="", description="Raw markdown content to analyze.")


class SentenceWeightResponse(BaseModel):
    index: int
    sentence: str
    normalizedSentence: str
    weight: float
    band: SemanticBand


class HealthResponse(BaseModel):
    ready: bool
    modelName: str
    device: str
    error: str | None = None
