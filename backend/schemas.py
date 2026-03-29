from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SemanticBand = Literal["low", "medium", "high"]


class AnalyzeRequest(BaseModel):
    text: str = Field(default="", description="Raw markdown content to analyze.")


class SentencePairRequest(AnalyzeRequest):
    sentenceXIndex: int = Field(ge=0)
    sentenceYIndex: int = Field(ge=0)


class SentenceWeightResponse(BaseModel):
    index: int
    sentence: str
    normalizedSentence: str
    weight: float
    band: SemanticBand
    sourceStart: int | None = None
    sourceEnd: int | None = None


class HealthResponse(BaseModel):
    ready: bool
    modelName: str
    device: str
    maxInputTokens: int | None = None
    error: str | None = None


class DocumentAttentionResponse(BaseModel):
    analysisText: str
    tokenCount: int
    sentenceCount: int
    sentenceWeights: list[SentenceWeightResponse]
    sentenceAttention: list[list[float]]
    warnings: list[str] = Field(default_factory=list)


class SentencePairAttentionResponse(BaseModel):
    sentenceXIndex: int
    sentenceYIndex: int
    sentenceX: str
    sentenceY: str
    columnTokens: list[str]
    rowTokens: list[str]
    attention: list[list[float]]
    warnings: list[str] = Field(default_factory=list)


class DeltaCaptureResponse(BaseModel):
    cacheKey: str
    analysisText: str
    storageKey: str
    artifactDir: str
    tokenCount: int
    layerCount: int
    headCount: int
    hiddenSize: int
    headDim: int
    runtimeDtype: str
    storageDtype: str
    estimatedStorageBytes: int
    tensorShapes: dict[str, list[int]]
    tensorPaths: dict[str, str]
    warnings: list[str] = Field(default_factory=list)
