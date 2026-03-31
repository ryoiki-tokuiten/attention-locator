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
    tokens: list[str]
    tensorShapes: dict[str, list[int]]
    tensorPaths: dict[str, str]
    warnings: list[str] = Field(default_factory=list)


class DeltaSourceContributionResponse(BaseModel):
    sourceIndex: int
    score: float
    scoreNorm: float
    share: float
    distance: int
    distanceBucket: Literal["local", "medium", "global"]
    sentence: str
    displaySentence: str
    sourceStart: int | None = None
    sourceEnd: int | None = None
    visible: bool


class DeltaSentenceVisualizationResponse(BaseModel):
    index: int
    sentence: str
    displaySentence: str
    sourceStart: int | None = None
    sourceEnd: int | None = None
    visible: bool
    incomingFlow: float
    incomingFlowNorm: float
    localShare: float
    mediumShare: float
    globalShare: float
    drift: float
    driftNorm: float
    peakLayer: int
    peakLayerRatio: float
    topSources: list[DeltaSourceContributionResponse] = Field(default_factory=list)
    vectorFlow: dict[str, float] | None = None


class DeltaGlobalEdgeResponse(BaseModel):
    id: str
    sourceIndex: int
    targetIndex: int
    score: float
    scoreNorm: float
    distance: int
    distanceBucket: Literal["local", "medium", "global"]
    color: str


class DeltaVectorFlowBoundsResponse(BaseModel):
    minX: float
    maxX: float
    minY: float
    maxY: float


class DeltaVectorFlowResponse(BaseModel):
    projection: str
    angleAmplification: float
    baseWeight: float
    deltaWeight: float
    unitCircleRadius: float
    bounds: DeltaVectorFlowBoundsResponse


class DeltaVisualizationResponse(BaseModel):
    analysisText: str
    tokenCount: int
    sentenceCount: int
    layerCount: int
    headCount: int
    hiddenSize: int
    defaultMode: str
    sentences: list[DeltaSentenceVisualizationResponse]
    driftByLayer: list[list[float]]
    globalTopEdges: list[DeltaGlobalEdgeResponse] = Field(default_factory=list)
    vectorFlow: DeltaVectorFlowResponse
    warnings: list[str] = Field(default_factory=list)
