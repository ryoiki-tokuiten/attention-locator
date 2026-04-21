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


# ── Vector Store Schemas ───────────────────────────────────────────────────────

class VectorIngestRequest(AnalyzeRequest):
    """Triggers delta capture + DB ingestion for the given markdown."""
    pass


class VectorIngestResponse(BaseModel):
    documentId: int
    contentHash: str
    sentenceCount: int
    layerCount: int
    vectorsInserted: int


class VectorSearchRequest(BaseModel):
    text: str = Field(description="Source markdown document.")
    sentenceIndex: int = Field(ge=0, description="Which sentence to use as the query vector.")
    sourceLayer: int = Field(ge=0, description="Which layer to take the query vector from.")
    targetLayer: int = Field(ge=0, description="Which layer to search in.")
    vectorType: Literal["residual_input", "residual_after_attn"] = "residual_after_attn"
    limit: int = Field(default=10, ge=1, le=100)


class VectorSearchResult(BaseModel):
    sentenceId: int
    sentenceText: str
    sentenceIndex: int
    attentionScore: float | None
    band: str | None
    documentId: int
    contentHash: str
    modelName: str
    layerIndex: int
    cosineDistance: float


class VectorSearchResponse(BaseModel):
    queryText: str
    sourceLayer: int
    targetLayer: int
    vectorType: str
    results: list[VectorSearchResult]


class LayerDriftEntry(BaseModel):
    fromLayer: int
    toLayer: int
    cosineSimilarity: float


class LayerDriftResponse(BaseModel):
    sentenceId: int
    sentenceText: str
    drift: list[LayerDriftEntry]


class DocumentLayerStateEntry(BaseModel):
    sentenceId: int
    sentenceIndex: int
    text: str
    attentionScore: float | None
    band: str | None
    layerIndex: int


class DocumentLayerStatesResponse(BaseModel):
    documentId: int
    vectorType: str
    states: list[DocumentLayerStateEntry]
