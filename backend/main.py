from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from backend.analysis import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MODEL_NAME, BaseSemanticMarkdownAnalyzer, FullContextSAVISAnalyzer
from backend.db import create_pool
from backend.schemas import (
    AnalyzeRequest,
    DeltaCaptureResponse,
    DeltaVisualizationResponse,
    DocumentAttentionResponse,
    DocumentLayerStatesResponse,
    HealthResponse,
    LayerDriftResponse,
    SentencePairAttentionResponse,
    SentencePairRequest,
    SentenceWeightResponse,
    VectorIngestRequest,
    VectorIngestResponse,
    VectorSearchRequest,
    VectorSearchResponse,
)
from backend.vector_store import (
    compute_content_hash,
    get_cross_layer_drift,
    get_document_layer_states,
    ingest_document,
    search_by_layer_vector,
)

logger = logging.getLogger(__name__)


def create_app(
    analyzer: BaseSemanticMarkdownAnalyzer | None = None,
    *,
    eager_load_default: bool = True,
    enable_vector_store: bool = True,
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Initialise model
        if app.state.analyzer is None and eager_load_default:
            try:
                app.state.analyzer = FullContextSAVISAnalyzer()
                app.state.ready = True
            except Exception as exc:  # pragma: no cover
                app.state.startup_error = str(exc)

        # Initialise DB pool
        if enable_vector_store:
            try:
                app.state.db_pool = await create_pool()
                logger.info("pgvector pool ready.")
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not connect to PostgreSQL: %s", exc)
                app.state.db_pool = None
        else:
            app.state.db_pool = None

        yield

        if app.state.db_pool is not None:
            await app.state.db_pool.close()

    app = FastAPI(
        title="Semantic Attention Markdown Backend",
        version="0.3.0",
        lifespan=lifespan,
    )

    app.state.ready = analyzer is not None or not eager_load_default
    app.state.analyzer = analyzer
    app.state.startup_error = None
    app.state.inference_lock = asyncio.Lock()
    app.state.db_pool = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def resolve_runtime_analyzer() -> BaseSemanticMarkdownAnalyzer | None:
        runtime_analyzer = getattr(app.state, "analyzer", None)
        if runtime_analyzer is not None:
            return runtime_analyzer

        if getattr(app.state, "startup_error", None):
            return None

        try:
            runtime_analyzer = FullContextSAVISAnalyzer()
            app.state.analyzer = runtime_analyzer
            app.state.ready = True
        except Exception as exc:  # pragma: no cover - exercised by runtime only
            app.state.startup_error = str(exc)
            app.state.ready = False
            return None

        return runtime_analyzer

    async def run_model_operation(operation, *, allow_not_implemented: bool = True):
        async with app.state.inference_lock:
            runtime_analyzer = resolve_runtime_analyzer()
            if not getattr(app.state, "ready", False) or runtime_analyzer is None:
                detail = getattr(app.state, "startup_error", None) or "Model is still loading."
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)

            try:
                return operation(runtime_analyzer)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
            except NotImplementedError as exc:
                if allow_not_implemented:
                    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    async def capture_delta_state(text: str):
        return await run_model_operation(
            lambda runtime_analyzer: (
                runtime_analyzer.capture_delta_document(text),
                runtime_analyzer._compute_state(text),
            )
        )

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        async with app.state.inference_lock:
            runtime_analyzer = resolve_runtime_analyzer()
        return HealthResponse(
            ready=bool(getattr(app.state, "ready", False) and runtime_analyzer),
            modelName=getattr(runtime_analyzer, "model_name", DEFAULT_MODEL_NAME),
            device=getattr(runtime_analyzer, "device", "uninitialized"),
            maxInputTokens=getattr(runtime_analyzer, "max_input_tokens", DEFAULT_MAX_INPUT_TOKENS),
            error=getattr(app.state, "startup_error", None),
        )

    @app.post("/analyze", response_model=list[SentenceWeightResponse])
    async def analyze_markdown(request: AnalyzeRequest) -> list[SentenceWeightResponse]:
        payload = await run_model_operation(
            lambda runtime_analyzer: runtime_analyzer.analyze_markdown(request.text),
            allow_not_implemented=False,
        )

        return [SentenceWeightResponse.model_validate(item) for item in payload]

    @app.post("/attention/document", response_model=DocumentAttentionResponse)
    async def analyze_attention_document(request: AnalyzeRequest) -> DocumentAttentionResponse:
        payload = await run_model_operation(
            lambda runtime_analyzer: runtime_analyzer.analyze_document(request.text),
        )

        return DocumentAttentionResponse.model_validate(payload)

    @app.post("/attention/pair", response_model=SentencePairAttentionResponse)
    async def analyze_attention_pair(request: SentencePairRequest) -> SentencePairAttentionResponse:
        payload = await run_model_operation(
            lambda runtime_analyzer: runtime_analyzer.analyze_sentence_pair(
                request.text,
                request.sentenceXIndex,
                request.sentenceYIndex,
            ),
        )

        return SentencePairAttentionResponse.model_validate(payload)

    @app.post("/delta/document", response_model=DeltaCaptureResponse)
    async def capture_delta_document(request: AnalyzeRequest) -> DeltaCaptureResponse:
        payload = await run_model_operation(
            lambda runtime_analyzer: runtime_analyzer.capture_delta_document(request.text),
        )

        return DeltaCaptureResponse.model_validate(payload)

    @app.post("/delta/visualization", response_model=DeltaVisualizationResponse)
    async def analyze_delta_visualization(request: AnalyzeRequest) -> DeltaVisualizationResponse:
        payload = await run_model_operation(
            lambda runtime_analyzer: runtime_analyzer.analyze_delta_visualization(request.text),
        )

        return DeltaVisualizationResponse.model_validate(payload)

    # ── Vector Store Endpoints ─────────────────────────────────────────────────

    def _require_db():
        pool = getattr(app.state, "db_pool", None)
        if pool is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector store database is not available. Check ATTN_DB_* env vars.",
            )
        return pool

    @app.post("/vectors/ingest", response_model=VectorIngestResponse)
    async def vector_ingest(request: VectorIngestRequest) -> VectorIngestResponse:
        """Run delta capture for a document and persist its residual vectors to pgvector."""
        pool = _require_db()

        capture, state = await capture_delta_state(request.text)

        sentence_payloads   = state.sentence_payloads
        sentence_boundaries = state.sentence_boundaries_ids

        doc_id = await ingest_document(
            pool,
            markdown_text=request.text,
            model_name=capture["storageKey"],
            sentences=sentence_payloads,
            sentence_boundaries=sentence_boundaries,
            residual_inputs_path=capture["tensorPaths"]["residual_inputs"],
            residual_after_attention_path=capture["tensorPaths"]["residual_after_attention"],
            layer_count=capture["layerCount"],
            head_count=capture["headCount"],
            hidden_size=capture["hiddenSize"],
            head_dim=capture["headDim"],
            token_count=capture["tokenCount"],
        )

        return VectorIngestResponse(
            documentId=doc_id,
            contentHash=compute_content_hash(request.text),
            sentenceCount=len(sentence_payloads),
            layerCount=capture["layerCount"],
            vectorsInserted=len(sentence_payloads) * capture["layerCount"] * 2,
        )

    @app.post("/vectors/search", response_model=VectorSearchResponse)
    async def vector_search(request: VectorSearchRequest) -> VectorSearchResponse:
        """
        Find sentences in the database whose residual state at `targetLayer` is
        closest (cosine) to the query sentence's residual state at `sourceLayer`.
        """
        pool = _require_db()

        capture, state = await capture_delta_state(request.text)

        n_sentences = len(state.sentences)
        if not 0 <= request.sentenceIndex < n_sentences:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"sentenceIndex {request.sentenceIndex} is out of range (0-{n_sentences - 1}).",
            )
        if request.sourceLayer >= capture["layerCount"] or request.targetLayer >= capture["layerCount"]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Layer index out of range. Model has {capture['layerCount']} layers (0-indexed).",
            )

        # Load residual vectors and extract the query sentence's mean-pooled vector.
        boundaries = state.sentence_boundaries_ids
        t_start = boundaries[request.sentenceIndex]
        t_end   = boundaries[request.sentenceIndex + 1] if request.sentenceIndex + 1 < len(boundaries) else capture["tokenCount"]

        tensor_key = (
            "residual_inputs" if request.vectorType == "residual_input"
            else "residual_after_attention"
        )
        tensor = np.load(capture["tensorPaths"][tensor_key], mmap_mode="r", allow_pickle=False)
        query_vec = tensor[request.sourceLayer, t_start:t_end, :].mean(axis=0).astype(np.float32)

        results = await search_by_layer_vector(
            pool,
            query_vector=query_vec,
            target_layer=request.targetLayer,
            vector_type=request.vectorType,
            limit=request.limit,
        )

        query_sentence = state.sentences[request.sentenceIndex] if state.sentences else ""
        return VectorSearchResponse(
            queryText=query_sentence,
            sourceLayer=request.sourceLayer,
            targetLayer=request.targetLayer,
            vectorType=request.vectorType,
            results=[
                {
                    "sentenceId":     r["sentence_id"],
                    "sentenceText":   r["sentence_text"],
                    "sentenceIndex":  r["sentence_index"],
                    "attentionScore": r["attention_score"],
                    "band":           r["band"],
                    "documentId":     r["document_id"],
                    "contentHash":    r["content_hash"],
                    "modelName":      r["model_name"],
                    "layerIndex":     r["layer_index"],
                    "cosineDistance": float(r["cosine_distance"]),
                }
                for r in results
            ],
        )

    @app.get("/vectors/drift/{sentence_id}", response_model=LayerDriftResponse)
    async def vector_drift(sentence_id: int) -> LayerDriftResponse:
        """
        Return pairwise cosine similarity between adjacent layers for a given sentence.
        Low similarity between layer N and N+1 means high representational drift at that layer.
        """
        pool = _require_db()

        async with pool.acquire() as conn:
            sent_row = await conn.fetchrow(
                "SELECT id, text FROM sentences WHERE id = $1", sentence_id
            )
        if sent_row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sentence not found.")

        drift_rows = await get_cross_layer_drift(pool, sentence_id=sentence_id)
        return LayerDriftResponse(
            sentenceId=sentence_id,
            sentenceText=sent_row["text"],
            drift=[
                {"fromLayer": r["from_layer"], "toLayer": r["to_layer"], "cosineSimilarity": float(r["cosine_similarity"])}
                for r in drift_rows
            ],
        )

    @app.get("/vectors/layers/{document_id}", response_model=DocumentLayerStatesResponse)
    async def vector_layer_states(
        document_id: int,
        vector_type: str = Query(default="residual_after_attn"),
    ) -> DocumentLayerStatesResponse:
        """List all sentence-layer state metadata (no raw embeddings) for a document."""
        pool = _require_db()
        if vector_type not in ("residual_input", "residual_after_attn"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="vector_type must be 'residual_input' or 'residual_after_attn'.",
            )
        rows = await get_document_layer_states(pool, document_id=document_id, vector_type=vector_type)
        return DocumentLayerStatesResponse(
            documentId=document_id,
            vectorType=vector_type,
            states=[
                {
                    "sentenceId":     r["sentence_id"],
                    "sentenceIndex":  r["sentence_index"],
                    "text":           r["text"],
                    "attentionScore": r["attention_score"],
                    "band":           r["band"],
                    "layerIndex":     r["layer_index"],
                }
                for r in rows
            ],
        )

    return app


app = create_app(eager_load_default=False)
