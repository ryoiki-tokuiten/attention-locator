from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.analysis import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MODEL_NAME, BaseSemanticMarkdownAnalyzer, FullContextSAVISAnalyzer
from backend.schemas import (
    AnalyzeRequest,
    DocumentAttentionResponse,
    HealthResponse,
    SentencePairAttentionResponse,
    SentencePairRequest,
    SentenceWeightResponse,
)


def create_app(
    analyzer: BaseSemanticMarkdownAnalyzer | None = None,
    *,
    eager_load_default: bool = True,
) -> FastAPI:
    app = FastAPI(
        title="Semantic Attention Markdown Backend",
        version="0.2.0",
    )

    app.state.ready = False
    app.state.analyzer = analyzer
    app.state.startup_error = None
    app.state.inference_lock = asyncio.Lock()

    if app.state.analyzer is None and eager_load_default:
        try:
            app.state.analyzer = FullContextSAVISAnalyzer()
            app.state.ready = True
        except Exception as exc:  # pragma: no cover - exercised by runtime only
            app.state.startup_error = str(exc)
    else:
        app.state.ready = True

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
        async with app.state.inference_lock:
            runtime_analyzer = resolve_runtime_analyzer()
            if not getattr(app.state, "ready", False) or runtime_analyzer is None:
                detail = getattr(app.state, "startup_error", None) or "Model is still loading."
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)

            try:
                payload = runtime_analyzer.analyze_markdown(request.text)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        return [SentenceWeightResponse.model_validate(item) for item in payload]

    @app.post("/attention/document", response_model=DocumentAttentionResponse)
    async def analyze_attention_document(request: AnalyzeRequest) -> DocumentAttentionResponse:
        async with app.state.inference_lock:
            runtime_analyzer = resolve_runtime_analyzer()
            if not getattr(app.state, "ready", False) or runtime_analyzer is None:
                detail = getattr(app.state, "startup_error", None) or "Model is still loading."
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)

            try:
                payload = runtime_analyzer.analyze_document(request.text)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
            except NotImplementedError as exc:
                raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        return DocumentAttentionResponse.model_validate(payload)

    @app.post("/attention/pair", response_model=SentencePairAttentionResponse)
    async def analyze_attention_pair(request: SentencePairRequest) -> SentencePairAttentionResponse:
        async with app.state.inference_lock:
            runtime_analyzer = resolve_runtime_analyzer()
            if not getattr(app.state, "ready", False) or runtime_analyzer is None:
                detail = getattr(app.state, "startup_error", None) or "Model is still loading."
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)

            try:
                payload = runtime_analyzer.analyze_sentence_pair(
                    request.text,
                    request.sentenceXIndex,
                    request.sentenceYIndex,
                )
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
            except NotImplementedError as exc:
                raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        return SentencePairAttentionResponse.model_validate(payload)

    return app


app = create_app(eager_load_default=False)
