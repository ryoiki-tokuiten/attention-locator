from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.analysis import DEFAULT_MODEL_NAME, BaseSemanticMarkdownAnalyzer, SlidingWindowISAAnalyzer
from backend.schemas import AnalyzeRequest, HealthResponse, SentenceWeightResponse


def create_app(analyzer: BaseSemanticMarkdownAnalyzer | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ready = False
        app.state.analyzer = analyzer
        app.state.startup_error = None

        if app.state.analyzer is None:
            try:
                app.state.analyzer = SlidingWindowISAAnalyzer()
                app.state.ready = True
            except Exception as exc:  # pragma: no cover - exercised by runtime only
                app.state.startup_error = str(exc)
        else:
            app.state.ready = True

        yield

    app = FastAPI(
        title="Semantic Attention Markdown Backend",
        version="0.1.0",
        lifespan=lifespan,
    )

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

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        runtime_analyzer = getattr(app.state, "analyzer", None)
        return HealthResponse(
            ready=bool(getattr(app.state, "ready", False) and runtime_analyzer),
            modelName=getattr(runtime_analyzer, "model_name", DEFAULT_MODEL_NAME),
            device=getattr(runtime_analyzer, "device", "uninitialized"),
            error=getattr(app.state, "startup_error", None),
        )

    @app.post("/analyze", response_model=list[SentenceWeightResponse])
    async def analyze_markdown(request: AnalyzeRequest) -> list[SentenceWeightResponse]:
        runtime_analyzer = getattr(app.state, "analyzer", None)
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

    return app


app = create_app()
