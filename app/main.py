from fastapi import FastAPI, Request, HTTPException, status, Depends
from time import perf_counter
from app.core import lifespan, settings
import logging
from app.db import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.schema import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    LLMMetadata,
    TokenUsage,
)
from app.retrieval.retrieval import retrieve
from app.generation import generate_answer, GenerationError
from app.embeddings.sentence_transformer import embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Pipeline & Evaluation API", lifespan=lifespan)

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = perf_counter()
        response = await call_next(request)
        process_time = (perf_counter() - start_time) * 1000
        response.headers["X-Process-Time"] = f"{process_time: .2f} ms"
        return response

    @app.get("/health", response_model=HealthResponse)
    async def check_health(session: AsyncSession = Depends(get_db)) -> HealthResponse:
        try:
            await session.execute(text("SELECT 1"))
            return HealthResponse(status="ok", message="Service is running")
        except Exception as e:
            logger.error("Health check database failure: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database is unavailable",
            )

    @app.post("/query", response_model=QueryResponse)
    async def query(
        request_body: QueryRequest,
        session: AsyncSession = Depends(get_db),
    ) -> QueryResponse:
        start = perf_counter()

        # 1. embed query
        query_vector = None
        if request_body.retrieval_mode in ("vector", "hybrid"):
            query_vector = await embedder.embed_query(request_body.query)

        # 2. retrieve relevant chunks
        chunks = await retrieve(
            session=session,
            query=request_body.query,
            top_k=request_body.top_k,
            mode=request_body.retrieval_mode,
            rrf_k=request_body.rrf_k,
            query_vector=query_vector,
            rerank=request_body.rerank,
            rerank_threshold=request_body.rerank_threshold,
            rrf_mode=request_body.rrf_mode,
            vector_weight=request_body.vector_weight,
            lexical_weight=request_body.lexical_weight,
        )

        if not chunks:
            latency_ms = (perf_counter() - start) * 1000
            return QueryResponse(
                answer="I cannot answer based on the provided context.",
                sources=[],
                retrieval_mode=request_body.retrieval_mode,
                llm_metadata=LLMMetadata(
                    model=settings.ollama_model,
                    token_usage=TokenUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                    ),
                ),
                latency_ms=round(latency_ms, 2),
            )

        # 3. generate answer
        try:
            result = await generate_answer(
                query=request_body.query,
                chunks=chunks,
            )
        except GenerationError as exc:
            logger.error("LLM Generation failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            )

        latency_ms = (perf_counter() - start) * 1000

        return QueryResponse(
            answer=result.answer,
            sources=[
                SourceChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    text=c.content,
                    score=round(c.score, 6),
                    start_offset=c.start_offset,
                    end_offset=c.end_offset,
                )
                for c in chunks
            ],
            retrieval_mode=request_body.retrieval_mode,
            llm_metadata=LLMMetadata(
                model=result.model,
                token_usage=TokenUsage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                ),
            ),
            latency_ms=round(latency_ms, 2),
        )

    return app


app: FastAPI = create_app()
