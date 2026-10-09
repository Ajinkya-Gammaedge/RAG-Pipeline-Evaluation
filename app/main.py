from fastapi import FastAPI, Request, HTTPException, status, Depends
from time import perf_counter
from app.schema import HealthResponse
from app.core import lifespan
import logging
from app.db import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

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
            logger.error(e)
            raise HTTPException(
                status_code=503,
                detail="Database is unavailable",
            )

    @app.get("/query")
    async def query():
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not Implemented"
        )

    return app


app: FastAPI = create_app()
