from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db import engine
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.connect() as conncetion:
            await conncetion.execute(text("SELECT 1"))
            logger.info("Database connection successful")

    except Exception as e:
        logger.error("Database conncetion failed", e)

        raise

    yield

    await engine.dispose()
    logger.info("Database connection pool closed")
