from sqlalchemy.ext.asyncio import create_async_engine
from app.core import settings

engine = create_async_engine(
    url=settings.database_url,
    echo=False,
    pool_pre_ping=True,
)
