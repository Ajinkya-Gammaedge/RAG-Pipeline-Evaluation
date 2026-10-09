from sqlalchemy.ext.asyncio import async_sessionmaker
from .engine import engine

session_local = async_sessionmaker(bind=engine, expire_on_commit=False)


async def get_db():
    async with session_local() as session:
        yield session
