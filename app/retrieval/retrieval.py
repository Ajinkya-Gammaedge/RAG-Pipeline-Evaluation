from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal
from sqlalchemy import select, func, Text
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.document_chunk import DocumentChunk
from app.embeddings.sentence_tranformer import embedder


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    id: int
    document_id: int
    chunk_id: int
    content: str
    score: float
    start_offset: int
    end_offset: int
    metadata: dict[str, Any]


async def vector_retrieval(
    session: AsyncSession,
    query: str,
    top_k: int = 10,
    query_vector: list[float] | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k chunks using pgvector cosine distance (HNSW index)
    Returns chunks ordered by similarity score descending
    """
    if query_vector is None:
        query_vector = await embedder.embed_query(query)

    distance_expr = DocumentChunk.embedding_vector.cosine_distance(query_vector)
    similarity_expr = (1.0 - distance_expr).label("similarity_score")

    stmt = (
        select(DocumentChunk, similarity_expr)
        .order_by(distance_expr.asc())
        .limit(top_k)
    )

    result = await session.execute(stmt)
    rows = result.all()

    return [
        RetrievedChunk(
            id=chunk.id,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            score=float(similarity),
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            metadata=chunk.chunk_metadata or {},
        )
        for chunk, similarity in rows
    ]


async def lexical_retrieval(
    session: AsyncSession,
    query: str,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k chunks using PostgreSQL Full-Text Search (ts_rank_cd, GIN index)
    Returns chunks ordered by rank score descending
    """
    ts_query = func.plainto_tsquery("english", query)

    rank_expr = func.ts_rank_cd(DocumentChunk.search_vector, ts_query).label(
        "rank_score"
    )

    stmt = (
        select(DocumentChunk, rank_expr)
        .where(DocumentChunk.search_vector.op("@@")(ts_query))
        .order_by(rank_expr.desc())
        .limit(top_k)
    )

    result = await session.execute(stmt)
    rows = result.all()

    return [
        RetrievedChunk(
            id=chunk.id,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            score=float(rank_score),
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            metadata=chunk.chunk_metadata or {},
        )
        for chunk, rank_score in rows
    ]


async def hybrid_retrieval(
    session: AsyncSession,
    query: str,
    top_k: int = 10,
    rrf_k: int = 60,
    query_vector: list[float] | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k chunks using Reciprocal Rank Fusion (RRF) of vector and lexical results
    Formula: score = sum(1 / (rrf_k + rank))
    """
    candidate_pool_size = max(top_k * 2, 60)

    vector_candidates = await vector_retrieval(
        session=session,
        query=query,
        top_k=candidate_pool_size,
        query_vector=query_vector,
    )
    lexical_candidates = await lexical_retrieval(
        session=session,
        query=query,
        top_k=candidate_pool_size,
    )

    rrf_scores: dict[int, float] = defaultdict(float)
    chunk_map: dict[int, RetrievedChunk] = {}

    for rank, chunk in enumerate(vector_candidates, start=1):
        rrf_scores[chunk.id] += 1.0 / (rrf_k + rank)
        chunk_map[chunk.id] = chunk

    for rank, chunk in enumerate(lexical_candidates, start=1):
        rrf_scores[chunk.id] += 1.0 / (rrf_k + rank)
        if chunk.id not in chunk_map:
            chunk_map[chunk.id] = chunk

    sorted_chunk_ids = sorted(
        rrf_scores.keys(),
        key=lambda cid: rrf_scores[cid],
        reverse=True,
    )[:top_k]

    return [
        RetrievedChunk(
            id=chunk_map[cid].id,
            document_id=chunk_map[cid].document_id,
            chunk_id=chunk_map[cid].chunk_id,
            content=chunk_map[cid].content,
            score=round(rrf_scores[cid], 6),
            start_offset=chunk_map[cid].start_offset,
            end_offset=chunk_map[cid].end_offset,
            metadata=chunk_map[cid].metadata,
        )
        for cid in sorted_chunk_ids
    ]


async def retrieve(
    session: AsyncSession,
    query: str,
    top_k: int = 10,
    mode: Literal["vector", "lexical", "hybrid"] = "hybrid",
    rrf_k: int = 60,
    query_vector: list[float] | None = None,
) -> list[RetrievedChunk]:

    if mode == "vector":
        return await vector_retrieval(
            session=session,
            query=query,
            top_k=top_k,
            query_vector=query_vector,
        )
    elif mode == "lexical":
        return await lexical_retrieval(
            session=session,
            query=query,
            top_k=top_k,
        )
    elif mode == "hybrid":
        return await hybrid_retrieval(
            session=session,
            query=query,
            top_k=top_k,
            rrf_k=rrf_k,
            query_vector=query_vector,
        )
    else:
        raise ValueError(
            f"Unknown retrieval mode: {mode!r}. Must be 'vector', 'lexical', or 'hybrid'."
        )
