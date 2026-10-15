from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal
from sqlalchemy import select, func, Text
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.document_chunk import DocumentChunk
from app.embeddings.sentence_transformer import embedder
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import reranker
import asyncio


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
    # ts_query = func.websearch_to_tsquery("english", query)
    or_query_expr = func.to_tsquery(
        "english",
        func.nullif(
            func.regexp_replace(
                func.plainto_tsquery("english", query).cast(Text),
                " & ",
                " | ",
                "g",
            ),
            "",
        ),
    )

    rank_expr = func.ts_rank_cd(DocumentChunk.search_vector, or_query_expr).label(
        "rank_score"
    )

    stmt = (
        select(DocumentChunk, rank_expr)
        # .where(DocumentChunk.search_vector.op("@@")(ts_query))
        .where(DocumentChunk.search_vector.op("@@")(or_query_expr))
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


def reciprocal_rank_fusion(
    vector_candidates: list[RetrievedChunk],
    lexical_candidates: list[RetrievedChunk],
    rrf_k: int = 60,
    top_k: int = 10,
    rrf_mode: Literal["standard", "weighted"] = "standard",
    vector_weight: float = 0.7,
    lexical_weight: float = 0.3,
) -> list[RetrievedChunk]:
    """
    - standard: 1:1 equal weighting (weight=1.0 for each source)
    - weighted: custom weights (default: 0.7 vector / 0.3 lexical)
    Formula: score(c) = sum( weight / (rrf_k + rank) )
    """
    if rrf_mode == "weighted":
        w_vec = vector_weight
        w_lex = lexical_weight
    else:
        w_vec = 1.0
        w_lex = 1.0

    rrf_scores: dict[int, float] = defaultdict(float)
    chunk_map: dict[int, RetrievedChunk] = {}

    for rank, chunk in enumerate(vector_candidates, start=1):
        rrf_scores[chunk.id] += w_vec / (rrf_k + rank)
        chunk_map[chunk.id] = chunk

    for rank, chunk in enumerate(lexical_candidates, start=1):
        rrf_scores[chunk.id] += w_lex / (rrf_k + rank)
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


async def hybrid_retrieval(
    session: AsyncSession,
    query: str,
    top_k: int = 10,
    rrf_k: int = 60,
    query_vector: list[float] | None = None,
    rrf_mode: Literal["standard", "weighted"] = "standard",
    vector_weight: float = 0.7,
    lexical_weight: float = 0.3,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k chunks using RRF of vector and lexical results
    Supports standard (1:1) and weighted RRF
    """

    candidate_pool_size = max(top_k * 2, 60)

    vector_candidates, lexical_candidates = await asyncio.gather(
        vector_retrieval(
            session=session,
            query=query,
            top_k=candidate_pool_size,
            query_vector=query_vector,
        ),
        lexical_retrieval(
            session=session,
            query=query,
            top_k=candidate_pool_size,
        ),
    )

    return reciprocal_rank_fusion(
        vector_candidates=vector_candidates,
        lexical_candidates=lexical_candidates,
        rrf_k=rrf_k,
        top_k=top_k,
        rrf_mode=rrf_mode,
        vector_weight=vector_weight,
        lexical_weight=lexical_weight,
    )


async def retrieve(
    session: AsyncSession,
    query: str,
    top_k: int = 10,
    mode: Literal["vector", "lexical", "hybrid"] = "hybrid",
    rrf_k: int = 60,
    query_vector: list[float] | None = None,
    rerank: bool = False,
    rerank_threshold: float | None = None,
    rerank_candidate_k: int | None = None,
    rrf_mode: Literal["standard", "weighted"] = "standard",
    vector_weight: float = 0.7,
    lexical_weight: float = 0.3,
) -> list[RetrievedChunk]:

    candidate_k = (rerank_candidate_k or max(top_k * 3, 30)) if rerank else top_k

    if mode == "vector":
        candidates = await vector_retrieval(
            session=session,
            query=query,
            top_k=candidate_k,
            query_vector=query_vector,
        )
    elif mode == "lexical":
        candidates = await lexical_retrieval(
            session=session,
            query=query,
            top_k=candidate_k,
        )
    elif mode == "hybrid":
        candidates = await hybrid_retrieval(
            session=session,
            query=query,
            top_k=candidate_k,
            rrf_k=rrf_k,
            query_vector=query_vector,
            rrf_mode=rrf_mode,
            vector_weight=vector_weight,
            lexical_weight=lexical_weight,
        )
    else:
        raise ValueError(
            f"Unknown retrieval mode: {mode!r}. Must be 'vector', 'lexical', or 'hybrid'."
        )

    if not rerank:
        return candidates

    return await reranker.rerank(
        query=query,
        chunks=candidates,
        top_k=top_k,
        threshold=rerank_threshold,
    )


def aggregate_document_scores(
    chunks: list[RetrievedChunk],
) -> dict[str, float]:
    """
    Aggregate chunk scores to document level for BEIR evaluation.
    Maps: {str(document_id): max_score}
    """
    doc_scores: dict[str, float] = {}
    for chunk in chunks:
        doc_id_str = str(chunk.document_id)
        if doc_id_str not in doc_scores or chunk.score > doc_scores[doc_id_str]:
            doc_scores[doc_id_str] = chunk.score
    return doc_scores
