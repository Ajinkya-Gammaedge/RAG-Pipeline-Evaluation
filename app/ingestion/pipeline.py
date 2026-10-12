from collections.abc import Sequence

from sqlalchemy import delete, select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.document_chunk import DocumentChunk
from app.ingestion.chunker import (
    PreparedChunk,
    prepare_document_chunks,
)

DOCUMENT_BATCH_SIZE = 100
EMBEDDING_BATCH_SIZE = 64


async def embed_in_batches(embedder, chunks: Sequence[PreparedChunk]):
    """Return vectors in exactly the same order as the input chunks."""
    vectors = []

    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]

        batch_vectors = await embedder.embed_documents(
            [chunk.content for chunk in batch]
        )

        if len(batch_vectors) != len(batch):
            raise ValueError(
                "Embedding provider returned an incorrect number of vectors"
            )

        vectors.extend(batch_vectors)

    return vectors


async def ingest_document_batch(
    session_factory: async_sessionmaker[AsyncSession],
    embedder,
    documents: Sequence[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> dict[str, int]:
    """
    expected document shape:
        {
            "id": "document-id",
            "title": "Document title",
            "text": "Document body",
        }
    """
    if not documents:
        return {
            "documents": 0,
            "chunks": 0,
            "embedded": 0,
            "unchanged": 0,
            "deleted": 0,
        }

    document_ids = [int(doc["id"]) for doc in documents]

    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate document IDs in ingestion batch")

    chunks_by_document: dict[int, list[PreparedChunk]] = {}
    all_chunks: list[PreparedChunk] = []

    # Step 1: prepare chunks for every document
    for doc in documents:
        document_id = int(doc["id"])

        chunks = prepare_document_chunks(
            document_id=document_id,
            title=doc.get("title", ""),
            text=doc.get("text", ""),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # empty documents must be validated explicitly
        if not chunks:
            raise ValueError(
                f"Document {document_id!r} has no usable text; "
                "refusing to delete its existing chunks."
            )

        chunks_by_document[document_id] = chunks
        all_chunks.extend(chunks)

    # Step 2: read the existing chunk hashes
    async with session_factory() as session:
        result = await session.execute(
            select(
                DocumentChunk.document_id,
                DocumentChunk.chunk_id,
                DocumentChunk.content_hash,
            ).where(DocumentChunk.document_id.in_(document_ids))
        )

        existing_hashes = {
            (document_id, chunk_id): content_hash
            for document_id, chunk_id, content_hash in result.all()
        }

    # Step 3: only embed chunks that are new or have changed
    changed_chunks = [
        chunk
        for chunk in all_chunks
        if existing_hashes.get((chunk.document_id, chunk.chunk_id))
        != chunk.content_hash
    ]

    unchanged_count = len(all_chunks) - len(changed_chunks)

    # Step 4: embed before opening the write transaction
    vectors = await embed_in_batches(embedder, changed_chunks)

    if vectors:
        expected_dimension = DocumentChunk.embedding_vector.type.dim
        for vector in vectors:
            if len(vector) != expected_dimension:
                raise ValueError(
                    f"Expected {expected_dimension}-dimensional embedding, "
                    f"received {len(vector)} dimensions"
                )

    rows = [
        {
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "content_hash": chunk.content_hash,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "metadata": chunk.metadata,
            "embedding_vector": vector,
        }
        for chunk, vector in zip(changed_chunks, vectors)
    ]

    # Step 5: delete stale chunks and upsert current chunks atomically
    deleted_count = 0

    async with session_factory() as session:
        async with session.begin():
            # delete stale chunks for each successfully prepared document
            for document_id, chunks in chunks_by_document.items():
                current_ids = [chunk.chunk_id for chunk in chunks]

                stmt = delete(DocumentChunk).where(
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.chunk_id.not_in(current_ids),
                )

                result = await session.execute(stmt)
                deleted_count += result.rowcount or 0

            # bulk insert or update changed chunks
            if rows:
                stmt = pg_insert(DocumentChunk).values(rows)

                stmt = stmt.on_conflict_do_update(
                    index_elements=[
                        DocumentChunk.document_id,
                        DocumentChunk.chunk_id,
                    ],
                    set_={
                        "content": stmt.excluded.content,
                        "content_hash": stmt.excluded.content_hash,
                        "start_offset": stmt.excluded.start_offset,
                        "end_offset": stmt.excluded.end_offset,
                        "metadata": stmt.excluded.metadata,
                        "embedding_vector": stmt.excluded.embedding_vector,
                        "updated_at": func.now(),
                    },
                    where=(DocumentChunk.content_hash != stmt.excluded.content_hash),
                )

                await session.execute(stmt)

    return {
        "documents": len(documents),
        "chunks": len(all_chunks),
        "embedded": len(changed_chunks),
        "unchanged": unchanged_count,
        "deleted": deleted_count,
    }
