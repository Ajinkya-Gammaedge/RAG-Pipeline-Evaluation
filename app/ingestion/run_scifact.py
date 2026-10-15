import asyncio
import logging
from app.db import engine, session_local
from app.ingestion.pipeline import (
    DOCUMENT_BATCH_SIZE,
    ingest_document_batch,
)
from app.embeddings.sentence_transformer import embedder
from scripts.download_scifact import load_scifact_corpus
from app.core import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    corpus = load_scifact_corpus()

    documents = [
        {
            "id": int(document_id),
            "title": record.get("title", ""),
            "text": record.get("text", ""),
        }
        for document_id, record in corpus.items()
    ]

    seed_limit = settings.seed_max_docs
    if seed_limit > 0:
        logger.info(
            "SEED_MAX_DOCS is set to %d. Seeding a subset of %d documents.",
            seed_limit,
            seed_limit,
        )
        documents = documents[:seed_limit]

    logger.info(
        "Ingestion config: chunk_size=%d, chunk_overlap=%d, doc_batch=%d, embed_batch=%d",
        settings.chunk_size,
        settings.chunk_overlap,
        DOCUMENT_BATCH_SIZE,
        settings.embedding_batch_size,
    )

    totals = {
        "documents": 0,
        "chunks": 0,
        "embedded": 0,
        "unchanged": 0,
        "deleted": 0,
    }

    try:
        for start in range(0, len(documents), DOCUMENT_BATCH_SIZE):
            batch = documents[start : start + DOCUMENT_BATCH_SIZE]

            stats = await ingest_document_batch(
                session_factory=session_local,
                embedder=embedder,
                documents=batch,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

            for key, value in stats.items():
                totals[key] += value

            logger.info(
                "Progress: %d/%d documents | batch=%s",
                min(start + len(batch), len(documents)),
                len(documents),
                stats,
            )

        logger.info("Ingestion completed: %s", totals)

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
