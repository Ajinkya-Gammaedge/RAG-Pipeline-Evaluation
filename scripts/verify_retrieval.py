import asyncio
import logging
from sqlalchemy import text

from app.db import session_local
from app.embeddings.sentence_tranformer import embedder
from app.retrieval.retrieval import retrieve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    test_query = "0-dimensional biomaterials show inductive properties"
    logger.info("Testing retrieval pipeline")
    logger.info("Query: %r", test_query)

    query_vector = await embedder.embed_query(test_query)

    async with session_local() as session:
        # 1. Vector retrieval
        logger.info("Vector Retrieval")
        vector_results = await retrieve(
            session=session,
            query=test_query,
            query_vector=query_vector,
            mode="vector",
            top_k=3,
        )

        for result in vector_results:
            logger.info(
                "[Vector] Doc %s | Chunk %s | Score %.4f | Content %s...",
                result.document_id,
                result.chunk_id,
                result.score,
                result.content[:70],
            )

        # 2. Lexical retrieval
        logger.info("Lexical Retrieval")
        lexical_results = await retrieve(
            session=session,
            query=test_query,
            mode="lexical",
            top_k=3,
        )

        for result in lexical_results:
            logger.info(
                "[Lexical] Doc %s | Chunk %s | Score %.4f | Content %s...",
                result.document_id,
                result.chunk_id,
                result.score,
                result.content[:70],
            )

        # 3. Hybrid RRF retrieval
        logger.info("Hybrid RRF Retrieval")
        hybrid_results = await retrieve(
            session=session,
            query=test_query,
            query_vector=query_vector,
            mode="hybrid",
            top_k=3,
            rrf_k=60,
        )

        for result in hybrid_results:
            logger.info(
                "[Hybrid] Doc %s | Chunk %s | Score %.4f | Content %s...",
                result.document_id,
                result.chunk_id,
                result.score,
                result.content[:70],
            )

        # 4. Explain vector query plan
        logger.info("Vector Search Query Plan (EXPLAIN ANALYZE)")

        vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"

        explain_stmt = text("""
            EXPLAIN (ANALYZE, BUFFERS)
            SELECT id, document_id, chunk_id,
                   embedding_vector <=> CAST(:query_vector AS vector) AS dist
            FROM document_chunks
            ORDER BY embedding_vector <=> CAST(:query_vector AS vector) ASC
            LIMIT 10;
        """)

        explain_result = await session.execute(
            explain_stmt,
            {"query_vector": vector_str},
        )

        for row in explain_result.all():
            logger.info("%s", row[0])


if __name__ == "__main__":
    asyncio.run(main())
