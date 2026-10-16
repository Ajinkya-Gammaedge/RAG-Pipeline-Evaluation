import asyncio
import logging
from typing import Sequence
import numpy as np
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    clipped = np.clip(x, -250, 250)
    return 1.0 / (1.0 + np.exp(-clipped))


class SentenceTransformerReranker:
    def __init__(
        self,
        model_name: str = settings.reranker_model,
        max_length: int = settings.reranker_max_length,
        batch_size: int = settings.reranker_batch_size,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            logger.info("Loading CrossEncoder reranker model: %s ...", self.model_name)
            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    async def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int = 10,
        threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank retrieved chunks using sentence-transformers CrossEncoder
        Scores are sigmoid-normalized into [0.0, 1.0] and filtered strictly by threshold
        """
        if not chunks:
            return []

        return await asyncio.to_thread(
            self._rerank_sync,
            query=query,
            chunks=list(chunks),
            top_k=top_k,
            threshold=threshold,
        )

    def _rerank_sync(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
        threshold: float | None,
    ) -> list[RetrievedChunk]:
        pairs = [[query, chunk.content] for chunk in chunks]

        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # If single item or scalar, ensure 1D array
        if np.isscalar(raw_scores):
            raw_scores = np.array([raw_scores])

        # calibrate logits to [0.0, 1.0] probabilities
        calibrated_scores = _sigmoid(raw_scores)

        scored_chunks = [
            (chunk, float(score)) for chunk, score in zip(chunks, calibrated_scores)
        ]

        # sort descending by rerank score
        scored_chunks.sort(key=lambda item: item[1], reverse=True)

        # apply threshold filtering
        if threshold is not None:
            filtered = [item for item in scored_chunks if item[1] >= threshold]
            if not filtered and scored_chunks:
                logger.warning(
                    "All %d chunks scored below threshold %.3f (max=%.3f).",
                    len(scored_chunks),
                    threshold,
                    scored_chunks[0][1],
                )
            candidate_list = filtered
        else:
            candidate_list = scored_chunks

        top_candidates = candidate_list[:top_k]

        return [
            RetrievedChunk(
                id=chunk.id,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=round(score, 6),
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                metadata=chunk.metadata,
            )
            for chunk, score in top_candidates
        ]


reranker = SentenceTransformerReranker()
