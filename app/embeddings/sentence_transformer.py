import asyncio
import logging
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = settings.embedding_model):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(
                "Loading SentenceTransformer embedding model: %s ...", self.model_name
            )
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return await asyncio.to_thread(
            self._encode,
            texts,
        )

    async def embed_query(self, text: str) -> list[float]:
        vectors = await asyncio.to_thread(
            self._encode,
            [text],
        )
        return vectors[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vectors.tolist()


embedder = SentenceTransformerEmbedder()
