import asyncio

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return await asyncio.to_thread(
            self._encode,
            texts,
        )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vectors.tolist()


embedder = SentenceTransformerEmbedder("BAAI/bge-base-en-v1.5")
