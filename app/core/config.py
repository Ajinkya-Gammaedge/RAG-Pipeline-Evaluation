from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    ollama_url: str
    ollama_model: str
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: int = 768
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 32
    reranker_max_length: int = 512
    seed_max_docs: int = 0
    document_batch_size: int = 100
    embedding_batch_size: int = 64
    chunk_size: int = 1000
    chunk_overlap: int = 150

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
