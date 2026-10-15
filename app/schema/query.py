from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    query: str = Field(
        min_length=1,
        max_length=200,
        description="Search query for scientific document retrieval",
        examples=["0-dimensional biomaterials show inductive properties"],
    )

    retrieval_mode: Literal["vector", "lexical", "hybrid"] = Field(
        default="hybrid",
        description="Retrieval strategy to use",
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )

    rrf_k: int = Field(
        default=60,
        ge=1,
        le=200,
        description="Reciprocal Rank Fusion constant, used in hybrid mode",
    )

    rrf_mode: Literal["standard", "weighted"] = Field(
        default="standard",
        description="RRF strategy: 'standard' for 1:1 equal weights, 'weighted' for custom weights",
    )

    vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Vector search weight used when rrf_mode='weighted'",
    )

    lexical_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Lexical search weight used when rrf_mode='weighted'",
    )

    rerank: bool = Field(
        default=False,
        description="Whether to apply cross-encoder reranking to retrieved candidates",
    )

    rerank_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score threshold [0.0, 1.0] to retain chunks",
    )


class SourceChunk(BaseModel):
    chunk_id: int
    document_id: int
    text: str = Field(description="Content of the retrieved text passage")
    score: float
    start_offset: int
    end_offset: int


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(
        default=0, description="Tokens consumed by the prompt and context passages"
    )
    completion_tokens: int = Field(
        default=0, description="Tokens generated in the response"
    )
    total_tokens: int = Field(
        default=0, description="Total tokens burned (prompt + completion)"
    )


class LLMMetadata(BaseModel):
    model: str = Field(description="Name/identifier of the LLM used")
    token_usage: TokenUsage = Field(description="Detailed token consumption")


class QueryResponse(BaseModel):
    answer: str = Field(
        description="LLM-generated answer grounded in retrieved context"
    )
    sources: list[SourceChunk] = Field(
        description="Chunks used as context for the answer"
    )
    retrieval_mode: str = Field(description="Retrieval strategy that was used")
    llm_metadata: LLMMetadata = Field(
        description="Metadata about LLM and token consumption"
    )
    latency_ms: float = Field(description="Total end-to-end latency in milliseconds")
