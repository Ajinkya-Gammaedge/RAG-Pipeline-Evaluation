from dataclasses import dataclass
from typing import Any


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
