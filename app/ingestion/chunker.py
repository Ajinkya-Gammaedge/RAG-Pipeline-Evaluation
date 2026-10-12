import hashlib
import re
from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    document_id: int
    chunk_id: int
    content: str
    content_hash: str
    start_offset: int
    end_offset: int
    metadata: dict[str, Any]


default_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator=True,
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def prepare_document_chunks(
    document_id: int,
    title: str,
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[PreparedChunk]:
    combined_text = "\n\n".join(
        part.strip() for part in (title, text) if part and part.strip()
    )

    normalized_text = normalize_text(combined_text)

    if not normalized_text:
        return []

    if chunk_size == 1000 and chunk_overlap == 150:
        splitter = default_splitter
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    texts = splitter.split_text(normalized_text)

    chunks: list[PreparedChunk] = []
    current_search_idx = 0

    doc_metadata = {
        "title": title or "",
        "source": "scifact",
    }

    for index, raw_chunk in enumerate(texts):
        chunk = raw_chunk.strip()
        if not chunk:
            continue

        start_offset = normalized_text.find(chunk, current_search_idx)
        if start_offset == -1:
            start_offset = normalized_text.find(chunk)
            if start_offset == -1:
                start_offset = 0

        end_offset = start_offset + len(chunk)
        current_search_idx = max(0, start_offset + 1)

        chunks.append(
            PreparedChunk(
                document_id=document_id,
                chunk_id=index,
                content=chunk,
                content_hash=hash_content(chunk),
                start_offset=start_offset,
                end_offset=end_offset,
                metadata=doc_metadata,
            )
        )

    return chunks
