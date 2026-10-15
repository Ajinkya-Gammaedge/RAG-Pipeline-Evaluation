from datetime import datetime

from app.db import Base
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Integer,
    Text,
    UniqueConstraint,
    Index,
    Computed,
    func,
    String,
    BigInteger,
    text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from app.core import settings


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    document_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    chunk_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    start_offset: Mapped[int] = mapped_column(
        Integer,
        server_default="0",
        nullable=False,
    )

    end_offset: Mapped[int] = mapped_column(
        Integer,
        server_default="0",
        nullable=False,
    )

    chunk_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )

    embedding_vector: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dimension),
        nullable=False,
    )

    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(content, ''))",
            persisted=True,
        ),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_id",
            name="uq_document_chunks_document_id",
        ),
        CheckConstraint(
            "chunk_id >= 0",
            name="ck_document_chunks_chunk_id_nonnegative",
        ),
        CheckConstraint(
            "start_offset >= 0 AND end_offset >= start_offset",
            name="ck_document_chunks_valid_offsets",
        ),
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding_vector",
            postgresql_using="hnsw",
            postgresql_ops={
                "embedding_vector": "vector_cosine_ops",
            },
        ),
        Index(
            "ix_document_chunks_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
    )
