"""Vector storage: PostgreSQL pgvector and in-memory fallback for tests."""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import date
from typing import Any

from sqlalchemy import Date, Text, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.knowledge.embeddings import EMBEDDING_DIM
from app.schemas.knowledge import PolicyChunkRecord, PolicyDocumentMeta

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover
    Vector = None  # type: ignore[misc, assignment]


class Base(DeclarativeBase):
    pass


class PolicyDocumentRow(Base):
    __tablename__ = "policy_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    company_id: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    effective_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    effective_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)


class PolicyChunkRow(Base):
    __tablename__ = "policy_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    company_id: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)


class PolicyVectorStore(ABC):
    @abstractmethod
    def upsert_document(self, meta: PolicyDocumentMeta) -> None:
        ...

    @abstractmethod
    def upsert_chunks(
        self,
        company_id: str,
        document_id: str,
        chunks: list[tuple[int, str, dict[str, Any], list[float]]],
    ) -> None:
        ...

    @abstractmethod
    def search(
        self,
        *,
        company_id: str,
        query_embedding: list[float],
        top_k: int,
        include_superseded: bool = False,
    ) -> list[PolicyChunkRecord]:
        ...

    @abstractmethod
    def clear_company(self, company_id: str) -> None:
        ...


class InMemoryPolicyVectorStore(PolicyVectorStore):
    def __init__(self) -> None:
        self._documents: dict[tuple[str, str], PolicyDocumentMeta] = {}
        self._chunks: list[dict[str, Any]] = []

    def upsert_document(self, meta: PolicyDocumentMeta) -> None:
        self._documents[(meta.company_id, meta.document_id)] = meta

    def upsert_chunks(
        self,
        company_id: str,
        document_id: str,
        chunks: list[tuple[int, str, dict[str, Any], list[float]]],
    ) -> None:
        self._chunks = [
            c
            for c in self._chunks
            if not (c["company_id"] == company_id and c["document_id"] == document_id)
        ]
        for index, content, metadata, embedding in chunks:
            self._chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "company_id": company_id,
                    "document_id": document_id,
                    "chunk_index": index,
                    "content": content,
                    "metadata": metadata,
                    "embedding": embedding,
                }
            )

    def _is_active(self, meta: PolicyDocumentMeta) -> bool:
        if meta.status != "active":
            return False
        today = date.today()
        if meta.effective_from and meta.effective_from > today:
            return False
        if meta.effective_to and meta.effective_to < today:
            return False
        return True

    def search(
        self,
        *,
        company_id: str,
        query_embedding: list[float],
        top_k: int,
        include_superseded: bool = False,
    ) -> list[PolicyChunkRecord]:
        from app.knowledge.embeddings import cosine_similarity

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self._chunks:
            if row["company_id"] != company_id:
                continue
            meta = self._documents.get((company_id, row["document_id"]))
            if meta is None:
                continue
            if not include_superseded and not self._is_active(meta):
                continue
            sim = cosine_similarity(query_embedding, row["embedding"])
            scored.append((sim, row))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        out: list[PolicyChunkRecord] = []
        for sim, row in scored[:top_k]:
            meta = self._documents[(company_id, row["document_id"])]
            out.append(
                PolicyChunkRecord(
                    chunk_id=row["chunk_id"],
                    company_id=company_id,
                    document_id=row["document_id"],
                    chunk_index=row["chunk_index"],
                    content=row["content"],
                    metadata={
                        **row["metadata"],
                        "title": meta.title,
                        "category": meta.category,
                        "status": meta.status,
                    },
                    similarity=sim,
                )
            )
        return out

    def clear_company(self, company_id: str) -> None:
        self._documents = {
            k: v for k, v in self._documents.items() if k[0] != company_id
        }
        self._chunks = [c for c in self._chunks if c["company_id"] != company_id]


class PostgresPolicyVectorStore(PolicyVectorStore):
    def __init__(self, database_url: str) -> None:
        if Vector is None:
            raise RuntimeError("pgvector package is required for PostgreSQL storage.")
        self._engine = create_engine(database_url, future=True)

    def upsert_document(self, meta: PolicyDocumentMeta) -> None:
        with Session(self._engine) as session:
            session.execute(
                text(
                    """
                    INSERT INTO policy_documents (
                        id, company_id, document_id, title, category, version,
                        status, effective_from, effective_to, source_path
                    ) VALUES (
                        :id, :company_id, :document_id, :title, :category, :version,
                        :status, :effective_from, :effective_to, :source_path
                    )
                    ON CONFLICT (company_id, document_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        category = EXCLUDED.category,
                        version = EXCLUDED.version,
                        status = EXCLUDED.status,
                        effective_from = EXCLUDED.effective_from,
                        effective_to = EXCLUDED.effective_to,
                        source_path = EXCLUDED.source_path
                    """
                ),
                {
                    "id": uuid.uuid4(),
                    "company_id": meta.company_id,
                    "document_id": meta.document_id,
                    "title": meta.title,
                    "category": meta.category,
                    "version": meta.version,
                    "status": meta.status,
                    "effective_from": meta.effective_from,
                    "effective_to": meta.effective_to,
                    "source_path": meta.source_path,
                },
            )
            session.commit()

    def upsert_chunks(
        self,
        company_id: str,
        document_id: str,
        chunks: list[tuple[int, str, dict[str, Any], list[float]]],
    ) -> None:
        with Session(self._engine) as session:
            session.execute(
                text(
                    "DELETE FROM policy_chunks WHERE company_id = :company_id AND document_id = :document_id"
                ),
                {"company_id": company_id, "document_id": document_id},
            )
            for index, content, metadata, embedding in chunks:
                session.execute(
                    text(
                        """
                        INSERT INTO policy_chunks (
                            id, company_id, document_id, chunk_index, content, metadata, embedding
                        ) VALUES (
                            :id, :company_id, :document_id, :chunk_index, :content,
                            CAST(:metadata AS jsonb), CAST(:embedding AS vector)
                        )
                        """
                    ),
                    {
                        "id": uuid.uuid4(),
                        "company_id": company_id,
                        "document_id": document_id,
                        "chunk_index": index,
                        "content": content,
                        "metadata": json.dumps(metadata),
                        "embedding": _vector_literal(embedding),
                    },
                )
            session.commit()

    def search(
        self,
        *,
        company_id: str,
        query_embedding: list[float],
        top_k: int,
        include_superseded: bool = False,
    ) -> list[PolicyChunkRecord]:
        status_clause = "" if include_superseded else "AND d.status = 'active'"
        sql = f"""
            SELECT
                c.id::text AS chunk_id,
                c.company_id,
                c.document_id,
                c.chunk_index,
                c.content,
                c.metadata,
                d.title,
                d.category,
                d.status,
                1 - (c.embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM policy_chunks c
            JOIN policy_documents d
              ON d.company_id = c.company_id AND d.document_id = c.document_id
            WHERE c.company_id = :company_id
              {status_clause}
              AND (d.effective_from IS NULL OR d.effective_from <= CURRENT_DATE)
              AND (d.effective_to IS NULL OR d.effective_to >= CURRENT_DATE)
            ORDER BY c.embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """
        with Session(self._engine) as session:
            rows = session.execute(
                text(sql),
                {
                    "company_id": company_id,
                    "embedding": _vector_literal(query_embedding),
                    "top_k": top_k,
                },
            ).mappings()
            return [
                PolicyChunkRecord(
                    chunk_id=row["chunk_id"],
                    company_id=row["company_id"],
                    document_id=row["document_id"],
                    chunk_index=row["chunk_index"],
                    content=row["content"],
                    metadata={
                        **(row["metadata"] or {}),
                        "title": row["title"],
                        "category": row["category"],
                        "status": row["status"],
                    },
                    similarity=float(row["similarity"]),
                )
                for row in rows
            ]

    def clear_company(self, company_id: str) -> None:
        with Session(self._engine) as session:
            session.execute(
                text("DELETE FROM policy_chunks WHERE company_id = :company_id"),
                {"company_id": company_id},
            )
            session.execute(
                text("DELETE FROM policy_documents WHERE company_id = :company_id"),
                {"company_id": company_id},
            )
            session.commit()


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def build_vector_store(database_url: str | None) -> PolicyVectorStore:
    if database_url:
        return PostgresPolicyVectorStore(database_url)
    return InMemoryPolicyVectorStore()
