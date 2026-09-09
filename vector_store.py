"""Vector-store provider abstraction (Chroma default + PostgreSQL/pgvector).

No code outside this module (and `postgres_vector_store.py`) may import
or instantiate Chroma.
Use get_vector_store() + VectorStore.build_index/search/get_status.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Tuple


class VectorStore(ABC):
    """Provider-level operations (pgvector-ready interface, Chroma now)."""

    @abstractmethod
    def build_index(self, chunks) -> int:
        """Add chunks to the index without destroying existing data."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, k: int = 5):
        """Return list of (Document, score) tuples."""
        raise NotImplementedError

    @abstractmethod
    def get_status(self) -> dict:
        raise NotImplementedError

    def list_sources(self, limit: int = 50):
        """Distinct source filenames in the index (for suggestions).

        Default implementation reports unknown (providers override).
        Returns list of {"file_name": str}.
        """
        return []

    def delete_by_document(self, document_id: str) -> int:
        """Remove a document's chunks. Default: unsupported (0 removed)."""
        return 0

    def chunks_for_source(self, file_name: str, limit: int = 8):
        """Same-file chunks as admissible file evidence for overviews.

        Default: none (providers override). Returns [(doc, None)] with
        real metadata; scores stay None (file-selected, not ranked).
        """
        return []

    def is_ready(self) -> bool:
        try:
            return bool(self.get_status().get("ready", False))
        except Exception:
            return False


class ChromaVectorStore(VectorStore):
    """Chroma implementation. Never deletes the existing database."""

    def __init__(self, persist_directory: str = "chroma_db",
                 embedding_provider=None, embedding_function=None):
        self.persist_directory = os.path.abspath(persist_directory)
        self._provider = embedding_provider
        self._embedding_function = embedding_function

    def _embedding(self):
        if self._embedding_function is not None:
            return self._embedding_function
        if self._provider is not None:
            return self._provider.get_embedding_function()
        # Lazy default (preserves original model).
        from embeddings import get_embedding_provider

        self._provider = get_embedding_provider()
        return self._provider.get_embedding_function()

    def _load_existing(self):
        from langchain_community.vectorstores import Chroma

        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self._embedding(),
        )

    @staticmethod
    def _chunk_ids(chunks) -> list:
        """Stable content-derived IDs so re-indexing upserts idempotently."""
        import hashlib

        ids = []
        for i, chunk in enumerate(chunks):
            meta = getattr(chunk, "metadata", {}) or {}
            key = meta.get("chunk_id") or "%s:%s:%s" % (
                meta.get("document_id", "?"),
                meta.get("start_index", i),
                getattr(chunk, "page_content", "") or "",
            )
            ids.append(hashlib.sha1(str(key).encode("utf-8")).hexdigest())
        return ids

    def build_index(self, chunks) -> int:
        """Add chunks idempotently (same content -> same IDs, upserted)."""
        from langchain_community.vectorstores import Chroma

        chunks = list(chunks or [])
        if not chunks:
            return 0
        ids = self._chunk_ids(chunks)
        if os.path.isdir(self.persist_directory) and os.listdir(self.persist_directory):
            db = self._load_existing()
            new_ids = db.add_documents(documents=chunks, ids=ids)
            return len(new_ids) if new_ids else len(chunks)
        else:
            os.makedirs(self.persist_directory, exist_ok=True)
            Chroma.from_documents(
                documents=chunks,
                embedding=self._embedding(),
                persist_directory=self.persist_directory,
                ids=ids,
            )
            return len(chunks)

    def delete_by_document(self, document_id: str) -> int:
        """Remove all chunks for a document_id. Returns removed count."""
        if not document_id:
            return 0
        try:
            db = self._load_existing()
            existing = db._collection.get(where={"document_id": document_id},
                                          include=[])
            ids = (existing or {}).get("ids", [])
            if ids:
                db._collection.delete(ids=list(ids))
            return len(ids)
        except Exception:
            return 0

    def search(self, query: str, k: int = 5):
        db = self._load_existing()
        return db.similarity_search_with_relevance_scores(query, k=k)

    def get_status(self) -> dict:
        exists = os.path.isdir(self.persist_directory) and bool(
            os.listdir(self.persist_directory)
        )
        chunk_count = None
        document_count = None
        ready = False
        error = None
        if exists:
            try:
                db = self._load_existing()
                # Chroma collection count (version-tolerant).
                try:
                    chunk_count = db._collection.count()
                except Exception:
                    chunk_count = None
                # Document count: distinct source documents (no new deps).
                try:
                    if chunk_count == 0:
                        document_count = 0
                    else:
                        metadatas = None
                        try:
                            got = db.get()
                            if isinstance(got, dict):
                                metadatas = got.get("metadatas")
                        except Exception:
                            metadatas = None
                        if metadatas is None:
                            try:
                                got = db._collection.get(include=["metadatas"])
                                if isinstance(got, dict):
                                    metadatas = got.get("metadatas")
                            except Exception:
                                metadatas = None
                        if metadatas is not None:
                            distinct = set()
                            for m in metadatas:
                                if not isinstance(m, dict):
                                    continue
                                key = (
                                    m.get("document_id")
                                    or m.get("source_path")
                                    or m.get("source")
                                    or m.get("file_name")
                                )
                                if key:
                                    distinct.add(key)
                            document_count = len(distinct)
                except Exception:
                    # Document count is best-effort; never break status.
                    pass
                ready = True if chunk_count is None else chunk_count > 0
                if chunk_count == 0:
                    ready = False
            except Exception as e:
                error = str(e)
                ready = False
        status = {
            "ready": ready,
            "exists": exists,
            "persist_directory": self.persist_directory,
            "chunk_count": chunk_count,
            "document_count": document_count,
        }
        if error:
            status["error"] = error
        return status

    def list_sources(self, limit: int = 50):
        """Distinct sources (best-effort; [] when unavailable).

        Returns [{"file_name": str, "document_id": str|None}] — the
        document_id powers the delete flow.
        """
        try:
            db = self._load_existing()
            try:
                got = db.get(limit=limit)
                metadatas = got.get("metadatas") if isinstance(got, dict) else None
            except Exception:
                metadatas = None
            if metadatas is None:
                got = db._collection.get(limit=limit, include=["metadatas"])
                metadatas = got.get("metadatas") if isinstance(got, dict) else []
            seen = {}
            for m in metadatas or []:
                name = (m or {}).get("file_name")
                if name and name not in seen:
                    seen[name] = (m or {}).get("document_id")
            return [{"file_name": n, "document_id": seen[n]}
                    for n in list(seen)[:limit]]
        except Exception:
            return []

    def chunks_for_source(self, file_name: str, limit: int = 8):
        """Same-file chunks (best-effort; [] when unavailable)."""
        from langchain_core.documents import Document

        try:
            db = self._load_existing()
            try:
                got = db.get(where={"file_name": file_name},
                             limit=limit)
                docs = got.get("documents") if isinstance(got, dict) else None
                metas = got.get("metadatas") if isinstance(got, dict) else None
            except Exception:
                docs, metas = None, None
            if docs is None:
                got = db._collection.get(where={"file_name": file_name},
                                         limit=limit,
                                         include=["documents", "metadatas"])
                docs = got.get("documents", [])
                metas = got.get("metadatas", [])
            out = []
            for content, meta in zip(docs or [], metas or []):
                out.append((Document(page_content=content or "",
                                     metadata=dict(meta or {})), None))
            return out
        except Exception:
            return []


def get_vector_store(config=None, embedding_provider=None) -> VectorStore:
    """Factory dispatched by config: VECTOR_STORE=chroma (default) or postgres."""
    name = "chroma"
    if config is not None:
        name = (getattr(config, "vector_store", name) or name).lower()
    if name in ("postgres", "postgresql", "pgvector", "pg"):
        from postgres_vector_store import get_postgres_store

        return get_postgres_store(config, embedding_provider)
    persist = "chroma_db"
    if config is not None:
        persist = getattr(config, "chroma_path", persist)
    if embedding_provider is None and config is not None:
        from embeddings import get_embedding_provider

        embedding_provider = get_embedding_provider(config)
    return ChromaVectorStore(
        persist_directory=persist, embedding_provider=embedding_provider
    )
