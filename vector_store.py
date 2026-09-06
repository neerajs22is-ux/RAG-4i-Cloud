"""Vector-store provider abstraction (Phase 1: Chroma only).

No code outside this module may import or instantiate Chroma.
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

    def build_index(self, chunks) -> int:
        """Append chunks to existing index (append-only). Returns added count."""
        from langchain_community.vectorstores import Chroma

        if not chunks:
            return 0
        if os.path.isdir(self.persist_directory) and os.listdir(self.persist_directory):
            db = self._load_existing()
            # add_documents preserves existing data (duplicates acceptable P1).
            ids = db.add_documents(documents=chunks)
            return len(ids) if ids else len(chunks)
        else:
            os.makedirs(self.persist_directory, exist_ok=True)
            Chroma.from_documents(
                documents=chunks,
                embedding=self._embedding(),
                persist_directory=self.persist_directory,
            )
            return len(chunks)

    def search(self, query: str, k: int = 5):
        db = self._load_existing()
        return db.similarity_search_with_relevance_scores(query, k=k)

    def get_status(self) -> dict:
        exists = os.path.isdir(self.persist_directory) and bool(
            os.listdir(self.persist_directory)
        )
        chunk_count = None
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
        }
        if error:
            status["error"] = error
        return status


def get_vector_store(config=None, embedding_provider=None) -> VectorStore:
    """Factory (chroma only in Phase 1)."""
    persist = "chroma_db"
    if config is not None:
        persist = getattr(config, "chroma_path", persist)
    if embedding_provider is None and config is not None:
        from embeddings import get_embedding_provider

        embedding_provider = get_embedding_provider(config)
    return ChromaVectorStore(
        persist_directory=persist, embedding_provider=embedding_provider
    )
