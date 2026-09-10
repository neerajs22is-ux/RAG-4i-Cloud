"""Vector-store provider abstraction (Chroma default + PostgreSQL/pgvector).

No code outside this module (and `postgres_vector_store.py`) may import
or instantiate Chroma.
Use get_vector_store() + VectorStore.build_index/search/get_status.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Tuple


class VectorStore(ABC):
    """Provider-level operations (pgvector-ready interface, Chroma now).

    Scope (Phase 5B): read/write methods accept an optional session_id.
    None means persistent-only; a valid opaque id adds that session's
    rows (persistent OR session). Malformed IDs raise ValueError; reads
    never return another session's rows.
    """

    @abstractmethod
    def build_index(self, chunks, session_id=None) -> int:
        """Add chunks to the index without destroying existing data."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, k: int = 5, session_id=None):
        """Return list of (Document, score) tuples."""
        raise NotImplementedError

    @abstractmethod
    def get_status(self) -> dict:
        raise NotImplementedError

    def list_sources(self, limit: int = 50, session_id=None):
        """Distinct source filenames in the index (for suggestions).

        Default implementation reports unknown (providers override).
        Returns list of {"file_name": str}.
        """
        return []

    def delete_by_document(self, document_id: str) -> int:
        """Remove a document's chunks. Default: unsupported (0 removed)."""
        return 0

    def chunks_for_source(self, file_name: str, limit: int = 8,
                          session_id=None):
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


def _scope_where(session_id=None):
    """Chroma metadata filter for a read. None -> persistent-only.

    Raises ValueError on malformed session IDs (loud, never a silent
    unscoped read).
    """
    if session_id is None:
        return {"scope": "persistent"}
    from document_scope import SESSION, validate_session_id
    sid = validate_session_id(session_id)
    return {"$or": [{"scope": "persistent"},
                    {"$and": [{"scope": SESSION},
                              {"session_id": sid}]}]}


# Persist directories whose legacy rows were already normalized to
# scope="persistent" in this process (backfill is idempotent anyway).
_backfilled_dirs = set()


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

    def build_index(self, chunks, session_id=None) -> int:
        """Add chunks idempotently (same content -> same IDs, upserted).

        Scope is stamped from session_id (None = persistent) via
        document_scope.resolve_chunk_scope: contradictory or session-less
        session writes raise ValueError. chunk_id derivation excludes
        scope, so re-indexing never duplicates.
        """
        from langchain_community.vectorstores import Chroma

        from document_scope import resolve_chunk_scope

        chunks = list(chunks or [])
        if not chunks:
            return 0
        stamped = []
        for chunk in chunks:
            scope, sid = resolve_chunk_scope(
                getattr(chunk, "metadata", {}) or {}, session_id)
            meta = dict(getattr(chunk, "metadata", {}) or {})
            meta["scope"] = scope
            if sid is None:
                meta.pop("session_id", None)
            else:
                meta["session_id"] = sid
            chunk.metadata = meta
            stamped.append(chunk)
        chunks = stamped
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

    def _ensure_persistent_scope(self, db) -> bool:
        """Normalize legacy rows (missing/invalid scope) to persistent.

        Chroma metadata filters cannot match missing keys, so pre-scope
        rows would vanish from filtered reads. This metadata-only backfill
        (no vectors touched) maps them deterministically to persistent
        scope; session rows are never altered. Runs once per directory
        per process; True on success (or nothing to do).
        """
        if self.persist_directory in _backfilled_dirs:
            return True
        import logging

        try:
            try:
                got = db._collection.get(include=["metadatas"])
            except Exception:
                got = db.get(include=["metadatas"])
            if not isinstance(got, dict):
                _backfilled_dirs.add(self.persist_directory)
                return True
            ids = got.get("ids", []) or []
            metas = got.get("metadatas", []) or []
            fix_ids, fix_metas = [], []
            dropped_sids = 0
            for i, m in zip(ids, metas):
                if not isinstance(m, dict):
                    continue
                scope = m.get("scope")
                sid = m.get("session_id")
                if scope == "persistent" and not sid:
                    continue
                if scope == "session" and isinstance(sid, str) and sid:
                    from document_scope import is_valid_session_id
                    if is_valid_session_id(sid):
                        continue
                clean = {k: v for k, v in m.items() if k != "session_id"}
                clean["scope"] = "persistent"
                if "session_id" in m:
                    dropped_sids += 1
                fix_ids.append(i)
                fix_metas.append(clean)
            if fix_ids:
                db._collection.update(ids=list(fix_ids), metadatas=fix_metas)
                logging.getLogger(__name__).info(
                    "scope backfill: %d rows normalized to persistent "
                    "(%d stray session_id dropped) in %s",
                    len(fix_ids), dropped_sids, self.persist_directory)
            _backfilled_dirs.add(self.persist_directory)
            return True
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Scope backfill failed for %s: %s",
                self.persist_directory, e)
            return False

    def _scope_filter(self, db, session_id=None):
        """Where-clause for a scoped read, or None for legacy fallback.

        Raises RuntimeError for session-bound reads when the backfill
        failed (refusing to filter un-migrated data rather than leaking
        or hiding rows). Unbound reads degrade to the exact 4.12
        unfiltered behavior with a warning.
        """
        import logging

        if self._ensure_persistent_scope(db):
            return _scope_where(session_id)
        if session_id is not None:
            raise RuntimeError(
                "Cannot run session-scoped search: scope metadata backfill "
                f"failed for {self.persist_directory}.")
        logging.getLogger(__name__).warning(
            "Scope backfill failed for %s; using legacy unfiltered read.",
            self.persist_directory)
        return None

    def search(self, query: str, k: int = 5, session_id=None):
        from document_scope import validate_session_id
        if session_id is not None:
            validate_session_id(session_id)
        db = self._load_existing()
        where = self._scope_filter(db, session_id)
        if where is None:
            return db.similarity_search_with_relevance_scores(query, k=k)
        return db.similarity_search_with_relevance_scores(
            query, k=k, filter=where)

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

    def list_sources(self, limit: int = 50, session_id=None):
        """Distinct sources (best-effort; [] when unavailable).

        Returns [{"file_name": str, "document_id": str|None}] — the
        document_id powers the delete flow. Unbound reads see persistent
        sources only; bound reads add the current session's sources.
        """
        if session_id is not None:
            from document_scope import validate_session_id
            validate_session_id(session_id)
        try:
            db = self._load_existing()
            where = self._scope_filter(db, session_id)
            try:
                got = db.get(where=where, limit=limit) if where is not None \
                    else db.get(limit=limit)
                metadatas = got.get("metadatas") if isinstance(got, dict) else None
            except Exception:
                metadatas = None
            if metadatas is None:
                kwargs = {"limit": limit, "include": ["metadatas"]}
                if where is not None:
                    kwargs["where"] = where
                got = db._collection.get(**kwargs)
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

    def chunks_for_source(self, file_name: str, limit: int = 8,
                            session_id=None):
        """Same-file chunks (best-effort; [] when unavailable)."""
        from langchain_core.documents import Document

        if session_id is not None:
            from document_scope import validate_session_id
            validate_session_id(session_id)
        try:
            db = self._load_existing()
            base = self._scope_filter(db, session_id)
            if base is None:
                where = {"file_name": file_name}
            else:
                where = {"$and": [{"file_name": file_name}, base]}
            try:
                got = db.get(where=where, limit=limit)
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
