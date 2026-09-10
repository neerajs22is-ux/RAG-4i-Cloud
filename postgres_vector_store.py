"""PostgreSQL/pgvector provider (Phase 2).

Conforms to the same `VectorStore` interface as `ChromaVectorStore`
(`build_index()` / `search()` / `get_status()`).

- Embeddings: same `all-MiniLM-L6-v2` provider, same 384-dim vectors.
- Schema: simple `chunks` table, `embedding vector(384)`, existing metadata.
- Writes are additive only (`ON CONFLICT (chunk_id) DO NOTHING`); the
  Chroma database is never touched by this module.
- Scores: pgvector cosine distance `<=>` (0..2) mapped to
  `score = 1.0 - distance`, so the preserved 0.3 relevance threshold
  behaves sensibly. Exact parity with Chroma scores is not required.
"""

TABLE = "chunks"
DIM = 384

SCHEMA_SQL = """CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  document_id TEXT,
  source_path TEXT,
  file_name TEXT,
  page INTEGER,
  content TEXT NOT NULL,
  embedding vector(384) NOT NULL,
  scope TEXT NOT NULL DEFAULT 'persistent',
  session_id TEXT
);
"""

# Additive migration for pre-scope tables: DEFAULT backfills existing
# rows to persistent, so legacy corpora map deterministically without
# any data rewrite.
SCOPE_MIGRATION_SQL = (
    "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "
    "scope TEXT DEFAULT 'persistent'; "
    "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS session_id TEXT;"
)

# Isolation predicate (one shape for both modes): with a NULL parameter
# the session disjunct is never true, yielding persistent-only reads.
# scope IS NULL covers any row predating the migration.
SCOPE_PREDICATE = (
    "(scope IS NULL OR scope = 'persistent' "
    "OR (scope = 'session' AND session_id = %s))"
)


class _Doc:
    """Minimal Document stand-in (page_content + metadata, like LangChain)."""

    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = dict(metadata or {})
        self.id = None


class PostgresVectorStore:
    """pgvector implementation of the VectorStore interface."""

    def __init__(self, dsn=None, config=None, embedding_provider=None,
                 embedding_function=None, table=TABLE):
        if dsn is None and config is not None:
            dsn = config.postgres_dsn()
        self.dsn = dsn or ""
        self.table = table
        self._provider = embedding_provider
        self._embedding_function = embedding_function
        self._config = config

    def _embedding(self):
        if self._embedding_function is not None:
            return self._embedding_function
        if self._provider is not None:
            return self._provider.get_embedding_function()
        from embeddings import get_embedding_provider

        self._provider = get_embedding_provider(self._config)
        return self._provider.get_embedding_function()

    def _connect(self):
        import psycopg2
        from pgvector.psycopg2 import register_vector

        conn = psycopg2.connect(self.dsn)
        try:
            register_vector(conn)
        except Exception:
            pass
        return conn

    def ensure_schema(self):
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(SCHEMA_SQL)
                    cur.execute(SCOPE_MIGRATION_SQL.format(table=self.table))
        finally:
            conn.close()

    @staticmethod
    def _missing_scope_column(exc) -> bool:
        """True only for a pre-migration table lacking scope columns."""
        if getattr(exc, "pgcode", "") == "42703":
            return True
        text = str(exc)
        return "does not exist" in text and (
            "scope" in text or "session_id" in text)

    def build_index(self, chunks, session_id=None) -> int:
        """Insert chunks additively; existing rows kept (conflicts ignored).

        Scope is stamped from session_id (None = persistent) via
        document_scope.resolve_chunk_scope; contradictory or session-less
        session writes raise ValueError.
        """
        from document_scope import resolve_chunk_scope

        chunks = list(chunks or [])
        if not chunks:
            return 0
        self.ensure_schema()
        # Validate scope upfront: loud failure before any embedding work.
        resolved = [resolve_chunk_scope(
            getattr(chunk, "metadata", {}) or {}, session_id)
            for chunk in chunks]
        texts = [getattr(c, "page_content", "") or "" for c in chunks]
        vectors = self._embedding().embed_documents(texts)
        inserted = 0
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    for chunk, vec, (scope, sid) in zip(
                            chunks, vectors, resolved):
                        meta = getattr(chunk, "metadata", {}) or {}
                        page = meta.get("page")
                        if not isinstance(page, int):
                            page = None
                        cur.execute(
                            f"INSERT INTO {self.table} "
                            "(chunk_id, document_id, source_path, file_name,"
                            " page, content, embedding, scope, session_id) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                            "ON CONFLICT (chunk_id) DO NOTHING",
                            (meta.get("chunk_id"), meta.get("document_id"),
                             meta.get("source_path") or meta.get("source"),
                             meta.get("file_name"), page,
                             getattr(chunk, "page_content", ""), vec,
                             scope, sid),
                        )
                        inserted += cur.rowcount or 0
        finally:
            conn.close()
        return inserted

    def _read_with_scope_retry(self, fn):
        """Run a scoped read; migrate once and retry on pre-scope tables."""
        try:
            return fn()
        except Exception as e:
            if not self._missing_scope_column(e):
                raise
            self.ensure_schema()
            return fn()

    def search(self, query: str, k: int = 5, session_id=None):
        """pgvector cosine search; returns [(doc, score)] with score in [-1,1].

        session_id None -> persistent-only; a valid id adds that session's
        rows (never another session's). Malformed IDs raise ValueError.
        """
        if session_id is not None:
            from document_scope import validate_session_id
            validate_session_id(session_id)
        qvec = self._embedding().embed_query(query)

        def _run():
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT content, document_id, source_path, file_name,"
                        f" page, chunk_id, embedding <=> %s::vector AS dist "
                        f"FROM {self.table} WHERE {SCOPE_PREDICATE} "
                        f"ORDER BY dist ASC LIMIT %s",
                        (qvec, session_id, k),
                    )
                    rows = cur.fetchall()
            finally:
                conn.close()
            return rows

        rows = self._read_with_scope_retry(_run)
        out = []
        for content, doc_id, src, fname, page, cid, dist in rows:
            try:
                score = 1.0 - float(dist)
            except (TypeError, ValueError):
                score = None
            out.append((_Doc(content, {
                "source_path": src, "source": src, "file_name": fname,
                "page": page if isinstance(page, int) else None,
                "document_id": doc_id, "chunk_id": cid,
            }), score))
        return out

    def list_sources(self, limit: int = 50, session_id=None):
        """Distinct sources with a representative document_id each."""
        if session_id is not None:
            from document_scope import validate_session_id
            validate_session_id(session_id)
        try:
            conn = self._connect()
        except Exception:
            return []

        def _run():
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT file_name, MIN(document_id) FROM {self.table} "
                        f"WHERE file_name IS NOT NULL AND {SCOPE_PREDICATE} "
                        f"GROUP BY 1 ORDER BY 1 "
                        f"LIMIT %s",
                        (session_id, limit))
                    return [{"file_name": r[0], "document_id": r[1]}
                            for r in cur.fetchall() if r[0]]

        try:
            try:
                return _run()
            except Exception as e:
                if not self._missing_scope_column(e):
                    return []
                self.ensure_schema()
                return _run()
        except Exception:
            return []
        finally:
            conn.close()

    def delete_by_document(self, document_id: str) -> int:
        """Delete a document's chunks. Returns removed row count."""
        if not document_id:
            return 0
        try:
            conn = self._connect()
        except Exception:
            return 0
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {self.table} WHERE document_id = %s",
                        (document_id,))
                    return cur.rowcount or 0
        except Exception:
            return 0
        finally:
            conn.close()

    def chunks_for_source(self, file_name: str, limit: int = 8,
                            session_id=None):
        """Same-file chunks (best-effort; [] when unavailable)."""
        if session_id is not None:
            from document_scope import validate_session_id
            validate_session_id(session_id)
        try:
            conn = self._connect()
        except Exception:
            return []

        def _run():
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT content, document_id, source_path, file_name,"
                        f" page, chunk_id FROM {self.table} "
                        f"WHERE file_name = %s AND {SCOPE_PREDICATE} "
                        f"LIMIT %s",
                        (file_name, session_id, limit))
                    return cur.fetchall()

        try:
            try:
                rows = _run()
            except Exception as e:
                if not self._missing_scope_column(e):
                    return []
                self.ensure_schema()
                rows = _run()
        except Exception:
            return []
        finally:
            conn.close()
        return [(_Doc(content, {
            "source_path": src, "source": src, "file_name": fname,
            "page": page if isinstance(page, int) else None,
            "document_id": doc_id, "chunk_id": cid,
        }), None) for content, doc_id, src, fname, page, cid in rows]

    def get_status(self) -> dict:
        status = {"provider": "postgres", "ready": False, "exists": False,
                  "persist_directory": None, "table": self.table,
                  "chunk_count": None, "document_count": None}
        try:
            conn = self._connect()
        except Exception as e:
            status["error"] = f"cannot connect: {e}"
            return status
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT to_regclass(%s)", (self.table,))
                    if cur.fetchone()[0] is None:
                        return status
                    status["exists"] = True
                    cur.execute(f"SELECT count(*) FROM {self.table}")
                    n = cur.fetchone()[0]
                    status["chunk_count"] = int(n)
                    cur.execute(
                        f"SELECT count(*) FROM (SELECT DISTINCT COALESCE("
                        f"document_id, source_path, file_name) FROM {self.table}"
                        f") d")
                    status["document_count"] = int(cur.fetchone()[0])
                    status["ready"] = status["chunk_count"] > 0
        except Exception as e:
            status["error"] = str(e)
            status["ready"] = False
        finally:
            conn.close()
        return status

    def is_ready(self) -> bool:
        try:
            return bool(self.get_status().get("ready", False))
        except Exception:
            return False


def get_postgres_store(config=None, embedding_provider=None,
                       dsn=None, table=TABLE):
    if dsn is None and config is not None:
        dsn = config.postgres_dsn()
    if embedding_provider is None and config is not None:
        from embeddings import get_embedding_provider

        embedding_provider = get_embedding_provider(config)
    return PostgresVectorStore(dsn=dsn, config=config,
                               embedding_provider=embedding_provider,
                               table=table)
