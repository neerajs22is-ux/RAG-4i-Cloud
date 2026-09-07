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
  embedding vector(384) NOT NULL
);
"""


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
        finally:
            conn.close()

    def build_index(self, chunks) -> int:
        """Insert chunks additively; existing rows kept (conflicts ignored)."""
        chunks = list(chunks or [])
        if not chunks:
            return 0
        self.ensure_schema()
        texts = [getattr(c, "page_content", "") or "" for c in chunks]
        vectors = self._embedding().embed_documents(texts)
        inserted = 0
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    for chunk, vec in zip(chunks, vectors):
                        meta = getattr(chunk, "metadata", {}) or {}
                        page = meta.get("page")
                        if not isinstance(page, int):
                            page = None
                        cur.execute(
                            f"INSERT INTO {self.table} "
                            "(chunk_id, document_id, source_path, file_name,"
                            " page, content, embedding) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s) "
                            "ON CONFLICT (chunk_id) DO NOTHING",
                            (meta.get("chunk_id"), meta.get("document_id"),
                             meta.get("source_path") or meta.get("source"),
                             meta.get("file_name"), page,
                             getattr(chunk, "page_content", ""), vec),
                        )
                        inserted += cur.rowcount or 0
        finally:
            conn.close()
        return inserted

    def search(self, query: str, k: int = 5):
        """pgvector cosine search; returns [(doc, score)] with score in [-1,1]."""
        qvec = self._embedding().embed_query(query)
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content, document_id, source_path, file_name,"
                    f" page, chunk_id, embedding <=> %s::vector AS dist "
                    f"FROM {self.table} ORDER BY dist ASC LIMIT %s",
                    (qvec, k),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
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

    def list_sources(self, limit: int = 50):
        """Distinct source filenames (best-effort; [] when unavailable)."""
        try:
            conn = self._connect()
        except Exception:
            return []
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT DISTINCT file_name FROM {self.table} "
                        f"WHERE file_name IS NOT NULL ORDER BY 1 LIMIT %s",
                        (limit,))
                    return [{"file_name": r[0]} for r in cur.fetchall() if r[0]]
        except Exception:
            return []
        finally:
            conn.close()

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
