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

# HNSW retrieval index (Step 3; infrastructure/indexing only).
# The retrieval query uses cosine distance (`embedding <=> %s::vector`),
# so the operator class MUST be vector_cosine_ops. No query, scoring,
# k, threshold, filter, or embedding semantics change with this index.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64


def hnsw_index_name(table=TABLE) -> str:
    """Deterministic per-table HNSW index name."""
    return f"{str(table)}_embedding_hnsw"


def hnsw_create_sql(table=TABLE, concurrently=True) -> str:
    """CREATE for the HNSW index (production form uses CONCURRENTLY).

    CONCURRENTLY avoids write/read locks on the live RDS table but
    cannot run inside a transaction block — callers must use an
    autocommit connection (see ensure_hnsw_index).
    """
    name = hnsw_index_name(table)
    cc = "CONCURRENTLY " if concurrently else ""
    return (
        f"CREATE INDEX {cc}IF NOT EXISTS {name} "
        f"ON {table} USING hnsw (embedding vector_cosine_ops) "
        f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});"
    )


def hnsw_drop_sql(table=TABLE, concurrently=True) -> str:
    """Rollback path: drops only the HNSW index, never table data."""
    name = hnsw_index_name(table)
    cc = "CONCURRENTLY " if concurrently else ""
    return f"DROP INDEX {cc}IF EXISTS {name};"


# Hybrid lexical retrieval (Step 4; additive, dense default unchanged).
# A GENERATED ALWAYS STORED tsvector keeps existing rows covered with no
# re-ingestion (Postgres backfills on ADD COLUMN) and stays in sync on
# every insert. Same scope predicate guards both candidate sources.
FTS_COLUMN = "content_tsv"
HYBRID_DENSE_N = 20
HYBRID_LEX_N = 20
HYBRID_DEFAULT_DENSE_WEIGHT = 0.5


def fts_index_name(table=TABLE) -> str:
    return f"{str(table)}_content_tsv_gin"


def fts_add_column_sql(table=TABLE) -> str:
    return (
        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {FTS_COLUMN} "
        f"tsvector GENERATED ALWAYS AS "
        f"(to_tsvector('english', content)) STORED;"
    )


def fts_create_index_sql(table=TABLE, concurrently=True) -> str:
    cc = "CONCURRENTLY " if concurrently else ""
    return (
        f"CREATE INDEX {cc}IF NOT EXISTS {fts_index_name(table)} "
        f"ON {table} USING gin ({FTS_COLUMN});"
    )


def fts_drop_sql(table=TABLE, concurrently=True) -> str:
    """Rollback: drops the GIN index and the generated column (data kept)."""
    cc = "CONCURRENTLY " if concurrently else ""
    return (
        f"DROP INDEX {cc}IF EXISTS {fts_index_name(table)}; "
        f"ALTER TABLE {table} DROP COLUMN IF EXISTS {FTS_COLUMN};"
    )


def fuse_scores(dense, lexical, dense_weight=HYBRID_DEFAULT_DENSE_WEIGHT):
    """Deterministic weighted fusion of two {chunk_id: score} maps.

    Each side is min-max normalized to [0,1] (single-valued side maps
    to 1.0; empty side contributes 0.0), then:
        fused = w * dense_norm + (1 - w) * lexical_norm
    Returns [(chunk_id, fused)] sorted by (-fused, chunk_id) so ties
    break identically on every run.
    """
    try:
        w = float(dense_weight)
    except (TypeError, ValueError):
        w = HYBRID_DEFAULT_DENSE_WEIGHT
    w = min(1.0, max(0.0, w))

    def _norm(m):
        if not m:
            return {}
        vals = list(m.values())
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            return {k: 1.0 for k in m}
        span = hi - lo
        return {k: (v - lo) / span for k, v in m.items()}

    dn, ln = _norm(dense), _norm(lexical)
    fused = {cid: w * dn.get(cid, 0.0) + (1.0 - w) * ln.get(cid, 0.0)
             for cid in set(dn) | set(ln)}
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))

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

    def _connect(self, autocommit=False):
        import psycopg2
        from pgvector.psycopg2 import register_vector

        conn = psycopg2.connect(self.dsn)
        if autocommit:
            # Must precede any statement (incl. register_vector): CONCURRENTLY
            # index builds cannot run inside a transaction block.
            conn.set_session(autocommit=True)
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

    def _autocommit_conn(self):
        """Connection safe for CONCURRENTLY index DDL.

        Prefers _connect(autocommit=True) (autocommit precedes every
        statement); falls back for _connect overrides without the flag.
        """
        try:
            return self._connect(autocommit=True)
        except TypeError:
            conn = self._connect()
            try:
                conn.set_session(autocommit=True)
            except Exception:
                try:
                    conn.autocommit = True
                except Exception:
                    pass
            return conn

    def ensure_hnsw_index(self, concurrently=True) -> str:
        """Create the HNSW index if missing; returns the index name.

        Additive only: never touches table data, no re-ingestion needed.
        CONCURRENTLY (production default) requires autocommit, so this
        uses a dedicated connection outside any transaction block.
        Roll back with drop_hnsw_index().
        """
        name = hnsw_index_name(self.table)
        conn = self._autocommit_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(hnsw_create_sql(self.table, concurrently))
        finally:
            conn.close()
        return name

    def drop_hnsw_index(self, concurrently=True) -> str:
        """Rollback: drop only the HNSW index. Data is never touched."""
        name = hnsw_index_name(self.table)
        conn = self._autocommit_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(hnsw_drop_sql(self.table, concurrently))
        finally:
            conn.close()
        return name

    def _index_ready(self, name) -> bool | None:
        """True if a valid index exists; None when DB unreachable."""
        try:
            conn = self._connect()
        except Exception:
            return None
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT indisvalid FROM pg_index i "
                        "JOIN pg_class c ON c.oid = i.indexrelid "
                        "WHERE c.relname = %s",
                        (name,))
                    row = cur.fetchone()
                    return bool(row and row[0])
        except Exception:
            return False
        finally:
            conn.close()

    def hnsw_ready(self) -> bool | None:
        """True if a valid HNSW index exists; None when DB unreachable."""
        return self._index_ready(hnsw_index_name(self.table))

    def fts_ready(self) -> bool | None:
        """True if a valid FTS GIN index exists; None when DB unreachable."""
        return self._index_ready(fts_index_name(self.table))

    def ensure_fts_index(self, concurrently=True) -> str:
        """Additive FTS migration: generated tsvector column + GIN index.

        The GENERATED ALWAYS column backfills existing rows (no
        re-ingestion) and maintains itself on insert. Roll back with
        drop_fts_index().
        """
        name = fts_index_name(self.table)
        conn = self._autocommit_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(fts_add_column_sql(self.table))
                cur.execute(fts_create_index_sql(self.table, concurrently))
        finally:
            conn.close()
        return name

    def drop_fts_index(self, concurrently=True) -> str:
        """Rollback: drop the GIN index and generated column (data kept)."""
        name = fts_index_name(self.table)
        conn = self._autocommit_conn()
        try:
            with conn.cursor() as cur:
                for stmt in fts_drop_sql(self.table, concurrently).split(";"):
                    if stmt.strip():
                        cur.execute(stmt + ";")
        finally:
            conn.close()
        return name

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
        # Index-only changes: keep retrieval semantics identical while new
        # chunks land on HNSW + FTS-indexed tables (no re-ingest needed).
        self.ensure_hnsw_index()
        self.ensure_fts_index()
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

    def _dense_rows(self, qvec, session_id, limit):
        """Raw dense candidate rows: (content, doc_id, src, fname, page,
        chunk_id, dist). Shared by dense and hybrid paths."""
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content, document_id, source_path, file_name,"
                    f" page, chunk_id, embedding <=> %s::vector AS dist "
                    f"FROM {self.table} WHERE {SCOPE_PREDICATE} "
                    f"ORDER BY dist ASC LIMIT %s",
                    (qvec, session_id, limit),
                )
                return cur.fetchall()
        finally:
            conn.close()

    def _lexical_rows(self, query, session_id, limit):
        """Raw FTS candidate rows: (content, doc_id, src, fname, page,
        chunk_id, rank). Same scope filter as dense (no leakage)."""
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content, document_id, source_path, file_name,"
                    f" page, chunk_id, ts_rank_cd({FTS_COLUMN}, "
                    f"plainto_tsquery('english', %s)) AS rank "
                    f"FROM {self.table} "
                    f"WHERE {FTS_COLUMN} @@ plainto_tsquery('english', %s) "
                    f"AND {SCOPE_PREDICATE} "
                    f"ORDER BY rank DESC LIMIT %s",
                    (query, query, session_id, limit),
                )
                return cur.fetchall()
        finally:
            conn.close()

    @staticmethod
    def _to_doc(content, doc_id, src, fname, page, cid):
        return _Doc(content, {
            "source_path": src, "source": src, "file_name": fname,
            "page": page if isinstance(page, int) else None,
            "document_id": doc_id, "chunk_id": cid,
        })

    def _resolve_mode(self, mode):
        if mode is not None:
            return str(mode).lower()
        cfg = self._config
        if cfg is not None:
            return str(getattr(cfg, "retrieval_mode", "dense") or "dense"
                       ).lower()
        return "dense"

    def _resolve_weight(self):
        cfg = self._config
        raw = getattr(cfg, "hybrid_dense_weight",
                      HYBRID_DEFAULT_DENSE_WEIGHT) if cfg is not None \
            else HYBRID_DEFAULT_DENSE_WEIGHT
        try:
            w = float(raw)
        except (TypeError, ValueError):
            w = HYBRID_DEFAULT_DENSE_WEIGHT
        return min(1.0, max(0.0, w))

    def search(self, query: str, k: int = 5, session_id=None, mode=None):
        """pgvector cosine search; returns [(doc, score)] with score in [-1,1].

        session_id None -> persistent-only; a valid id adds that session's
        rows (never another session's). Malformed IDs raise ValueError.
        mode: None (config RAG_RETRIEVAL_MODE, default "dense"), "dense"
        (unchanged single-path behavior), or "hybrid" (dense top-N +
        FTS top-N fused deterministically, deduplicated by chunk_id).
        """
        if session_id is not None:
            from document_scope import validate_session_id
            validate_session_id(session_id)
        qvec = self._embedding().embed_query(query)

        if self._resolve_mode(mode) == "hybrid":
            return self._hybrid_search(query, qvec, k, session_id)

        def _run():
            return self._dense_rows(qvec, session_id, k)

        rows = self._read_with_scope_retry(_run)
        out = []
        for content, doc_id, src, fname, page, cid, dist in rows:
            try:
                score = 1.0 - float(dist)
            except (TypeError, ValueError):
                score = None
            out.append((self._to_doc(content, doc_id, src, fname, page, cid),
                        score))
        return out

    def _hybrid_search(self, query, qvec, k, session_id):
        """Dense + FTS fusion. Same [(doc, score)] shape; scores in [0,1].

        Both candidate sources share the scope predicate. Fused ranking
        is deterministic: weighted normalized blend, ties by chunk_id.
        """
        def _dense():
            return self._dense_rows(qvec, session_id, HYBRID_DENSE_N)

        def _lex():
            return self._lexical_rows(query, session_id, HYBRID_LEX_N)

        dense_rows = self._read_with_scope_retry(_dense)
        try:
            lex_rows = self._read_with_scope_retry(_lex)
        except Exception:
            # Pre-FTS tables (column missing): degrade to dense candidates
            # rather than failing the query. Dense errors still propagate
            # exactly as in dense-only mode.
            lex_rows = []

        docs, dense, lex = {}, {}, {}
        for content, doc_id, src, fname, page, cid, dist in dense_rows:
            try:
                dense[cid] = 1.0 - float(dist)
            except (TypeError, ValueError):
                continue
            docs[cid] = self._to_doc(content, doc_id, src, fname, page, cid)
        for content, doc_id, src, fname, page, cid, rank in lex_rows:
            try:
                lex[cid] = float(rank)
            except (TypeError, ValueError):
                continue
            if cid not in docs:
                docs[cid] = self._to_doc(content, doc_id, src, fname,
                                         page, cid)
        ranked = fuse_scores(dense, lex, self._resolve_weight())
        return [(docs[cid], score) for cid, score in ranked[:k]
                if cid in docs]

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
                  "chunk_count": None, "document_count": None,
                  "hnsw_index": None, "fts_index": None}
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
                    status["hnsw_index"] = self.hnsw_ready()
                    status["fts_index"] = self.fts_ready()
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
