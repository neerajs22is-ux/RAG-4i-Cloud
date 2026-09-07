"""PostgreSQL provider tests (mocked; no AWS needed)."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})
        self.id = None


class FakeEmbeddings:
    def __init__(self):
        self.docs_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.docs_calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return [0.1, 0.2, 0.3]


class TestPostgresConfig(unittest.TestCase):
    def setUp(self):
        self._old = dict(os.environ)
        for k in ["DATABASE_URL", "DB_HOST", "DB_PORT", "DB_NAME",
                  "DB_USER", "DB_PASSWORD", "VECTOR_STORE"]:
            os.environ.pop(k, None)
        import config
        config.reset_config_cache()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old)
        import config
        config.reset_config_cache()

    def test_defaults_keep_chroma_and_tunnel_port(self):
        import config
        cfg = config.load_config()
        self.assertEqual(cfg.vector_store, "chroma")
        self.assertEqual(cfg.db_port, 15432)
        self.assertEqual(cfg.db_name, "rag4i")
        self.assertIn("15432", cfg.postgres_dsn())

    def test_database_url_wins(self):
        os.environ["DATABASE_URL"] = "postgresql://u:p@h:5433/db"
        os.environ["DB_HOST"] = "other"
        import config
        config.reset_config_cache()
        self.assertEqual(config.load_config().postgres_dsn(),
                         "postgresql://u:p@h:5433/db")


class TestPostgresSchema(unittest.TestCase):
    def test_schema_has_required_columns_and_384(self):
        from postgres_vector_store import DIM, SCHEMA_SQL
        self.assertEqual(DIM, 384)
        low = SCHEMA_SQL.lower()
        for col in ["document_id", "chunk_id", "source_path", "file_name",
                    "page", "content", "embedding"]:
            self.assertIn(col, low)
        self.assertIn("vector(384)", low)
        self.assertIn("create extension", low)


class TestPostgresBuildSearch(unittest.TestCase):
    def _store(self, **kw):
        from postgres_vector_store import PostgresVectorStore
        kw.setdefault("dsn", "postgresql://u:p@localhost:15432/rag4i")
        kw.setdefault("embedding_function", FakeEmbeddings())
        return PostgresVectorStore(**kw)

    def test_build_uses_embedder_and_conflict_nothing(self):
        cur = mock.MagicMock()
        cur.rowcount = 1
        conn = mock.MagicMock()
        conn.__enter__.return_value = conn
        conn.cursor.return_value.__enter__.return_value = cur
        with mock.patch("psycopg2.connect", return_value=conn) as mc:
            st = self._store()
            n = st.build_index([FakeDoc("hello", {"chunk_id": "c1"})])
            self.assertEqual(n, 1)
            mc.assert_called()
            sql = cur.execute.call_args_list[-1][0][0]
            self.assertIn("ON CONFLICT (chunk_id) DO NOTHING", sql)
            # Same embedder interface as Chroma path.
            self.assertEqual(len(st._embedding().embed_documents(["x"])[0]), 3)

    def test_search_maps_score_and_metadata(self):
        cur = mock.MagicMock()
        cur.fetchall.return_value = [
            ("lease 36 months", "d1", "/p/lease.pdf", "lease.pdf", 4, "c1", 0.2),
        ]
        conn = mock.MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        with mock.patch("psycopg2.connect", return_value=conn):
            res = self._store().search("lock-in", k=5)
            self.assertEqual(len(res), 1)
            doc, score = res[0]
            self.assertAlmostEqual(score, 0.8)  # 1.0 - 0.2
            self.assertEqual(doc.metadata["file_name"], "lease.pdf")
            self.assertEqual(doc.metadata["page"], 4)

    def test_backend_threshold_applies_to_pg(self):
        import backend
        import config
        cfg = config.load_config()

        class LowPG:
            def search(self, q, k=5):
                d = FakeDoc("x", {"source_path": "/a.pdf", "source": "/a.pdf",
                                  "file_name": "a.pdf", "page": 0,
                                  "document_id": "d", "chunk_id": "c"})
                return [(d, 0.1)]
        self.assertEqual(backend.retrieve_documents("q", config=cfg,
                                                    vector_store=LowPG()), [])
        self.assertEqual(cfg.retrieval_k, 5)
        self.assertAlmostEqual(cfg.relevance_threshold, 0.3)


class TestProviderSwitching(unittest.TestCase):
    def test_chroma_default_postgres_opt_in(self):
        import config
        from vector_store import ChromaVectorStore, get_vector_store
        from postgres_vector_store import PostgresVectorStore
        config.reset_config_cache()
        self.assertIsInstance(get_vector_store(config.load_config()),
                              ChromaVectorStore)
        os.environ["VECTOR_STORE"] = "postgres"
        try:
            config.reset_config_cache()
            vs = get_vector_store(config.load_config())
            self.assertIsInstance(vs, PostgresVectorStore)
        finally:
            del os.environ["VECTOR_STORE"]
            config.reset_config_cache()


if __name__ == "__main__":
    unittest.main()
