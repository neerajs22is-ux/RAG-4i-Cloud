"""HNSW index tests (mocked; no AWS/DB needed).

Proves: migration creates the intended HNSW index with the correct
operator class, retrieval SQL is unchanged, k/threshold/provider
behavior is unchanged, local fallback untouched.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


def _mock_conn():
    cur = mock.MagicMock()
    cur.rowcount = 1
    conn = mock.MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cur
    return conn, cur


class TestHnswDefinition(unittest.TestCase):
    def test_create_uses_hnsw_cosine_ops_and_params(self):
        from postgres_vector_store import hnsw_create_sql
        sql = hnsw_create_sql("chunks")
        low = sql.lower()
        self.assertIn("create index concurrently if not exists", low)
        self.assertIn("chunks_embedding_hnsw", low)
        self.assertIn("using hnsw", low)
        # Operator must match the retrieval distance (<=> = cosine).
        self.assertIn("vector_cosine_ops", low)
        self.assertIn("m = 16", low)
        self.assertIn("ef_construction = 64", low)

    def test_name_derives_per_table(self):
        from postgres_vector_store import hnsw_index_name
        self.assertEqual(hnsw_index_name("chunks"), "chunks_embedding_hnsw")
        self.assertEqual(hnsw_index_name("custom"), "custom_embedding_hnsw")

    def test_rollback_drops_only_index(self):
        from postgres_vector_store import hnsw_drop_sql
        sql = hnsw_drop_sql("chunks").lower()
        self.assertIn("drop index concurrently if exists", sql)
        self.assertIn("chunks_embedding_hnsw", sql)
        self.assertNotIn("drop table", sql)
        self.assertNotIn("delete", sql)


class TestHnswExecution(unittest.TestCase):
    def _store(self, **kw):
        from postgres_vector_store import PostgresVectorStore
        kw.setdefault("dsn", "postgresql://u:p@localhost:15432/rag4i")
        kw.setdefault("embedding_function", FakeEmbeddings())
        return PostgresVectorStore(**kw)

    def test_ensure_uses_autocommit_for_concurrently(self):
        # CREATE INDEX CONCURRENTLY cannot run in a transaction block.
        conn, cur = _mock_conn()
        with mock.patch("psycopg2.connect", return_value=conn):
            name = self._store().ensure_hnsw_index()
        self.assertEqual(name, "chunks_embedding_hnsw")
        # Autocommit must be set before any statement (register_vector
        # included) so CONCURRENTLY never runs in a transaction block.
        conn.set_session.assert_called_with(autocommit=True)
        sql = cur.execute.call_args_list[-1][0][0].lower()
        self.assertIn("using hnsw", sql)

    def test_build_index_applies_hnsw_additively(self):
        conn, cur = _mock_conn()
        from document_scope import resolve_chunk_scope  # noqa: F401
        with mock.patch("psycopg2.connect", return_value=conn):
            from postgres_vector_store import PostgresVectorStore

            class Doc:
                page_content = "hello"
                metadata = {"chunk_id": "c1"}

            n = self._store().build_index([Doc()])
        self.assertEqual(n, 1)
        stmts = [c[0][0].lower() for c in cur.execute.call_args_list]
        self.assertTrue(any("on conflict (chunk_id) do nothing" in s
                            for s in stmts))
        self.assertTrue(any("using hnsw" in s for s in stmts))

    def test_search_sql_still_cosine_ordered_limit(self):
        conn, cur = _mock_conn()
        cur.fetchall.return_value = [("t", "d", "/s", "f", 0, "c", 0.2)]
        with mock.patch("psycopg2.connect", return_value=conn):
            res = self._store().search("q", k=5)
        sql = cur.execute.call_args_list[-1][0][0]
        self.assertIn("embedding <=>", sql)
        self.assertIn("ORDER BY dist ASC", sql)
        self.assertIn("LIMIT", sql)
        self.assertEqual(cur.execute.call_args_list[-1][0][1][-1], 5)
        self.assertAlmostEqual(res[0][1], 0.8)


class TestFrozenBehavior(unittest.TestCase):
    def test_k_and_threshold_unchanged(self):
        import config
        config.reset_config_cache()
        cfg = config.load_config()
        self.assertEqual(cfg.retrieval_k, 5)
        self.assertAlmostEqual(cfg.relevance_threshold, 0.3)
        self.assertEqual(cfg.embedding_model, "all-MiniLM-L6-v2")
        self.assertEqual(cfg.chunk_size, 1000)
        self.assertEqual(cfg.chunk_overlap, 200)

    def test_local_fallback_still_default(self):
        import config
        from vector_store import ChromaVectorStore, get_vector_store
        config.reset_config_cache()
        self.assertIsInstance(get_vector_store(config.load_config()),
                              ChromaVectorStore)


if __name__ == "__main__":
    unittest.main()
