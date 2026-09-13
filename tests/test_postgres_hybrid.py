"""Hybrid postgres retrieval tests (mocked; no AWS/DB needed).

Proves: FTS index definition, lexical exact-term search, dense path
unchanged, hybrid dedupe + deterministic fusion, filters/isolation,
dense default, exact-term win, no provider changes.
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


def _row(cid, score_or_rank, content="text"):
    return (content, "d1", "/s.pdf", "s.pdf", 1, cid, score_or_rank)


class Cfg:
    retrieval_mode = "dense"
    hybrid_dense_weight = 0.5


class HybridCfg(Cfg):
    retrieval_mode = "hybrid"


class TestFtsSchema(unittest.TestCase):
    def test_add_column_is_generated_stored_tsvector(self):
        from postgres_vector_store import fts_add_column_sql
        low = fts_add_column_sql("chunks").lower()
        self.assertIn("add column if not exists content_tsv", low)
        self.assertIn("generated always as", low)
        self.assertIn("to_tsvector('english', content)", low)
        self.assertIn("stored", low)

    def test_gin_index_and_rollback(self):
        from postgres_vector_store import (fts_create_index_sql,
                                           fts_drop_sql, fts_index_name)
        self.assertEqual(fts_index_name("chunks"), "chunks_content_tsv_gin")
        low = fts_create_index_sql("chunks").lower()
        self.assertIn("create index concurrently if not exists", low)
        self.assertIn("using gin (content_tsv)", low)
        drop = fts_drop_sql("chunks").lower()
        self.assertIn("drop index concurrently if exists", drop)
        self.assertIn("drop column if exists content_tsv", drop)
        self.assertNotIn("drop table", drop)
        self.assertNotIn("delete", drop)


class TestLexicalQuery(unittest.TestCase):
    def _store(self, **kw):
        from postgres_vector_store import PostgresVectorStore
        kw.setdefault("dsn", "postgresql://u:p@localhost:15432/rag4i")
        kw.setdefault("embedding_function", FakeEmbeddings())
        kw.setdefault("config", HybridCfg())
        return PostgresVectorStore(**kw)

    def test_lexical_uses_tsquery_rank_and_scope(self):
        from document_scope import validate_session_id  # noqa
        conn, cur = _mock_conn()
        # dense branch then lexical branch (two connections).
        dense = mock.MagicMock()
        dense.__enter__.return_value = dense
        dcur = mock.MagicMock()
        dcur.fetchall.return_value = [_row("c1", 0.2)]
        dense.cursor.return_value.__enter__.return_value = dcur
        lex = mock.MagicMock()
        lex.__enter__.return_value = lex
        lcur = mock.MagicMock()
        lcur.fetchall.return_value = [_row("c2", 0.9)]
        lex.cursor.return_value.__enter__.return_value = lcur
        with mock.patch("psycopg2.connect", side_effect=[dense, lex]):
            res = self._store().search("indemnification", k=5)
        lex_sql = lcur.execute.call_args_list[-1][0][0]
        self.assertIn("plainto_tsquery('english'", lex_sql)
        self.assertIn("ts_rank_cd", lex_sql)
        # Same isolation filter as dense (proof 6+7).
        self.assertIn("scope", lex_sql)
        self.assertIn("session_id", lex_sql)
        params = lcur.execute.call_args_list[-1][0][1]
        self.assertEqual(params[0], "indemnification")
        self.assertEqual(params[1], "indemnification")
        self.assertEqual({d.metadata["chunk_id"] for d, _ in res},
                         {"c1", "c2"})


class TestFusion(unittest.TestCase):
    def test_weighted_formula_and_defaults(self):
        from postgres_vector_store import (HYBRID_DEFAULT_DENSE_WEIGHT,
                                           fuse_scores)
        self.assertEqual(HYBRID_DEFAULT_DENSE_WEIGHT, 0.5)
        ranked = fuse_scores({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, 0.5)
        self.assertEqual([c for c, _ in ranked], ["a", "b"])  # tie->chunk_id
        self.assertAlmostEqual(dict(ranked)["a"], 0.5)
        top = fuse_scores({"a": 0.0}, {"a": 5.0, "b": 1.0}, 0.5)
        self.assertEqual(top[0][0], "a")  # lexical max normalizes to 1.0

    def test_deterministic_tie_break(self):
        from postgres_vector_store import fuse_scores
        r1 = fuse_scores({"x": 0.5, "y": 0.5}, {}, 0.5)
        r2 = fuse_scores({"y": 0.5, "x": 0.5}, {}, 0.5)
        self.assertEqual(r1, r2)
        self.assertEqual([c for c, _ in r1], ["x", "y"])

    def test_weight_clamped_and_bad_input_safe(self):
        from postgres_vector_store import fuse_scores
        self.assertEqual(fuse_scores({"a": 1.0}, {}, 9.0)[0][1], 1.0)
        # Bad weight falls back to default 0.5: 0.5 * norm(1.0) = 0.5.
        self.assertEqual(fuse_scores({"a": 1.0}, {}, "junk")[0][1], 0.5)
        self.assertEqual(fuse_scores({}, {}, 0.5), [])


class TestModes(unittest.TestCase):
    def _store(self, cfg, **kw):
        from postgres_vector_store import PostgresVectorStore
        kw.setdefault("dsn", "postgresql://u:p@localhost:15432/rag4i")
        kw.setdefault("embedding_function", FakeEmbeddings())
        kw["config"] = cfg
        return PostgresVectorStore(**kw)

    def test_dense_default_single_query_unchanged(self):
        conn, cur = _mock_conn()
        cur.fetchall.return_value = [_row("c1", 0.2)]
        with mock.patch("psycopg2.connect", return_value=conn):
            res = self._store(Cfg()).search("q", k=5)
        stmts = [c[0][0] for c in cur.execute.call_args_list]
        self.assertFalse(any("ts_rank" in s for s in stmts))
        self.assertTrue(any("embedding <=>" in s for s in stmts))
        self.assertAlmostEqual(res[0][1], 0.8)  # 1.0 - 0.2 exactly

    def test_hybrid_dedupes_and_improves_exact_term(self):
        # Dense under-ranks the exact-term chunk; lexical ranks it top.
        conns = []
        dense_rows = [_row("c_distractor", 0.05), _row("c_exact", 0.5),
                      _row("c_low", 0.9)]
        lex_rows = [_row("c_exact", 2.5, "indemnification clause"),
                    _row("c_other", 0.3, "other text")]
        # Three connections: dense-only search, hybrid dense branch,
        # hybrid lexical branch.
        for rows in (dense_rows, dense_rows, lex_rows):
            c, cu = _mock_conn()
            cu.fetchall.return_value = rows
            conns.append(c)
        with mock.patch("psycopg2.connect", side_effect=conns):
            dense_res = self._store(Cfg()).search("indemnification", k=5)
            hyb_res = self._store(HybridCfg()).search("indemnification",
                                                      k=5, mode="hybrid")
        self.assertEqual(dense_res[0][0].metadata["chunk_id"],
                         "c_distractor")
        self.assertEqual(hyb_res[0][0].metadata["chunk_id"], "c_exact")
        # Dedupe: chunk appears once even when in both sources.
        ids = [d.metadata["chunk_id"] for d, _ in hyb_res]
        self.assertEqual(len(ids), len(set(ids)))

    def test_session_id_forwarded_to_both_branches(self):
        import re
        conns = []
        for _ in range(2):
            c, cu = _mock_conn()
            cu.fetchall.return_value = []
            conns.append(c)
        with mock.patch("psycopg2.connect", side_effect=conns):
            self._store(HybridCfg()).search(
                "q", k=5, session_id="a" * 32, mode="hybrid")
        for c in conns:
            cu = c.cursor.return_value.__enter__.return_value
            sql, params = cu.execute.call_args_list[-1][0]
            self.assertIn("session_id", sql)
            self.assertIn("a" * 32, params)


class TestNoProviderChanges(unittest.TestCase):
    def test_chroma_still_default_and_dense_default(self):
        import config
        from vector_store import ChromaVectorStore, get_vector_store
        config.reset_config_cache()
        cfg = config.load_config()
        self.assertIsInstance(get_vector_store(cfg), ChromaVectorStore)
        self.assertEqual(cfg.retrieval_mode, "dense")
        self.assertAlmostEqual(cfg.hybrid_dense_weight, 0.5)
        self.assertEqual(cfg.retrieval_k, 5)
        self.assertAlmostEqual(cfg.relevance_threshold, 0.3)

    def test_env_switch(self):
        import config
        os.environ["RAG_RETRIEVAL_MODE"] = "hybrid"
        try:
            config.reset_config_cache()
            self.assertEqual(config.load_config().retrieval_mode, "hybrid")
        finally:
            del os.environ["RAG_RETRIEVAL_MODE"]
            config.reset_config_cache()


if __name__ == "__main__":
    unittest.main()
