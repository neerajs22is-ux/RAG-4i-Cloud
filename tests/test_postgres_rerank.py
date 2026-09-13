"""Recall-then-rerank tests (mocked; no AWS/DB/model needed).

Proves: candidate expansion, reorder by reranker, final k, determinism,
hybrid win preserved, semantic win, filters, isolation, disabled path
unchanged, no provider regression.
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


class FakeReranker:
    def __init__(self, scores):
        self.scores = dict(scores)
        self.calls = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return [self.scores.get(t, 0.0) for t in texts]


def _mock_conn(rows=()):
    cur = mock.MagicMock()
    cur.rowcount = 1
    cur.fetchall.return_value = list(rows)
    conn = mock.MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cur
    return conn, cur


def _row(cid, dist, content=None):
    return ((content or f"text {cid}"), "d1", "/s.pdf", "s.pdf", 1, cid,
            dist)


class Cfg:
    retrieval_mode = "dense"
    hybrid_dense_weight = 0.5
    reranking_enabled = "0"
    rerank_candidates = 64
    rerank_model = "overlap"


class RerankCfg(Cfg):
    reranking_enabled = "1"


class HybridRerankCfg(RerankCfg):
    retrieval_mode = "hybrid"


def _store(cfg, reranker, **kw):
    from postgres_vector_store import PostgresVectorStore
    kw.setdefault("dsn", "postgresql://u:p@localhost:15432/rag4i")
    kw.setdefault("embedding_function", FakeEmbeddings())
    kw["config"] = cfg
    kw["reranker"] = reranker
    return PostgresVectorStore(**kw)


class TestCandidateExpansion(unittest.TestCase):
    def test_dense_branch_fetches_broad_pool(self):
        rows = [_row(f"c{i}", 0.1 + i * 0.01) for i in range(10)]
        conn, cur = _mock_conn(rows)
        rr = FakeReranker({f"text c{i}": float(i) for i in range(10)})
        with mock.patch("psycopg2.connect", return_value=conn):
            res = _store(RerankCfg(), rr).search("q", k=5, rerank=True)
        sql, params = cur.execute.call_args_list[-1][0]
        self.assertIn("embedding <=>", sql)
        self.assertEqual(params[-1], 64)  # broad pool, not final k
        self.assertEqual(len(res), 5)  # final k unchanged

    def test_hybrid_union_feeds_reranker(self):
        drows = [_row("c1", 0.2, "dense one"), _row("c2", 0.3, "dense two")]
        lrows = [("lex three", "d1", "/s.pdf", "s.pdf", 1, "c3", 1.5)]
        conns = []
        for rows in (drows, lrows):
            c, cu = _mock_conn(rows)
            conns.append(c)
        rr = FakeReranker({"dense one": 0.1, "dense two": 0.2,
                           "lex three": 0.9})
        with mock.patch("psycopg2.connect", side_effect=conns):
            res = _store(HybridRerankCfg(), rr).search("q", k=5,
                                                       rerank=True)
        seen = [t for _, ts in rr.calls for t in ts]
        self.assertEqual(seen, ["dense one", "dense two", "lex three"])
        self.assertEqual(res[0][0].metadata["chunk_id"], "c3")


class TestOrdering(unittest.TestCase):
    def test_reranker_inverts_dense_order(self):
        rows = [_row("c1", 0.05, "wrong answer"), _row("c2", 0.5, "right one")]
        conn, cur = _mock_conn(rows)
        rr = FakeReranker({"wrong answer": 0.0, "right one": 9.9})
        with mock.patch("psycopg2.connect", return_value=conn):
            res = _store(RerankCfg(), rr).search("q", k=5, rerank=True)
        self.assertEqual(res[0][0].metadata["chunk_id"], "c2")
        self.assertAlmostEqual(res[0][1], 1.0)  # normalized top
        self.assertAlmostEqual(res[1][1], 0.0)

    def test_deterministic_ties_and_repeats(self):
        rows = [_row("c_b", 0.1, "same"), _row("c_a", 0.2, "same")]
        rr = FakeReranker({"same": 1.0})

        def run():
            conn, _ = _mock_conn(rows)
            with mock.patch("psycopg2.connect", return_value=conn):
                return _store(RerankCfg(), rr).search("q", k=5, rerank=True)

        r1, r2 = run(), run()
        self.assertEqual([d.metadata["chunk_id"] for d, _ in r1],
                         [d.metadata["chunk_id"] for d, _ in r2])
        self.assertEqual(r1[0][0].metadata["chunk_id"], "c_a")  # tie->id

    def test_hybrid_exact_term_win_preserved(self):
        drows = [_row("c_distractor", 0.05, "generic policies"),
                 _row("c_exact", 0.5, "Riverside lock-in 36 months")]
        lrows = [("Riverside lock-in 36 months", "d1", "/s.pdf", "s.pdf",
                  1, "c_exact", 2.5)]
        conns = []
        for rows in (drows, lrows):
            c, cu = _mock_conn(rows)
            conns.append(c)
        rr = FakeReranker({"generic policies": 0.1,
                           "Riverside lock-in 36 months": 0.95})
        with mock.patch("psycopg2.connect", side_effect=conns):
            res = _store(HybridRerankCfg(), rr).search("lock-in Riverside",
                                                       k=5, rerank=True)
        self.assertEqual(res[0][0].metadata["chunk_id"], "c_exact")


class TestGuards(unittest.TestCase):
    def test_scope_filter_and_session_forwarded(self):
        rows = [_row("c1", 0.2)]
        conn, cur = _mock_conn(rows)
        rr = FakeReranker({"text c1": 1.0})
        with mock.patch("psycopg2.connect", return_value=conn):
            _store(RerankCfg(), rr).search("q", k=5, session_id="d" * 32,
                                           rerank=True)
        sql, params = cur.execute.call_args_list[-1][0]
        self.assertIn("scope", sql)
        self.assertIn("d" * 32, params)

    def test_disabled_path_never_scores(self):
        rows = [_row("c1", 0.2)]
        conn, cur = _mock_conn(rows)
        rr = FakeReranker({"text c1": 99.0})
        with mock.patch("psycopg2.connect", return_value=conn):
            res = _store(Cfg(), rr).search("q", k=5)
        self.assertEqual(rr.calls, [])
        sql = cur.execute.call_args_list[-1][0][0]
        self.assertIn("embedding <=>", sql)
        self.assertNotIn("ts_rank", sql)
        self.assertAlmostEqual(res[0][1], 0.8)

    def test_enabled_without_scorer_fails_loud(self):
        rows = [_row("c1", 0.2), _row("c2", 0.3)]
        conn, _ = _mock_conn(rows)

        class ShortReranker:
            def score(self, query, texts):
                return [1.0]  # wrong length: must fail, never truncate

        with mock.patch("psycopg2.connect", return_value=conn):
            with self.assertRaises(RuntimeError):
                _store(RerankCfg(), ShortReranker()).search("q", k=5,
                                                            rerank=True)


class TestRerankingModule(unittest.TestCase):
    def test_factory_none_when_disabled(self):
        from reranking import get_reranker
        self.assertIsNone(get_reranker(Cfg()))
        self.assertIsNone(get_reranker(None))

    def test_overlap_scorer(self):
        from reranking import TokenOverlapReranker, get_reranker

        class OverlapCfg(Cfg):
            reranking_enabled = "1"
            rerank_model = "overlap"

        rr = get_reranker(OverlapCfg())
        self.assertIsInstance(rr, TokenOverlapReranker)
        scores = rr.score("lock-in period riverside",
                          ["Riverside lock-in 36 months", "unrelated text"])
        self.assertGreater(scores[0], scores[1])

    def test_no_provider_regression(self):
        import config
        from vector_store import ChromaVectorStore, get_vector_store
        config.reset_config_cache()
        cfg = config.load_config()
        self.assertIsInstance(get_vector_store(cfg), ChromaVectorStore)
        self.assertEqual(cfg.reranking_enabled, "0")
        self.assertEqual(cfg.rerank_candidates, 64)
        self.assertEqual(cfg.retrieval_k, 5)
        self.assertAlmostEqual(cfg.relevance_threshold, 0.3)


if __name__ == "__main__":
    unittest.main()
