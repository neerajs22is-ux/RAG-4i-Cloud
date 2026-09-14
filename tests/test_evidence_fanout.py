"""Bounded fan-out tests (Phase 4 Step 4.1, mocked, $0).

Forces M1-M10 through fan_out_retrieval/merge_fanout plus the backend
wiring (DIRECT unchanged). No model, retrieval, or benchmark changes.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def src(text, cid, fname="f.pdf", score=0.9):
    return {"content": text, "file_name": fname, "page": 0,
            "score": score, "document_id": "d", "chunk_id": cid,
            "source": "/p", "source_path": "/p"}


LEASE_A = "Riverside lock-in period is 36 months."
LEASE_B = "Harbourview lock-in is 24 months."
AUDIT = "Revenue Rs 450000000, no qualification."


class Counter:
    def __init__(self, mapping=None, fail_on=()):
        self.calls = []
        self.mapping = mapping or {}
        self.fail_on = set(fail_on)

    def __call__(self, query):
        self.calls.append(query)
        for key in self.fail_on:
            if key in query:
                raise RuntimeError("subquery store down")
        for key, hits in self.mapping.items():
            if key in query:
                return list(hits)
        return []


def _plan(*queries):
    return {"mode": "decompose", "original_query": " | ".join(queries),
            "queries": list(queries), "rationale": "t", "constraints": []}


class TestFanout(unittest.TestCase):
    def _run(self, plan, fn):
        from evidence_fanout import fan_out_retrieval, merge_fanout
        per_sub, record = fan_out_retrieval(plan, fn)
        return per_sub, merge_fanout(per_sub), record

    def test_m1_two_docs_two_calls(self):
        fn = Counter({"lock-in period": [src(LEASE_A, "c1")],
                      "dividend": [src(AUDIT, "c2")]})
        per_sub, merged, _ = self._run(
            _plan("What is the lock-in period?",
                  "What dividend was approved?"), fn)
        self.assertEqual(len(fn.calls), 2)
        self.assertEqual(len(merged), 2)
        tags = {s["chunk_id"]: s["evidence_tag"] for s in merged}
        self.assertEqual(tags, {"c1": "Q1", "c2": "Q2"})

    def test_m2_three_docs_all_retained(self):
        fn = Counter({"lock-in": [src(LEASE_A, "c1")],
                      "dividend": [src(AUDIT, "c2")],
                      "uptime": [src("Uptime 99.9%.", "c3")]})
        per_sub, merged, record = self._run(
            _plan("What is the lock-in?", "What dividend?", "What uptime?"),
            fn)
        self.assertEqual(len(fn.calls), 3)
        self.assertEqual(len(merged), 3)
        self.assertEqual(record["subqueries"][2]["n_sources"], 1)

    def test_m3_separate_paths_no_contamination(self):
        fn = Counter({"Riverside": [src(LEASE_A, "c1", "riverside.pdf")],
                      "Harbourview": [src(LEASE_B, "c2", "harbour.pdf")]})
        _, merged, _ = self._run(
            _plan("Riverside lock-in?", "Harbourview lock-in?"), fn)
        by_tag = {}
        for s in merged:
            by_tag.setdefault(s["evidence_tag"], []).append(
                s["file_name"])
        self.assertEqual(by_tag, {"Q1": ["riverside.pdf"],
                                  "Q2": ["harbour.pdf"]})

    def test_m4_mixed_support_represented(self):
        from evidence_verification import verify_evidence
        plan = _plan("What is the lock-in period?",
                     "What are the parking fees?")
        per_sub, merged, _ = self._run(plan, Counter({
            "lock-in": [src(LEASE_A, "c1")], "parking": []}))
        v = verify_evidence("lock-in and parking", merged, plan=plan)
        self.assertEqual(len(v["subqueries"]), 2)
        self.assertTrue(v["subqueries"][0]["sufficient"])
        self.assertFalse(v["subqueries"][1]["sufficient"])
        self.assertFalse(v["sufficient"])

    def test_m5_duplicate_merged_once(self):
        shared = src(LEASE_A, "c1")
        fn = Counter({"lock-in": [shared], "period": [dict(shared)]})
        _, merged, _ = self._run(_plan("lock-in?", "period?"), fn)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["subqueries"], [0, 1])
        self.assertEqual(merged[0]["evidence_tag"], "Q1+Q2")

    def test_m6_constraint_survives(self):
        fn = Counter({"2024-03-15": [src("Signed 2024-03-15.", "c1")]})
        per_sub, merged, _ = self._run(
            _plan("Signed 2024-03-15?", "Who signed?"), fn)
        self.assertIn("2024-03-15", per_sub[0]["query"])
        self.assertIn("2024-03-15", merged[0]["content"])

    def test_m7_negation_preserved(self):
        fn = Counter({"permit": [src("Subletting is not permitted.", "c1")]})
        per_sub, merged, _ = self._run(
            _plan("Does Harbourview permit subletting?",
                  "What is the notice?"), fn)
        self.assertIn("permit subletting", per_sub[0]["query"].lower())
        self.assertIn("not permitted", merged[0]["content"])

    def test_m8_over_three_capped(self):
        from evidence_fanout import fan_out_retrieval
        fn = Counter()
        per_sub, record = fan_out_retrieval(
            {"mode": "decompose", "queries": ["a?", "b?", "c?", "d?", "e?"],
             "original_query": "x", "rationale": "t", "constraints": []},
            fn)
        self.assertLessEqual(len(per_sub), 3)
        self.assertLessEqual(len(fn.calls), 3)

    def test_m9_subquery_failure_isolated(self):
        fn = Counter({"lock-in": [src(LEASE_A, "c1")]},
                     fail_on=("dividend",))
        per_sub, merged, record = self._run(
            _plan("What is the lock-in?", "What dividend?"), fn)
        self.assertEqual(len(fn.calls), 2)
        self.assertIsNotNone(per_sub[1]["error"])
        self.assertIsNone(per_sub[0]["error"])
        self.assertEqual([s["chunk_id"] for s in merged], ["c1"])
        self.assertIsNotNone(record["subqueries"][1]["error"])

    def test_m10_direct_unchanged(self):
        import backend
        import config
        from query_router import observe_query

        class FakeDoc:
            def __init__(self, content, metadata=None):
                self.page_content = content
                self.metadata = dict(metadata or {})

        class FakeStore:
            def search(self, query, k=5):
                return [(FakeDoc(
                    LEASE_A, {"file_name": "f.pdf", "page": 0,
                              "document_id": "d", "chunk_id": "c"}), 0.9)]

            def chunks_for_source(self, file_name, limit=8):
                return []

            def get_status(self):
                return {"ready": True}

        cfg = config.load_config()
        obs = observe_query("What is the lock-in period?", None)
        out = backend._prepare_generation(
            "What is the lock-in period?", cfg, FakeStore(), None,
            None, obs, None)
        self.assertIsNone(out["answer"])
        self.assertIsNone(out["fanout"])
        self.assertEqual(len(out["retrieved"]), 1)
        self.assertNotIn("evidence_tag", out["retrieved"][0])
        self.assertIsInstance(out["verification"], dict)

    def test_coverage_preferred_over_top_scores(self):
        from evidence_fanout import merge_fanout
        per_sub = [
            {"index": 0, "query": "a?",
             "sources": [src("t1", "c1", score=0.99),
                         src("t2", "c2", score=0.98)], "error": None},
            {"index": 1, "query": "b?",
             "sources": [src("t3", "c3", score=0.50)], "error": None},
        ]
        merged = merge_fanout(per_sub, cap=2)
        self.assertEqual([s["chunk_id"] for s in merged], ["c1", "c3"])


if __name__ == "__main__":
    unittest.main()
