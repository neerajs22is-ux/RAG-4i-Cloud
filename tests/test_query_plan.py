"""Bounded query-planning tests (Step 2.1, mocked, $0).

Covers DIRECT/REWRITE/DECOMPOSE classification, the 3-query bound,
constraint preservation, no-invention, malformed fallback, and safe
integration with the frozen retrieval path.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _words(text):
    import re
    return re.findall(r"[A-Za-z0-9]+", str(text or "").lower())


class TestClassify(unittest.TestCase):
    def test_simple_query_is_direct(self):
        from query_plan import plan_query
        p = plan_query("What is the lock-in period in the Riverside lease?")
        self.assertEqual(p["mode"], "direct")
        self.assertEqual(p["queries"],
                         ["What is the lock-in period in the Riverside lease?"])
        self.assertTrue(p["rationale"])

    def test_noisy_query_is_rewritten_once(self):
        from query_plan import plan_query
        p = plan_query("Please kindly tell me about the lease renewal???")
        self.assertEqual(p["mode"], "rewrite")
        self.assertEqual(len(p["queries"]), 1)
        self.assertNotEqual(p["queries"][0],
                            "Please kindly tell me about the lease renewal???")
        for token in ("lease", "renewal"):
            self.assertIn(token, p["queries"][0].lower())

    def test_multi_intent_is_decomposed(self):
        from query_plan import plan_query
        p = plan_query("What is the lock-in period? Who approved the "
                       "March dividend?")
        self.assertEqual(p["mode"], "decompose")
        self.assertEqual(len(p["queries"]), 2)
        self.assertIn("lock-in", p["queries"][0].lower())
        self.assertIn("dividend", p["queries"][1].lower())

    def test_too_many_intents_falls_back_direct(self):
        from query_plan import plan_query
        q = ("What is the lock-in? Who approved dividends? "
             "What is the uptime? How many leave days? "
             "What is the loan tenure?")
        p = plan_query(q)
        self.assertEqual(p["mode"], "direct")
        self.assertEqual(p["queries"], [q])

    def test_single_conjunction_stays_direct(self):
        from query_plan import plan_query
        q = "Compare the parties in lease-riverside.pdf and lease.pdf."
        p = plan_query(q)
        self.assertEqual(p["mode"], "direct")
        self.assertEqual(p["queries"], [q])


class TestConstraints(unittest.TestCase):
    def test_entities_numbers_files_preserved(self):
        from query_plan import plan_query
        q = ("Please tell me how long the Clause 9.1 confidentiality "
             "obligation lasts in nda-mutual.pdf?")
        p = plan_query(q)
        blob = " ".join(p["queries"]).lower()
        for token in ("9.1", "nda-mutual.pdf", "confidentiality"):
            self.assertIn(token, blob)
        for c in p["constraints"]:
            self.assertIn(str(c).lower(), blob)

    def test_dropped_constraint_forces_direct(self):
        from query_plan import _validate
        p = _validate("rewrite", "What does lease-riverside.pdf say?",
                      ["What does it say?"], ["lease-riverside.pdf"], "x")
        self.assertEqual(p["mode"], "direct")
        self.assertEqual(p["queries"],
                         ["What does lease-riverside.pdf say?"])

    def test_no_invention(self):
        from query_plan import plan_query
        for q in ("Please kindly tell me about the lease renewal???",
                  "What is the lock-in period? Who approved the dividend?",
                  "How long does Clause 9.1 last in nda-mutual.pdf?"):
            p = plan_query(q)
            vocab = set(_words(q))
            for out in p["queries"]:
                for w in _words(out):
                    self.assertIn(w, vocab, (q, out, w))


class TestMalformed(unittest.TestCase):
    def test_bad_mode_empty_and_overflow(self):
        from query_plan import _validate
        for bad in ({"mode": "magic", "queries": ["a"]},
                    {"mode": "rewrite", "queries": []},
                    {"mode": "rewrite", "queries": ["a", "b"]},
                    {"mode": "decompose", "queries": ["only one"]},
                    {"mode": "direct", "queries": []}):
            p = _validate(bad["mode"], "orig", bad["queries"], [], "t")
            self.assertEqual(p["mode"], "direct")
            self.assertEqual(p["queries"], ["orig"])

    def test_empty_and_none_input(self):
        from query_plan import plan_query
        for q in ("", "   ", None):
            p = plan_query(q)
            self.assertEqual(p["mode"], "direct")


class TestIntegration(unittest.TestCase):
    def test_direct_forms_identical(self):
        from backend import build_query_forms
        from query_plan import effective_retrieval_query
        for q in ("What is the lock-in period?",
                  "Are there exceptions to renewal?",
                  "What about exceptions?",
                  "Compare the lock-in clauses in contract.pdf and "
                  "lease.pdf."):
            self.assertEqual(build_query_forms(effective_retrieval_query(q)),
                             build_query_forms(q))

    def test_rewrite_flows_through_retrieval(self):
        import backend
        import config

        class FakeDoc:
            def __init__(self, content, metadata=None):
                self.page_content = content
                self.metadata = dict(metadata or {})

        class FakeStore:
            def search(self, query, k=5):
                if "renewal" in query.lower():
                    return [(FakeDoc(
                        "renewal text",
                        {"file_name": "c.pdf", "page": 0,
                         "document_id": "d", "chunk_id": "c"}), 0.9)]
                return []

        hits = backend.retrieve_documents(
            "Please tell me about renewal???",
            config=config.load_config(), vector_store=FakeStore())
        self.assertTrue(hits)
        self.assertEqual(hits[0]["file_name"], "c.pdf")

    def test_benchmark_suite_stays_direct(self):
        # Step 2.2 lock: all 40 frozen scenario queries are single-intent
        # lookups the planner must leave verbatim (no rewrite/decompose).
        import json
        path = os.path.join(os.path.dirname(__file__), "benchmarks",
                            "scenarios_v1.json")
        with open(path, encoding="utf-8") as f:
            scenarios = json.load(f)["scenarios"]
        self.assertEqual(len(scenarios), 40)
        from query_plan import plan_query
        for s in scenarios:
            p = plan_query(s["query"])
            self.assertEqual(p["mode"], "direct", s["scenario_id"])
            self.assertEqual(p["queries"], [s["query"]], s["scenario_id"])

    def test_trailing_filler_with_punct_stripped(self):
        # Add-on hardening: "thanks???" must not survive rewriting.
        from query_plan import plan_query
        p = plan_query("Hi, can you please tell me about the Riverside "
                       "lease lock-in period, thanks???")
        self.assertEqual(p["mode"], "rewrite")
        self.assertEqual(len(p["queries"]), 1)
        self.assertNotIn("thanks", p["queries"][0].lower())
        self.assertNotIn("?", p["queries"][0])
        for token in ("riverside", "lease", "lock-in", "period"):
            self.assertIn(token, p["queries"][0].lower())

    def test_planner_never_breaks_path(self):
        import backend
        import config

        class BoomStore:
            def search(self, query, k=5):
                raise RuntimeError("plan must not mask store errors")

        with self.assertRaises(RuntimeError):
            backend.retrieve_documents(
                "Please tell me about renewal???",
                config=config.load_config(), vector_store=BoomStore())


if __name__ == "__main__":
    unittest.main()
