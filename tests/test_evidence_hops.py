"""Bounded two-hop tests (Phase 4 Step 4.2, mocked, $0).

Forces H1-H8 plus provenance/contamination checks. No model,
retrieval, or benchmark changes involved.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VEGA = ("Vega Advisors was engaged in January for advisory support. "
        "Contact Maya Rao for onboarding.")
RETAINER = "The Vega Advisors engagement carries a Rs 200000 retainer."
MSA = ("Master services agreement with Nortel Systems. Liability is "
       "capped at Rs 10000000 per claim.")
CAP = "The Nortel MSA liability cap is Rs 10000000 per claim."
SIGNED = "Policy signed on 2024-03-15 by the board."
LEASE36 = "The Riverside lease sets a lock-in period of 36 months."
LEASE24 = "The Riverside lease sets a lock-in period of 24 months."


def src(text, cid="c1", fname="f.pdf"):
    return {"content": text, "file_name": fname, "page": 0,
            "score": 0.9, "document_id": "d", "chunk_id": cid,
            "source": "/p", "source_path": "/p"}


class Counter:
    def __init__(self, mapping=None):
        self.calls = []
        self.mapping = mapping or {}

    def __call__(self, query):
        self.calls.append(query)
        for key, hits in self.mapping.items():
            if key.lower() in query.lower():
                return list(hits)
        return []


class TestHops(unittest.TestCase):
    def _run(self, q, hop1, fn, **kw):
        from evidence_hops import run_two_hop
        return run_two_hop(q, hop1, fn, **kw)

    def test_h1_entity_discovery(self):
        fn = Counter({"retainer": [src(RETAINER, cid="c2")]})
        merged, rec = self._run(
            "What monthly retainer does Vega Advisors receive?",
            [src(VEGA)], fn)
        self.assertTrue(rec["hop2"]["triggered"])
        self.assertIn("Vega Advisors", rec["hop2"]["query"])
        self.assertEqual(rec["hops_used"], 2)
        self.assertEqual(len(fn.calls), 1)
        self.assertEqual(
            {s["chunk_id"] for s in merged}, {"c1", "c2"})

    def test_h2_document_discovery(self):
        fn = Counter({"Nortel": [src(CAP, cid="c2")]})
        merged, rec = self._run(
            "What liability cap applies under the Nortel agreement?",
            [src(MSA)], fn)
        self.assertTrue(rec["hop2"]["triggered"])
        self.assertIn("Nortel", rec["hop2"]["query"])
        self.assertEqual(
            {s["chunk_id"] for s in merged}, {"c1", "c2"})

    def test_h3_dependent_numeric_lookup(self):
        fn = Counter({"dividend": [src("Dividend Rs 12 approved.", "c2")]})
        merged, rec = self._run(
            "What dividend did the March board approve?",
            [src("Board met in March with Rao present.", "c1")], fn)
        self.assertTrue(rec["hop2"]["triggered"])
        self.assertEqual(
            {s["chunk_id"] for s in merged}, {"c1", "c2"})

    def test_h4_dependent_date_lookup(self):
        fn = Counter({"signed": [src(SIGNED, cid="c2")]})
        merged, rec = self._run(
            "When was the policy signed by the board?",
            [src("The board adopted a new policy in March.", "c1")], fn)
        self.assertTrue(rec["hop2"]["triggered"])
        self.assertIn("2024-03-15", (rec["hop2"]["query"] +
                                     " ".join(s["content"]
                                              for s in merged)))

    def test_h5_insufficient_hop1_no_hop2(self):
        fn = Counter({"anything": [src("Unrelated filler text.", "c9")]})
        merged, rec = self._run(
            "What is the lock-in period?", [src(LEASE36)], fn)
        self.assertFalse(rec["hop2"]["triggered"])
        self.assertEqual(rec["hops_used"], 1)
        self.assertEqual(len(fn.calls), 0)
        self.assertEqual([s["chunk_id"] for s in merged], ["c1"])

    def test_h6_conflict_never_resolves(self):
        fn = Counter({"months": [src("Extra months text.", "c9")]})
        merged, rec = self._run(
            "What is the lock-in period?",
            [src(LEASE36, cid="c1"), src(LEASE24, cid="c2")], fn)
        self.assertFalse(rec["hop2"]["triggered"])
        self.assertEqual(len(fn.calls), 0)
        self.assertEqual({s["chunk_id"] for s in merged}, {"c1", "c2"})

    def test_h7_constraints_preserved(self):
        from evidence_hops import derive_hop2
        out = derive_hop2(
            "What dividend was approved on 2024-03-15?",
            [src("Board met in March.", "c1")])
        self.assertIsNotNone(out)
        self.assertIn("2024-03-15", out["query"])

    def test_h8_no_third_hop(self):
        from evidence_hops import run_two_hop
        fn = Counter({"x": [src("More text here.", "c9")]})
        merged, rec = self._run("What is X?", [src("Partial X.", "c1")],
                                fn)
        self.assertLessEqual(rec["hops_used"], 2)
        self.assertEqual(rec["max_hops"], 2)
        # A second invocation over merged output cannot chain: either it
        # verifies sufficient (no hop) or derives at most one more query
        # against already-seen evidence, never a growing chain.
        merged2, rec2 = run_two_hop("What is X?", merged, fn)
        total_calls = len(fn.calls)
        self.assertLessEqual(total_calls, 2)
        self.assertLessEqual(rec2["hops_used"], 2)

    def test_provenance_and_tags(self):
        from evidence_hops import run_two_hop
        fn = Counter({"retainer": [src(RETAINER, cid="c2")]})
        merged, rec = self._run(
            "What monthly retainer does Vega Advisors receive?",
            [src(VEGA)], fn)
        by_id = {s["chunk_id"]: s for s in merged}
        self.assertEqual(by_id["c1"]["hops"], [1])
        self.assertEqual(by_id["c2"]["hops"], [2])
        self.assertEqual(by_id["c1"]["evidence_tag"], "H1")
        self.assertEqual(by_id["c2"]["evidence_tag"], "H2")
        self.assertIn("bridge", rec["hop2"]["finding"])
        self.assertTrue(rec["hop2"]["finding"]["unresolved"])

    def test_no_contamination(self):
        from evidence_hops import run_two_hop
        fn = Counter({"retainer": [src(RETAINER, cid="c2")]})
        merged, _ = self._run(
            "What monthly retainer does Vega Advisors receive?",
            [src(VEGA)], fn)
        for s in merged:
            if s["chunk_id"] == "c1":
                self.assertEqual(s["hops"], [1])
                self.assertNotIn("Rs 200000", s["content"])

    def test_hop2_failure_returns_hop1(self):
        def boom(query):
            raise RuntimeError("hop store down")

        from evidence_hops import run_two_hop
        merged, rec = self._run(
            "What monthly retainer does Vega Advisors receive?",
            [src(VEGA)], boom)
        self.assertTrue(rec["hop2"]["triggered"])
        self.assertIsNotNone(rec["hop2"]["error"])
        self.assertEqual([s["chunk_id"] for s in merged], ["c1"])


if __name__ == "__main__":
    unittest.main()
