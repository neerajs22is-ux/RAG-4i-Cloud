"""Bounded correction tests (Step 3.2, mocked, $0).

Forces R1-R10 through run_bounded_correction with counting fake
retrievers. No model, retrieval, or benchmark changes involved.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

WEAK = "The Riverside lease mentions a lock-in."
STRONG = "The Riverside lease sets a lock-in period of 36 months."
GENERIC = "Confidentiality obligations last 3 years under the agreement."
TERM = "The Clause 9.1 obligation is long: confidentiality of 3 years."
A36 = "The Riverside lease sets a lock-in period of 36 months."
A24 = "The Riverside lease sets a lock-in period of 24 months."
CLARIFY = "The 24-month term applies to Harbourview only."
LEASE = ("The Riverside lease sets a lock-in period of 36 months with "
         "5 percent escalation.")


def src(text, cid="c1", fname="f.pdf"):
    return {"content": text, "file_name": fname, "page": 0,
            "score": 0.9, "document_id": "d", "chunk_id": cid,
            "source": "/p", "source_path": "/p"}


class Counter:
    def __init__(self, hits=None, fail=False):
        self.calls = []
        self.hits = hits or []
        self.fail = fail

    def __call__(self, query):
        self.calls.append(query)
        if self.fail:
            raise RuntimeError("store down")
        return self.hits


class TestCorrection(unittest.TestCase):
    def _run(self, q, initial, fn, **kw):
        from evidence_verification import run_bounded_correction
        return run_bounded_correction(q, initial, fn, **kw)

    def test_r1_insufficient_corrected_to_supported(self):
        fn = Counter([src(STRONG, cid="c2")])
        final, v = self._run("What is the lock-in period?",
                             [src(WEAK)], fn)
        self.assertEqual(v["reason"], "all-supported")
        self.assertTrue(v["sufficient"])
        rec = v["correction"]
        self.assertTrue(rec["triggered"])
        self.assertEqual(rec["rounds"], 1)
        self.assertEqual(rec["verifications"], 2)
        self.assertEqual(len(fn.calls), 1)
        self.assertEqual(rec["gained"], ["c2"])
        self.assertEqual(len(final), 2)

    def test_r2_exact_term_query(self):
        fn = Counter([src(TERM, cid="c2")])
        final, v = self._run(
            'How long does the "Clause 9.1" obligation last?',
            [src(GENERIC)], fn)
        rec = v["correction"]
        self.assertTrue(rec["triggered"])
        self.assertIn("9.1", rec["query"])
        self.assertTrue(v["sufficient"])
        self.assertEqual(len(fn.calls), 1)

    def test_r3_conflict_clarified_then_stops(self):
        fn = Counter([src(CLARIFY, cid="c3")])
        final, v = self._run(
            "What is the lock-in period?",
            [src(A36, cid="c1"), src(A24, cid="c2")], fn)
        rec = v["correction"]
        self.assertTrue(rec["triggered"])
        self.assertEqual(rec["rounds"], 1)
        self.assertEqual(rec["verifications"], 2)
        self.assertEqual(len(fn.calls), 1)
        # Values still disagree: honest stop, conflict recorded.
        self.assertEqual(v["reason"], "conflicting-evidence")
        self.assertTrue(v["conflicting_claims"])
        rounds = {s["chunk_id"]: s["retrieval_round"] for s in final}
        self.assertEqual(rounds, {"c1": 1, "c2": 1, "c3": 2})

    def test_r4_still_insufficient_stops(self):
        fn = Counter([src(WEAK, cid="c9")])
        final, v = self._run("What is the lock-in period and escalation?",
                             [src(WEAK)], fn)
        rec = v["correction"]
        self.assertTrue(rec["triggered"])
        self.assertEqual(rec["rounds"], 1)
        self.assertEqual(len(fn.calls), 1)
        self.assertFalse(v["sufficient"])

    def test_r5_number_date_unit_preserved(self):
        seen = {}

        def fn(q):
            seen["q"] = q
            return [src(STRONG, cid="c2")]

        calls = []
        orig = fn
        fn2 = lambda q: (calls.append(q), orig(q))[1]
        self._run("What dividend was approved on 2024-03-15 for "
                  "Rs 450000000?", [src(WEAK)], fn2)
        self.assertEqual(len(calls), 1)
        self.assertIn("2024-03-15", calls[0])
        self.assertIn("450000000", calls[0])

    def test_r6_negation_preserved(self):
        import re
        calls = []
        fn = lambda q: (calls.append(q), [src(STRONG, cid="c2")])[1]
        self._run("Is smoking not permitted near the lease office?",
                  [src(WEAK)], fn)
        self.assertEqual(len(calls), 1)
        self.assertTrue(re.search(r"\bnot\b", calls[0].lower()))

    def test_r7_no_meaningful_query_no_research(self):
        fn = Counter([src(LEASE, cid="c2")])
        final, v = self._run("What are the parking fees?", [src(LEASE)],
                             fn)
        self.assertFalse(v["correction"]["triggered"])
        self.assertEqual(len(fn.calls), 0)
        self.assertEqual(final, [src(LEASE)])

    def test_r8_duplicates_deduplicated(self):
        dup = src(WEAK, cid="c1")
        fn = Counter([dup])
        final, v = self._run("What is the lock-in period?",
                             [src(WEAK)], fn)
        self.assertEqual([s["chunk_id"] for s in final], ["c1"])
        self.assertEqual(v["correction"]["gained"], [])
        self.assertEqual(v["correction"]["rounds"], 1)

    def test_r9_provenance_rounds(self):
        fn = Counter([src(STRONG, cid="c2")])
        final, v = self._run("What is the lock-in period?",
                             [src(WEAK)], fn)
        by_id = {s["chunk_id"]: s["retrieval_round"] for s in final}
        self.assertEqual(by_id, {"c1": 1, "c2": 2})
        ref_rounds = {r["chunk_id"]: r["retrieval_round"]
                      for r in v["evidence_refs"]}
        self.assertEqual(ref_rounds, {"c1": 1, "c2": 2})

    def test_r10_retrieval_failure_safe(self):
        fn = Counter(fail=True)
        init = [src(WEAK)]
        final, v = self._run("What is the lock-in period?", init, fn)
        self.assertEqual(len(fn.calls), 1)
        self.assertEqual(final, init)
        self.assertEqual(v["correction"]["reason"], "retrieval-error")
        self.assertFalse(v["correction"]["triggered"])

    def test_bounds_single_round_single_reverify(self):
        fn = Counter([src(WEAK, cid="c9")])
        _, v = self._run("What is the lock-in period and escalation?",
                         [src(WEAK)], fn)
        rec = v["correction"]
        self.assertLessEqual(rec["rounds"], 1)
        self.assertLessEqual(rec["verifications"], 2)
        self.assertLessEqual(len(fn.calls), 1)


if __name__ == "__main__":
    unittest.main()
