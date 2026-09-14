"""Bounded evidence-verification tests (Step 3.1, mocked, $0).

Forces V1-V12 behavior with local fixtures. No model, retrieval, or
benchmark changes involved.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LEASE = ("The Riverside lease sets a lock-in period of 36 months with "
         "5 percent escalation.")
LEASE_PARA = ("The lock-in period under the Riverside lease is 36 months, "
              "with escalation of 5 percent.")
LEASE_SYNONYM = ("Under the Riverside agreement, tenants face a 36-month "
                 "lock-in and 5% escalation.")
PARKING = "The garage offers monthly parking permits."
SHORT = "The Riverside lease mentions a lock-in period."
OTHER_TERM = "The Harbourview lease sets a lock-in period of 24 months."
PCT_OTHER = "The Harbourview lease applies 7 percent escalation."
SMOKE = "Smoking is not permitted on the premises."
SIGNED = "Policy signed on 2024-03-15 by the board."
SIGNED_OTHER = "Policy signed on 2023-03-15 by the board."
NOTICE = "Tenants must give 30 days notice before vacating."


def src(text, cid="c1", fname="lease.pdf", page=0):
    return {"content": text, "file_name": fname, "page": page,
            "score": 0.9, "document_id": "d1", "chunk_id": cid,
            "source": "/p", "source_path": "/p"}


class TestVerdicts(unittest.TestCase):
    def _v(self, q, texts, **kw):
        from evidence_verification import verify_evidence
        return verify_evidence(
            q, [src(t, cid=f"c{i}") for i, t in enumerate(texts)], **kw)

    def test_v1_exact_supported(self):
        r = self._v("What is the lock-in period in the Riverside lease?",
                    [LEASE])
        self.assertTrue(r["sufficient"])
        self.assertEqual(r["reason"], "all-supported")
        self.assertTrue(r["supported_claims"])
        self.assertEqual(r["unsupported_claims"], [])
        self.assertEqual(r["conflicting_claims"], [])
        self.assertTrue(r["verification_required"])
        self.assertEqual(r["evidence_refs"][0]["chunk_id"], "c0")

    def test_v2_paraphrase_supported(self):
        r = self._v("What is the lock-in period in the Riverside lease?",
                    [LEASE_PARA])
        self.assertTrue(r["sufficient"])

    def test_synonym_substitution_never_falsely_supported(self):
        # Open synonyms ("agreement" for "lease") are beyond the
        # deterministic core: insufficient, never supported.
        r = self._v("What is the lock-in period in the Riverside lease?",
                    [LEASE_SYNONYM])
        self.assertFalse(r["sufficient"])
        self.assertEqual(r["reason"], "insufficient-evidence")

    def test_v3_absent_unsupported(self):
        r = self._v("What are the monthly parking permit fees?", [LEASE])
        self.assertFalse(r["sufficient"])
        self.assertEqual(r["reason"], "unsupported-claims")
        self.assertTrue(r["unsupported_claims"])

    def test_v4_related_insufficient(self):
        r = self._v("What is the lock-in period and the escalation?",
                    [SHORT])
        self.assertFalse(r["sufficient"])
        self.assertEqual(r["reason"], "insufficient-evidence")
        self.assertTrue(r["supported_claims"])
        self.assertTrue(r["unsupported_claims"])

    def test_v5_conflicting_sources(self):
        r = self._v("What is the lock-in period?", [LEASE, OTHER_TERM])
        self.assertFalse(r["sufficient"])
        self.assertEqual(r["reason"], "conflicting-evidence")
        self.assertTrue(r["conflicting_claims"])

    def test_v6_conflicting_numbers(self):
        q = "What escalation applies?"
        r = self._v(q, [LEASE, PCT_OTHER])
        self.assertFalse(r["sufficient"])
        self.assertTrue(r["conflicting_claims"])

    def test_v7_negation_not_falsely_supported(self):
        r = self._v("Is smoking permitted on the premises?", [SMOKE])
        self.assertTrue(r["sufficient"])
        self.assertEqual(r["conflicting_claims"], [])
        r2 = self._v("Is parking permitted on the premises?", [SMOKE])
        self.assertFalse(r2["sufficient"])

    def test_v8_entity_qualifier_distinct(self):
        r = self._v("Must landlords give notice?", [NOTICE])
        self.assertFalse(r["sufficient"])
        self.assertTrue(r["unsupported_claims"])
        r2 = self._v("Must tenants give notice?", [NOTICE])
        self.assertTrue(r2["sufficient"])

    def test_v9_date_distinct(self):
        r = self._v("Was the policy signed on 2024-03-15?", [SIGNED])
        self.assertTrue(r["sufficient"])
        r2 = self._v("Was the policy signed on 2024-03-15?",
                     [SIGNED_OTHER])
        self.assertFalse(r2["sufficient"])

    def test_v10_mixed_support_separate(self):
        r = self._v("What is the lock-in period and the parking fee?",
                    [LEASE])
        self.assertFalse(r["sufficient"])
        self.assertTrue(r["supported_claims"])
        self.assertTrue(r["unsupported_claims"])
        self.assertNotEqual(set(r["supported_claims"]),
                            set(r["unsupported_claims"]))

    def test_v11_decompose_represented(self):
        from evidence_verification import verify_evidence
        plan = {"mode": "decompose",
                "queries": ["What is the lock-in period?",
                            "What are the parking fees?"]}
        r = verify_evidence("lock-in and parking",
                            [src(LEASE), src(PARKING, cid="c2")], plan=plan)
        self.assertEqual(len(r["subqueries"]), 2)
        self.assertTrue(r["subqueries"][0]["sufficient"])
        self.assertFalse(r["subqueries"][1]["sufficient"])
        self.assertFalse(r["sufficient"])
        self.assertEqual(r["reason"], "subquery-evidence-gap")

    def test_v12_failure_fails_open(self):
        from evidence_verification import verify_evidence
        r = verify_evidence("What is the lock-in period?", 42)
        self.assertTrue(r["sufficient"])
        self.assertEqual(r["reason"], "verifier-error")
        self.assertFalse(r["verification_required"])


class TestWiring(unittest.TestCase):
    def test_prepare_exposes_verification_on_generation_path(self):
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
                    LEASE, {"file_name": "lease.pdf", "page": 0,
                            "document_id": "d", "chunk_id": "c"}), 0.9)]

            def chunks_for_source(self, file_name, limit=8):
                return []

            def get_status(self):
                return {"ready": True}

        class FakeLLM:
            def generate(self, context, question, prompt_template):
                raise AssertionError("no generation in prepare")

            def is_reachable(self, timeout=3.0):
                return True

        cfg = config.load_config()
        obs = observe_query("What is the lock-in period?", None)
        out = backend._prepare_generation(
            "What is the lock-in period?", cfg, FakeStore(), FakeLLM(),
            None, obs, None)
        self.assertIsNone(out["answer"])
        self.assertTrue(out["retrieved"])
        v = out["verification"]
        self.assertIsInstance(v, dict)
        self.assertTrue(v["sufficient"])
        self.assertIn("supported_claims", v)

    def test_decided_paths_carry_none(self):
        import backend
        import config
        from query_router import observe_query

        class FakeStore:
            def search(self, query, k=5):
                return []

            def get_status(self):
                return {"ready": True}

        cfg = config.load_config()
        obs = observe_query("What is the lock-in period?", None)
        out = backend._prepare_generation(
            "What is the lock-in period?", cfg, FakeStore(), None,
            None, obs, None)
        self.assertIsNotNone(out["answer"])
        self.assertIsNone(out["verification"])


if __name__ == "__main__":
    unittest.main()
