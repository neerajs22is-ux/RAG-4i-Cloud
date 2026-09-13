"""Bounded groundedness verification tests (no AWS/DB/model needed).

Proves: grounded pass, unsupported/contradiction/qualifier/numeric
detection, one repair fixes, second failure stops safe, max-one-repair,
disabled unchanged, provenance intact, retrieval/rerank intact, suite
passes (via running alongside existing tests).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EV = ["The Riverside lease sets a lock-in period of 36 months with "
      "5 percent escalation. Early termination is not permitted."]


def src(content=EV[0]):
    return {"content": content, "file_name": "lease.pdf", "page": 4,
            "score": 0.9, "document_id": "d1", "chunk_id": "c1",
            "source": "/p/lease.pdf", "source_path": "/p/lease.pdf"}


class Cfg:
    groundedness_enabled = "0"


class OnCfg(Cfg):
    groundedness_enabled = "1"


class FakeProvider:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, context, question, template):
        self.calls.append((context, question, template))
        if not self.replies:
            raise AssertionError("repair called more than scripted")
        return self.replies.pop(0)


class TestChecker(unittest.TestCase):
    def test_grounded_pass(self):
        from groundedness import check_groundedness
        f = check_groundedness("q", "The lock-in period is 36 months.",
                               EV)
        self.assertTrue(f["grounded"])
        self.assertEqual(f["unsupported_claims"], [])

    def test_unsupported_claim_detected(self):
        from groundedness import check_groundedness
        f = check_groundedness("q", "The lease includes free parking "
                                    "for all tenants.", EV)
        self.assertFalse(f["grounded"])
        self.assertEqual(len(f["unsupported_claims"]), 1)

    def test_contradiction_detected(self):
        from groundedness import check_groundedness
        f = check_groundedness("q", "The lock-in period is 48 months.",
                               EV)
        self.assertFalse(f["grounded"])
        self.assertTrue(f["contradictions"])

    def test_missing_qualifier_detected(self):
        from groundedness import check_groundedness
        f = check_groundedness(
            "q", "The Riverside lease lock-in period of 36 months allows "
                 "early termination with notice.", EV)
        self.assertFalse(f["grounded"])
        text = " ".join(f["contradictions"] + f["missing_qualifiers"] +
                        f["unsupported_claims"])
        self.assertIn("not", text)

    def test_numeric_mismatch_detected(self):
        from groundedness import check_groundedness
        f = check_groundedness("q", "The escalation is 7 percent.", EV)
        self.assertFalse(f["grounded"])
        text = " ".join(f["contradictions"] + f["numeric_mismatches"] +
                        f["unsupported_claims"])
        self.assertIn("7", text)


class TestOrchestrator(unittest.TestCase):
    def test_repair_fixes_and_returns_repaired(self):
        from groundedness import verify_response
        fixed = "The lock-in period is 36 months."
        p = FakeProvider([fixed])
        final, meta = verify_response(
            "q", "The lock-in period is 48 months.", [src()], OnCfg(), p)
        self.assertEqual(final, fixed)
        self.assertEqual(meta["checks"], 2)
        self.assertEqual(meta["repairs"], 1)
        self.assertTrue(meta["repaired"] and meta["grounded"])
        # Repair used retrieved evidence, not verifier text.
        ctx = p.calls[0][0]
        self.assertIn("Riverside lease", ctx)
        self.assertNotIn("48 months", ctx.split("Findings:")[0])

    def test_second_failure_stops_safe_with_original(self):
        from groundedness import verify_response
        p = FakeProvider(["The lock-in period is 60 months."])
        final, meta = verify_response(
            "q", "The lock-in period is 48 months.", [src()], OnCfg(), p)
        self.assertEqual(final, "The lock-in period is 48 months.")
        self.assertEqual(meta["repairs"], 1)  # never a second repair
        self.assertEqual(meta["checks"], 2)
        self.assertFalse(meta["repaired"])
        self.assertEqual(meta["failure"], "still-ungrounded")
        self.assertEqual(len(p.calls), 1)

    def test_disabled_path_unchanged(self):
        from groundedness import maybe_verify_response, verify_response
        p = FakeProvider(["should never be used"])
        final, meta = verify_response("q", "anything 99 months.",
                                      [src()], Cfg(), p)
        self.assertEqual(final, "anything 99 months.")
        self.assertFalse(meta["groundedness_used"])
        self.assertEqual(p.calls, [])
        self.assertEqual(maybe_verify_response("q", "x", [src()], Cfg()),
                         "x")

    def test_no_evidence_no_repair(self):
        from groundedness import maybe_verify_response
        p = FakeProvider(["should never be used"])
        out = maybe_verify_response("q", "text", [], OnCfg())
        self.assertEqual(out, "text")

    def test_provenance_intact(self):
        from groundedness import (build_evidence_block,
                                  build_repair_prompt, check_groundedness)
        block = build_evidence_block([src()])
        self.assertIn("lease.pdf", block)
        self.assertIn("p. 4", block)
        f = check_groundedness("q", "The lock-in is 48 months.", EV)
        prompt, ctx = build_repair_prompt("q", "draft", f, block)
        self.assertIn("ONLY the evidence", prompt)
        self.assertIs(ctx, block)


class TestUntouched(unittest.TestCase):
    def test_retrieval_rerank_paths_intact(self):
        import config
        from postgres_vector_store import PostgresVectorStore
        config.reset_config_cache()
        cfg = config.load_config()
        self.assertEqual(cfg.groundedness_enabled, "0")
        self.assertEqual(cfg.retrieval_k, 5)
        self.assertAlmostEqual(cfg.relevance_threshold, 0.3)
        self.assertEqual(cfg.reranking_enabled, "0")
        self.assertEqual(cfg.retrieval_mode, "dense")
        store = PostgresVectorStore(dsn="x", embedding_function=None,
                                    config=cfg)
        self.assertFalse(store._resolve_rerank(None))
        self.assertEqual(store._resolve_mode(None), "dense")


if __name__ == "__main__":
    unittest.main()
