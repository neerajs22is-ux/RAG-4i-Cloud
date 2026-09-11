"""Free tests for the final benchmark scaffolding (no live calls)."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSuiteGold(unittest.TestCase):
    def test_suite_loads_with_complete_gold(self):
        from tests.benchmarks.isolations import load_scenarios
        data = load_scenarios()
        self.assertEqual(len(data["scenarios"]), 40)
        self.assertTrue(data["scenarios"])

    def test_distribution(self):
        from tests.benchmarks.isolations import load_scenarios
        data = load_scenarios()
        sc = data["scenarios"]
        main = [s for s in sc if s["repetition"] == "main"]
        smoke = [s for s in sc if s["repetition"] == "smoke"]
        self.assertEqual(len(main) + len(smoke), 40)
        self.assertGreaterEqual(len(main), 25)
        self.assertGreaterEqual(len(smoke), 8)

    def test_severity_values(self):
        from tests.benchmarks.isolations import load_scenarios
        for sc in load_scenarios()["scenarios"]:
            self.assertIn(sc["severity"], ("cosmetic", "material",
                                           "critical"))


class TestGrader(unittest.TestCase):
    def _grade(self, **over):
        from tests.benchmarks.isolations import grade_deterministic
        scenario = {"query": "What is the lock-in period?",
                    "required_facts": ["36 months"],
                    "forbidden_claims": ["24 months"],
                    "expected_documents": ["lease-riverside.pdf"],
                    "expected_workflow": "normal",
                    "expected_support_level": "direct",
                    "expected_clarification": None,
                    "expected_scope": "persistent",
                    "severity": over.pop("severity", "material")}
        sources = [{"file_name": "lease-riverside.pdf",
                    "content": "Lock-in period is 36 months."}]
        info = {"workflow": "normal"}
        kw = dict(scenario=scenario, answer="The lock-in is 36 months.",
                  sources=sources, info=info, evidence=sources)
        kw.update(over)
        return grade_deterministic(**kw)

    def test_grounded_passes(self):
        grade = self._grade()
        self.assertTrue(grade["grounded"])
        self.assertEqual(grade["failures"], [])

    def test_missing_fact_and_forbidden(self):
        grade = self._grade(answer="The lock-in is 24 months.")
        self.assertFalse(grade["grounded"])
        cats = grade["failure_categories"]
        self.assertIn("MISSED_REQUESTED_ASPECT", cats)
        self.assertIn("UNSUPPORTED_CLAIM", cats)

    def test_wrong_document(self):
        grade = self._grade(
            sources=[{"file_name": "other.pdf", "content": "36 months"}])
        self.assertIn("WRONG_DOCUMENT", grade["failure_categories"])

    def test_critical_flagged(self):
        grade = self._grade(severity="critical",
                            answer="The lock-in is 24 months.")
        self.assertIn("CRITICAL", grade["failure_categories"])

    def test_rubber_stamp(self):
        grade = self._grade(
            answer="The lock-in is 24 months.",
            reviewer_out={"verdict": {"verdict": "pass"},
                          "schema_valid": True})
        self.assertTrue(grade["rubber_stamp"])
        self.assertIn("REVIEWER_RUBBER_STAMP",
                      grade["failure_categories"])

    def test_repair_bound(self):
        grade = self._grade(repair_count=2)
        self.assertFalse(grade["grounded"])

    def test_schema_failures(self):
        grade = self._grade(
            planner_out={"schema_valid": False, "failure": "schema"})
        self.assertIn("SCHEMA_FAILURE", grade["failure_categories"])


class TestJudgeValidator(unittest.TestCase):
    def good(self):
        return {"schema_version": 1, "overall_score": 7.5,
                "dimensions": {"clarity": 8, "completeness": 7,
                               "synthesis": 7, "usefulness": 8},
                "critical_failure": False, "rationale": "Clear and cited."}

    def test_accept(self):
        from tests.benchmarks.isolations import validate_judge
        clean, failure = validate_judge(self.good())
        self.assertIsNone(failure)
        self.assertEqual(clean["overall_score"], 7.5)

    def test_reject_shapes(self):
        from tests.benchmarks.isolations import validate_judge
        bad = self.good()
        del bad["rationale"]
        _, failure = validate_judge(bad)
        self.assertEqual(failure, "schema")
        _, failure = validate_judge("not json at all")
        self.assertEqual(failure, "schema")
        over = self.good()
        over["overall_score"] = 11
        _, failure = validate_judge(over)
        self.assertEqual(failure, "schema")
        long_r = self.good()
        long_r["rationale"] = "x" * 501
        _, failure = validate_judge(long_r)
        self.assertEqual(failure, "schema")

    def test_prompt_hides_identity(self):
        from tests.benchmarks.isolations import build_judge_input
        prompt = build_judge_input("Q?", "evidence text",
                                   {"B": "ans-b", "A": "ans-a"})
        self.assertNotIn("qwen", prompt.lower())
        self.assertNotIn("mistral", prompt.lower())
        self.assertNotIn("arm", prompt.lower())
        self.assertIn("[A]", prompt)
        self.assertIn("[B]", prompt)


class TestCalibrationPacket(unittest.TestCase):
    def test_pending_human_no_scores(self):
        from tests.benchmarks.isolations import calibration_packet
        results = [{"scenario_id": "S%02d" % i, "arm": "ARM1",
                    "run": 1, "question": "Q%d" % i,
                    "evidence_excerpt": "ev", "answer": "ans %d" % i}
                   for i in range(40)]
        packet = calibration_packet(results, seed=74154)
        self.assertEqual(packet["calibration_status"], "PENDING_HUMAN")
        frac = packet["sample_size"] / packet["population"]
        self.assertGreaterEqual(frac, 0.15)
        self.assertLessEqual(frac, 0.20)
        for item in packet["items"]:
            self.assertIsNone(item["human_overall_score"])
            self.assertIsNone(item["human_critical_failure"])


class TestBudget(unittest.TestCase):
    def test_worst_case_within_cap(self):
        from tests.benchmarks.run_final import worst_case_budget
        budget = worst_case_budget()
        self.assertTrue(budget["within_cap"], budget)
        self.assertLessEqual(budget["worst"], 8.0)

    def test_unpriced_model_stops(self):
        from tests.benchmarks.run_final import price_for
        with self.assertRaises(ValueError):
            price_for("mystery.model-xyz")


class TestGateFree(unittest.TestCase):
    def test_gate_passes_free(self):
        from tests.benchmarks.run_final import emit_gate
        gate, ok = emit_gate()
        self.assertTrue(ok, gate)
        self.assertEqual(gate["GATE"], "PASS")
        self.assertEqual(gate["m_corpus_docs"], 25)
        self.assertEqual(gate["scenarios"], 40)

    def test_common_reviewer_no_blocked(self):
        from tests.benchmarks.run_final import FINAL_ARMS, JUDGE_MODEL
        for arm, spec in FINAL_ARMS.items():
            self.assertEqual(spec["reviewer"][0],
                             "qwen.qwen3-32b-v1:0", arm)
            self.assertEqual(spec["reviewer"][1], "us-east-1", arm)
        # Judge stays outside every test arm (answer/planner/reviewer).
        judge_id = JUDGE_MODEL[0]
        for arm, spec in FINAL_ARMS.items():
            for role in ("answer", "planner", "reviewer"):
                self.assertNotEqual(spec[role][0], judge_id,
                                    (arm, role))


class TestIsolationRunners(unittest.TestCase):
    def _snap(self):
        from tests.benchmarks.harness import load_cases
        from tests.benchmarks.isolations import (load_scenarios,
                                                 snapshot_evidence)
        suite = load_scenarios()
        docs = load_cases("tests/benchmarks/cases_m.json")["documents"]
        sc = next(s for s in suite["scenarios"]
                  if s["scenario_id"] == "S01-normal-lockin")
        return snapshot_evidence(sc, docs)

    def test_snapshot_stable_ids(self):
        snap1 = self._snap()
        snap2 = self._snap()
        self.assertEqual([e["chunk_id"] for e in snap1["evidence"]],
                         [e["chunk_id"] for e in snap2["evidence"]])
        self.assertTrue(snap1["evidence"])

    def test_planner_runner_unpacks_tuple(self):
        from query_planner import DeterministicPlanner, PlanCache
        from tests.benchmarks.isolations import run_planner_isolation
        snap = self._snap()
        bundle = {"planner": DeterministicPlanner(), "cache": PlanCache(),
                  "provider": "d", "model": "n", "timeout_s": 5,
                  "_meta_base": {}}
        out = run_planner_isolation(
            bundle, snap, sorted({e["file_name"] for e in snap["evidence"]
                                  if e["file_name"]}))
        self.assertTrue(out["schema_valid"], out)
        self.assertIsNone(out["failure"])

    def test_reviewer_runner_unpacks_tuple(self):
        from answer_reviewer import DeterministicReviewer
        from tests.benchmarks.isolations import run_reviewer_isolation
        snap = self._snap()
        bundle = {"reviewer": DeterministicReviewer(), "provider": "d",
                  "model": "n", "timeout_s": 5, "_meta_base": {}}
        out = run_reviewer_isolation(bundle, snap, "An answer.",
                                     "normal")
        self.assertTrue(out["schema_valid"], out)
        self.assertEqual(out["verdict"]["verdict"], "pass")

    def test_answer_isolation_echo(self):
        from tests.benchmarks.harness import EchoLLM
        from tests.benchmarks.isolations import run_answer_isolation
        snap = self._snap()
        out = run_answer_isolation(EchoLLM(), snap, stream_probe=False)
        self.assertIn("36 months", out["answer"])
        self.assertEqual(out["evidence_ids"],
                         [e["chunk_id"] for e in snap["evidence"]])


if __name__ == "__main__":
    unittest.main()
