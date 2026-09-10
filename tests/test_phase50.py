"""Phase 5.0 foundation tests: benchmark fixtures, harness determinism,
metrics math, thresholds schema, and build/architecture invariants.

Deterministic only. No live AWS calls, no model downloads.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BENCH = os.path.join(os.path.dirname(__file__), "benchmarks")
ROOT = os.path.join(os.path.dirname(__file__), "..")
CATEGORIES = {"normal", "conversational", "partial", "unsupported",
              "followup", "comparison", "extraction", "summary",
              "ambiguous", "session-upload"}


class TestBenchmarkFixtures(unittest.TestCase):
    def test_cases_load_and_cover_all_categories(self):
        from tests.benchmarks.harness import CATEGORIES as H_CATS, load_cases
        data = load_cases()
        self.assertEqual(H_CATS, CATEGORIES)
        covered = {c["category"] for c in data["cases"]}
        self.assertEqual(covered, CATEGORIES)

    def test_no_real_personal_data_in_fixtures(self):
        from tests.benchmarks.harness import load_cases
        data = load_cases()
        blob = json.dumps(data)
        for marker in ("@example.com", "@gmail.com", "4111-1111",
                       "+91-98", "Aadhaar", "(This is a real"):
            self.assertNotIn(marker, blob)

    def test_each_case_has_evaluable_expectations(self):
        from tests.benchmarks.harness import load_cases
        for c in load_cases()["cases"]:
            self.assertIn("workflow", c["expect"], c["id"])
            self.assertTrue(c["query"].strip(), c["id"])
            self.assertTrue(c["documents"], c["id"])

    def test_session_cases_are_gated_not_live(self):
        from tests.benchmarks.harness import load_cases, run_all
        data = load_cases()
        gated = [c for c in data["cases"] if c.get("requires") == "5C"]
        self.assertTrue(gated)
        out = run_all()
        for r in out["results"]:
            if r["id"] in {c["id"] for c in gated}:
                self.assertEqual(r["status"], "skipped")
                self.assertIn("5C", r["reason"])


class TestHarnessDeterminism(unittest.TestCase):
    def test_repeated_runs_agree(self):
        from tests.benchmarks.harness import run_all
        first = run_all()
        second = run_all()

        def scrub(results):
            return [{k: (v if k != "metrics" else
                         {mk: mv for mk, mv in v.items()
                          if mk not in ("wall_ms", "retrieval_ms",
                                        "support_ms", "generation_ms",
                                        "total_ms")})
                     for k, v in r.items()} for r in results]

        self.assertEqual(scrub(first["results"]), scrub(second["results"]))
        self.assertEqual(first["summary"]["total"], second["summary"]["total"])

    def test_baseline_behaves(self):
        from tests.benchmarks.harness import run_all
        out = run_all()
        by_id = {r["id"]: r for r in out["results"]}
        # Deterministic doubles: grounded paths pass, gated paths skip.
        self.assertEqual(by_id["normal-lockin"]["status"], "pass")
        self.assertEqual(by_id["extract-notice-periods"]["status"], "pass")
        self.assertEqual(by_id["conv-greeting"]["status"], "pass")
        self.assertGreaterEqual(out["summary"]["passed"], 8)

    def test_no_document_text_in_results(self):
        from tests.benchmarks.harness import run_all
        blob = json.dumps(run_all()["results"]).lower()
        self.assertNotIn("lock-in period in the lease deed is 36 months", blob)


class TestMetricsMath(unittest.TestCase):
    def test_cost_math(self):
        from tests.benchmarks.metrics import estimate_cost_usd
        cost, how = estimate_cost_usd("haiku-4.5", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 6.0)
        self.assertEqual(how, "estimate")
        cost, how = estimate_cost_usd("nope", 10, 10)
        self.assertIsNone(cost)

    def test_token_estimate_shape(self):
        from tests.benchmarks.metrics import estimate_tokens
        n, how = estimate_tokens("hello world, this is a test")
        self.assertGreater(n, 0)
        self.assertIn(how, ("tiktoken-cl100k", "chars-div-4"))

    def test_compute_bundle_keys(self):
        from tests.benchmarks.metrics import compute
        m = compute("ans", [], {"timings": {}}, {}, model_key="echo")
        for key in ("answer_chars", "source_count", "guard_flagged",
                    "tokens_in", "tokens_out", "cost_usd", "workflow",
                    "planner_invoked", "support"):
            self.assertIn(key, m)
        self.assertFalse(m["planner_invoked"])
        self.assertEqual(m["cost_usd"], 0.0)


class TestThresholdsConfig(unittest.TestCase):
    def test_thresholds_schema(self):
        path = os.path.join(BENCH, "thresholds.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["version"], 1)
        self.assertTrue(data["gates"])
        for g in data["gates"]:
            for key in ("metric", "operator", "rationale", "stage", "status"):
                self.assertIn(key, g, g)
            self.assertIn(g["status"], ("decided", "pending-calibration"))
            if g["status"] == "pending-calibration":
                self.assertIsNone(g["value"], g)

    def test_hard_gates_decided(self):
        path = os.path.join(BENCH, "thresholds.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        by_metric = {g["metric"]: g for g in data["gates"]}
        self.assertEqual(by_metric["cross-session chunk leakage"]["value"], 0)
        self.assertEqual(by_metric["benchmark pass rate"]["value"], 1.0)


class TestPhase50Invariants(unittest.TestCase):
    def test_setup_pins_python311(self):
        with open(os.path.join(ROOT, "deploy", "setup_ec2.sh"),
                  encoding="utf-8") as f:
            script = f.read()
        self.assertIn("python3.11", script)

    def test_llm_factory_seam_exists(self):
        # 5A extends get_llm_provider dispatch; factory + ABC must exist.
        import inspect
        import llm_provider
        self.assertTrue(hasattr(llm_provider, "get_llm_provider"))
        self.assertTrue(hasattr(llm_provider, "LLMProvider"))
        sig = inspect.signature(llm_provider.LLMProvider.generate)
        self.assertIn("prompt_template", sig.parameters)

    def test_no_cloud_provider_wired_yet(self):
        # 5.0 is foundation only: no Bedrock/cloud imports in app code.
        # (git grep exits 1 on no-match, which is the expected outcome.)
        import subprocess
        try:
            out = subprocess.check_output(
                ["git", "grep", "-l", "-i", "langchain_aws|bedrock",
                 "--", "*.py", ":!tests/test_phase50.py"],
                cwd=ROOT, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        except subprocess.CalledProcessError as e:
            self.assertEqual(e.returncode, 1, e.output)
            out = ""
        self.assertEqual(out, "", f"cloud provider already wired: {out}")

    def test_frozen_pipeline_files_untouched_by_50(self):
        # 5.0 adds docs/scaffold only: backend/prompts/router/support stay
        # exactly as the 4.12 baseline tree has them (checked vs HEAD).
        import subprocess
        out = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--",
             "backend.py", "query_router.py", "answer_support.py",
             "workflows.py", "config.py", "llm_provider.py"],
            cwd=ROOT).decode("utf-8").strip()
        self.assertEqual(out, "", f"pipeline files touched: {out}")


if __name__ == "__main__":
    unittest.main()
