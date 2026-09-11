"""Live-arm benchmark preparation tests (no live AWS calls).

Validates the M-corpus, arm definitions, pricing keys, and the loud
gating of live helpers. BENCHMARK_LIVE_BEDROCK is never set to 1 here.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

M_CORPUS = os.path.join(os.path.dirname(__file__), "benchmarks",
                        "cases_m.json")


class TestMCorpus(unittest.TestCase):
    def test_loads_and_validates(self):
        from tests.benchmarks.harness import load_cases
        data = load_cases(M_CORPUS)
        self.assertEqual(data["version"], 1)
        self.assertGreaterEqual(len(data["cases"]), 20)
        self.assertLessEqual(len(data["cases"]), 30)
        self.assertGreaterEqual(len(data["documents"]), 25)

    def test_frozen_baseline_untouched(self):
        from tests.benchmarks.harness import load_cases
        data = load_cases()
        self.assertEqual(data["version"], 1)
        self.assertEqual(len(data["cases"]), 14)

    def test_echo_run_passes(self):
        from tests.benchmarks.harness import run_all
        out = run_all(cases_path=M_CORPUS)
        self.assertEqual(out["summary"]["failed"], 0)
        self.assertGreater(out["summary"]["passed"], 0)

    def test_categories_cover_brief(self):
        from tests.benchmarks.harness import load_cases
        data = load_cases(M_CORPUS)
        cats = {c["category"] for c in data["cases"]}
        for required in ("normal", "comparison", "summary", "extraction",
                         "ambiguous", "session-upload", "followup",
                         "unsupported", "conversational"):
            self.assertIn(required, cats)


class TestPricingKeys(unittest.TestCase):
    def test_arm_model_keys_priced(self):
        from tests.benchmarks.metrics import estimate_cost_usd
        for key in ("haiku-4.5", "sonnet-5", "luna"):
            cost, how = estimate_cost_usd(key, 4000, 600)
            self.assertIsNotNone(cost, key)
            self.assertGreater(cost, 0)
            self.assertEqual(how, "estimate")

    def test_luna_cheaper_than_haiku(self):
        from tests.benchmarks.metrics import estimate_cost_usd
        luna, _ = estimate_cost_usd("luna", 4000, 600)
        haiku, _ = estimate_cost_usd("haiku-4.5", 4000, 600)
        self.assertLess(luna, haiku)


class TestArmDefs(unittest.TestCase):
    def test_exact_ids(self):
        from tests.benchmarks.harness import LIVE_ARMS
        self.assertEqual(
            LIVE_ARMS["A"]["answer_model"],
            "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        self.assertEqual(LIVE_ARMS["B"]["answer_model"],
                         "us.anthropic.claude-sonnet-5")
        self.assertEqual(LIVE_ARMS["C"]["answer_model"],
                         "in.openai.gpt-5.6-luna")
        self.assertEqual(LIVE_ARMS["C"]["answer_region"], "ap-south-1")
        for arm in "ABC":
            for role in ("answer", "reasoning", "reviewer"):
                self.assertTrue(LIVE_ARMS[arm][role + "_model"])
                self.assertTrue(LIVE_ARMS[arm][role + "_region"])

    def test_no_global_profiles(self):
        from tests.benchmarks.harness import LIVE_ARMS
        for arm, spec in LIVE_ARMS.items():
            for role in ("answer", "reasoning", "reviewer"):
                self.assertNotIn(
                    "global.", spec[role + "_model"], (arm, role))


class TestLiveGating(unittest.TestCase):
    def test_helpers_none_without_flag(self):
        from tests.benchmarks.harness import (live_bedrock_llm,
                                              live_bedrock_planner,
                                              live_bedrock_reviewer)
        old = os.environ.pop("BENCHMARK_LIVE_BEDROCK", None)
        try:
            self.assertIsNone(live_bedrock_llm())
            self.assertIsNone(live_bedrock_planner())
            self.assertIsNone(live_bedrock_reviewer())
        finally:
            if old is not None:
                os.environ["BENCHMARK_LIVE_BEDROCK"] = old

    def test_run_live_arm_refuses_without_flag(self):
        from tests.benchmarks.harness import run_live_arm
        old = os.environ.pop("BENCHMARK_LIVE_BEDROCK", None)
        try:
            with self.assertRaises(ValueError):
                run_live_arm("A")
        finally:
            if old is not None:
                os.environ["BENCHMARK_LIVE_BEDROCK"] = old

    def test_run_live_arm_rejects_unknown_arm(self):
        from tests.benchmarks.harness import run_live_arm
        os.environ["BENCHMARK_LIVE_BEDROCK"] = "1"
        try:
            with self.assertRaises(ValueError):
                run_live_arm("Z")
        finally:
            del os.environ["BENCHMARK_LIVE_BEDROCK"]

    def test_run_live_arm_needs_deps_when_flagged(self):
        # Flagged but langchain-aws absent locally: must raise (blocked),
        # never silently substitute another model.
        from tests.benchmarks.harness import run_live_arm
        import importlib.util
        if importlib.util.find_spec("langchain_aws") is not None:
            self.skipTest("langchain-aws installed here")
        os.environ["BENCHMARK_LIVE_BEDROCK"] = "1"
        try:
            with self.assertRaises(Exception):
                run_live_arm("A")
        finally:
            del os.environ["BENCHMARK_LIVE_BEDROCK"]


if __name__ == "__main__":
    unittest.main()
