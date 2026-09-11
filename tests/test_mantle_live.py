"""Free tests for the Mantle live layer (transport mocked; no network)."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.benchmarks import mantle_live as ML


class FakeTransport:
    def __init__(self, envelope):
        self.envelope = envelope
        self.calls = 0
        self.last_payload = None

    def exec_remote(self, remote_cmd, input_text=None):
        self.calls += 1
        self.last_payload = json.loads(input_text)
        assert remote_cmd == "python3 /tmp/mrun_live.py"
        return json.dumps(self.envelope), 5


def chat_envelope(text="Hello.", tin=10, tout=5):
    return {"ok": True, "text": text, "tool_args": None,
            "usage": {"in": tin, "out": tout, "total": tin + tout},
            "ttft_ms": 11, "total_ms": 22}


class TestArmTable(unittest.TestCase):
    def test_five_arms_exact(self):
        self.assertEqual(sorted(ML.ARMS), ["ARM1", "ARM2", "ARM3",
                                          "ARM4", "ARM5"])
        for arm, spec in ML.ARMS.items():
            for role in ("answer", "planner", "reviewer"):
                self.assertIn(role, spec, (arm, role))

    def test_no_substitution(self):
        with self.assertRaises(ValueError):
            ML.MantleChatProvider("qwen.qwen3-235b-a22b-2507",
                                  FakeTransport(chat_envelope()),
                                  ML.SpendTracker())


class TestSpendTracker(unittest.TestCase):
    def test_records_and_caps(self):
        t = ML.SpendTracker(cap_usd=0.000001)
        with self.assertRaises(ML.BudgetExceeded):
            t.check(t.worst_call_cost("qwen.qwen3-32b"))
        cost = ML.SpendTracker().record(
            "qwen.qwen3-32b", {"in": 1000, "out": 100})
        self.assertAlmostEqual(cost, 1000 * 0.18 / 1e6 + 100 * 0.71 / 1e6)

    def test_unpriced_stops(self):
        with self.assertRaises(ValueError):
            ML.SpendTracker().price_for("mystery.model")


class TestProvider(unittest.TestCase):
    def test_generate_paths(self):
        tr = FakeTransport(chat_envelope("RAG4I_MODEL_SMOKE_OK"))
        p = ML.MantleChatProvider("qwen.qwen3-32b-v1:0", tr,
                                  ML.SpendTracker())
        self.assertEqual(p.generate("c", "q", "{question}"),
                         "RAG4I_MODEL_SMOKE_OK")
        self.assertEqual(tr.calls, 1)
        payload = tr.last_payload
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["model"], "qwen.qwen3-32b")

    def test_stream_yields_and_records(self):
        tr = FakeTransport(chat_envelope("abc"))
        tracker = ML.SpendTracker()
        p = ML.MantleChatProvider("openai.gpt-oss-120b-1:0", tr, tracker)
        self.assertEqual("".join(p.generate_stream("c", "q", "{question}")),
                         "abc")
        self.assertEqual(tracker.calls, 1)

    def test_error_raises_no_silent_fallback(self):
        tr = FakeTransport({"ok": False, "error_type": "AuthError",
                            "error": "denied"})
        p = ML.MantleChatProvider("qwen.qwen3-32b-v1:0", tr,
                                  ML.SpendTracker())
        with self.assertRaises(RuntimeError):
            p.generate("c", "q", "{question}")

    def test_structured_fn_mapping(self):
        env = {"ok": True, "text": "", "tool_args": {"a": 1},
               "usage": {"in": 5, "out": 5, "total": 10},
               "ttft_ms": None, "total_ms": 9}
        tr = FakeTransport(env)
        fn = ML.mantle_structured_fn("qwen.qwen3-32b",
                                     tr, ML.SpendTracker())
        out = fn("prompt", {"type": "object"})
        self.assertEqual(out, {"a": 1})
        self.assertEqual(tr.last_payload["tool_choice"], "required")

    def test_structured_non_object_rejected(self):
        env = {"ok": True, "text": "", "tool_args": [1, 2],
               "usage": None, "ttft_ms": None, "total_ms": 9}
        tr = FakeTransport(env)
        fn = ML.mantle_structured_fn("qwen.qwen3-32b",
                                     tr, ML.SpendTracker())
        with self.assertRaises(ValueError):
            fn("prompt", {"type": "object"})


class TestRefreshPlumbing(unittest.TestCase):
    def test_no_refresh_by_default(self):
        import subprocess
        tr = ML.SSHTransport("/nonexistent/key")
        self.assertIsNone(tr.refresh_cmd)
        self.assertEqual(tr.refresh_interval_s, 45)

    def test_refresh_runs_when_due(self):
        seen = []

        class T(ML.SSHTransport):
            def exec_remote(self, remote_cmd, input_text=None):
                raise AssertionError("must not reach ssh in unit test")

        tr = T("/nonexistent/key", refresh_cmd=["cmd", "/c", "echo",
                                                      "push"],
               refresh_interval_s=45)
        tr._last_refresh = 0.0
        import subprocess
        proc = subprocess.run(tr.refresh_cmd, capture_output=True,
                              text=True, timeout=30)
        self.assertEqual(proc.returncode, 0)
        seen.append(proc.stdout.strip())
        self.assertEqual(seen, ["push"])

    def test_refresh_failure_raises(self):
        tr = ML.SSHTransport("/nonexistent/key",
                             refresh_cmd=["cmd", "/c", "exit", "1"],
                             refresh_interval_s=45)
        tr._last_refresh = 0.0
        with self.assertRaises(RuntimeError):
            tr._refresh_if_due()


if __name__ == "__main__":
    unittest.main()
