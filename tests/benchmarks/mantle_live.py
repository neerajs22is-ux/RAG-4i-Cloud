"""Live Mantle execution layer for the short exploratory benchmark.

Routes every paid call through ONE operator-side SSH channel to EC2,
where /tmp/mrun_live.py calls Bedrock Mantle (ap-south-1) with
instance-role SigV4 credentials. Local orchestration stays fully
deterministic: retrieval, prompts, validators, grading, budgets.

New file only: no existing file is modified. No model is ever
substituted: unknown IDs raise. A hard $8 program cap aborts BEFORE
any call that would breach it. Attested spend is computed from
MEASURED per-call usage; regional rates below are labeled ESTIMATE.
"""

import json
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

REGION = "ap-south-1"
EC2_USER_HOST = "ec2-user@3.90.188.164"
BUDGET_CAP_USD = 8.0
REMOTE_RUNNER = "python3 /tmp/mrun_live.py"

# ap-south-1 $/1M (in, out), cross-checked 2026-09-11 against AWS
# published regional tables. Labeled ESTIMATE in every report; token
# counts are MEASURED per call.
PRICES = {
    "qwen.qwen3-32b": (0.18, 0.71),
    "qwen.qwen3-235b-a22b-2507-v1:0": (0.26, 1.04),
    "openai.gpt-oss-120b": (0.18, 0.71),
    "mistral.mistral-large-3-675b-instruct": (0.59, 1.76),
    "mistral.devstral-2-123b": (0.48, 2.40),
}

# Runtime ID -> Mantle ID. Both recorded in every trace. Anything not
# listed here raises instead of being guessed.
MANTLE_IDS = {
    "qwen.qwen3-32b-v1:0": "qwen.qwen3-32b",
    "qwen.qwen3-235b-a22b-2507-v1:0": "qwen.qwen3-235b-a22b-2507-v1:0",
    "openai.gpt-oss-120b-1:0": "openai.gpt-oss-120b",
    "mistral.mistral-large-3-675b-instruct":
        "mistral.mistral-large-3-675b-instruct",
    "mistral.devstral-2-123b": "mistral.devstral-2-123b",
}

ARMS = {
    "ARM1": {"answer": "qwen.qwen3-235b-a22b-2507-v1:0",
             "planner": "qwen.qwen3-32b-v1:0",
             "reviewer": "qwen.qwen3-235b-a22b-2507-v1:0"},
    "ARM2": {"answer": "openai.gpt-oss-120b-1:0",
             "planner": "qwen.qwen3-32b-v1:0",
             "reviewer": "qwen.qwen3-235b-a22b-2507-v1:0"},
    "ARM3": {"answer": "mistral.mistral-large-3-675b-instruct",
             "planner": "qwen.qwen3-32b-v1:0",
             "reviewer": "qwen.qwen3-235b-a22b-2507-v1:0"},
    "ARM4": {"answer": "mistral.devstral-2-123b",
             "planner": "qwen.qwen3-32b-v1:0",
             "reviewer": "qwen.qwen3-235b-a22b-2507-v1:0"},
    "ARM5": {"answer": "openai.gpt-oss-120b-1:0",
             "planner": "qwen.qwen3-32b-v1:0",
             "reviewer": "qwen.qwen3-32b-v1:0"},
}


class BudgetExceeded(Exception):
    pass


class SpendTracker:
    """Counts every paid call before it happens. Hard $8 cap."""

    def __init__(self, cap_usd=BUDGET_CAP_USD):
        self.cap = float(cap_usd)
        self.spent_estimate = 0.0
        self.calls = 0
        self.retries = 0
        self.by_model = {}
        self.call_log = []

    def price_for(self, mantle_id):
        for key, price in PRICES.items():
            if mantle_id == key or mantle_id.startswith(key):
                return price
        raise ValueError("unpriced model %r: STOP, do not guess"
                         % (mantle_id,))

    def check(self, projected_usd):
        if self.spent_estimate + projected_usd > self.cap:
            raise BudgetExceeded(
                "cap $%.2f would breach: spent $%.4f + projected $%.4f"
                % (self.cap, self.spent_estimate, projected_usd))

    def worst_call_cost(self, mantle_id, in_tok=6000, out_tok=1000):
        per_in, per_out = self.price_for(mantle_id)
        return in_tok * per_in / 1e6 + out_tok * per_out / 1e6

    def record(self, mantle_id, usage, timers=None, scenario_id=None,
               arm=None, mode=None, role=None):
        per_in, per_out = self.price_for(mantle_id)
        tin = (usage or {}).get("in") or 0
        tout = (usage or {}).get("out") or 0
        cost = tin * per_in / 1e6 + tout * per_out / 1e6
        self.spent_estimate += cost
        self.calls += 1
        entry = self.by_model.setdefault(
            mantle_id, {"calls": 0, "tokens_in": 0, "tokens_out": 0,
                        "cost_usd": 0.0})
        entry["calls"] += 1
        entry["tokens_in"] += tin
        entry["tokens_out"] += tout
        entry["cost_usd"] += cost
        self.call_log.append({"seq": self.calls, "arm": arm, "mode": mode,
                              "role": role, "model": mantle_id,
                              "scenario_id": scenario_id,
                              "tokens_in": tin, "tokens_out": tout,
                              "cost_usd": round(cost, 6),
                              "timers": timers or {}})
        return cost

    def save(self, path):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"spent_estimate": self.spent_estimate,
                       "calls": self.calls, "retries": self.retries,
                       "by_model": self.by_model,
                       "call_log": self.call_log}, f, indent=1)

    def load(self, path):
        import json
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        self.spent_estimate = float(state.get("spent_estimate", 0.0))
        self.calls = int(state.get("calls", 0))
        self.retries = int(state.get("retries", 0))
        self.by_model = state.get("by_model", {})
        self.call_log = state.get("call_log", [])
        # Re-sequence from the loaded log so seq stays unique.
        if self.call_log:
            self.calls = max(self.calls,
                             max(c.get("seq", 0) for c in self.call_log))
        return self


class SSHTransport:
    """Exactly one remote exec per Mantle call. No multiplexing needed."""

    def __init__(self, key_path, host=EC2_USER_HOST, timeout_s=180,
                 refresh_cmd=None, refresh_interval_s=45):
        self.key_path = key_path
        self.host = host
        self.timeout_s = timeout_s
        self.calls = 0
        # Optional access plumbing: EIC pushes expire after 60s, so a
        # long run re-pushes automatically. This is an auth call, not an
        # infrastructure/IAM/quota change. Runs at most once per
        # refresh_interval_s, never per model call on failure paths.
        self.refresh_cmd = refresh_cmd
        self.refresh_interval_s = refresh_interval_s
        self._last_refresh = 0.0

    def _refresh_if_due(self):
        if not self.refresh_cmd:
            return
        import time
        now = time.monotonic()
        if now - self._last_refresh < self.refresh_interval_s:
            return
        proc = subprocess.run(
            self.refresh_cmd, capture_output=True, text=True, timeout=90)
        if proc.returncode != 0:
            raise RuntimeError("EIC key refresh failed rc=%d: %s" % (
                proc.returncode, (proc.stderr or proc.stdout or "")[:200]))
        self._last_refresh = time.monotonic()

    def exec_remote(self, remote_cmd, input_text=None):
        """Run one remote command; returns (stdout, transport_ms)."""
        import time
        self._refresh_if_due()
        t0 = time.monotonic()
        proc = subprocess.run(
            ["ssh", "-i", self.key_path, "-o", "BatchMode=yes",
             "-o", "ConnectTimeout=12", "-o", "StrictHostKeyChecking=no",
             self.host, remote_cmd],
            input=input_text, capture_output=True, text=True,
            timeout=self.timeout_s)
        ms = max(0, int((time.monotonic() - t0) * 1000))
        self.calls += 1
        if proc.returncode != 0:
            raise RuntimeError("ssh exec failed rc=%d: %s"
                               % (proc.returncode,
                                  (proc.stderr or "")[:300]))
        return proc.stdout, ms


def mantle_request(transport, payload):
    """Exactly one remote Mantle call. Returns the envelope dict."""
    out, transport_ms = transport.exec_remote(
        "python3 /tmp/mrun_live.py", input_text=json.dumps(payload))
    try:
        env = json.loads(out.strip().splitlines()[-1])
    except Exception as e:
        raise RuntimeError("unparseable Mantle envelope: %s | out=%s"
                           % (e, out[:300]))
    env["transport_ms"] = transport_ms
    return env


class CallContext:
    """Thread-local-ish attribution for the currently executing call."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.arm = None
        self.mode = None
        self.role = None
        self.scenario_id = None


CTX = CallContext()


class MantleChatProvider:
    """LLMProvider over Mantle Chat Completions (answer roles)."""

    def __init__(self, runtime_model_id, transport, tracker,
                 temperature=0.0, max_tokens=100):
        if runtime_model_id not in MANTLE_IDS:
            raise ValueError("unresolved Mantle ID for %r: refusing to "
                             "guess" % (runtime_model_id,))
        self.runtime_model_id = runtime_model_id
        self.mantle_model_id = MANTLE_IDS[runtime_model_id]
        self.transport = transport
        self.tracker = tracker
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._reachable = False
        self._last_ttft_ms = None
        self._last_usage = None

    @property
    def describe(self):
        return "Mantle(%s)" % self.mantle_model_id

    def _call(self, messages, stream, max_tokens=None):
        self.tracker.check(self.tracker.worst_call_cost(
            self.mantle_model_id))
        env = mantle_request(self.transport, {
            "model": self.mantle_model_id,
            "messages": messages,
            "max_tokens": self.max_tokens if max_tokens is None
            else max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        })
        if not env.get("ok"):
            raise RuntimeError("mantle error: %s: %s" % (
                env.get("error_type"), (env.get("error") or "")[:300]))
        self._reachable = True
        self.tracker.record(
            self.mantle_model_id, env.get("usage"),
            timers={"ttft_ms": env.get("ttft_ms"),
                    "total_ms": env.get("total_ms"),
                    "transport_ms": env.get("transport_ms")},
            scenario_id=CTX.scenario_id, arm=CTX.arm, mode=CTX.mode,
            role="answer")
        return env

    def generate(self, context, question, prompt_template):
        prompt = prompt_template.format(context=context, question=question)
        env = self._call(
            [{"role": "user", "content": prompt}], stream=False)
        return env.get("text") or ""

    def generate_stream(self, context, question, prompt_template):
        prompt = prompt_template.format(context=context, question=question)
        env = self._call(
            [{"role": "user", "content": prompt}], stream=True)
        self._last_ttft_ms = env.get("ttft_ms")
        self._last_usage = env.get("usage")
        yield env.get("text") or ""

    def is_reachable(self, timeout=3.0):
        # Latched by real calls only; never spends to probe.
        return self._reachable


def mantle_structured_fn(mantle_model_id, transport, tracker,
                         role="planner"):
    """structured_fn(prompt, schema) -> dict over Mantle tools path."""
    tracker.price_for(mantle_model_id)  # raises if unpriced

    def fn(prompt_text, schema):
        tracker.check(tracker.worst_call_cost(mantle_model_id))
        env = mantle_request(transport, {
            "model": mantle_model_id,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": 800,
            "temperature": 0,
            "stream": False,
            "tools": [{"type": "function",
                       "function": {"name": "emit",
                                    "description": "Emit the result "
                                    "object.",
                                    "parameters": schema}}],
            "tool_choice": "required",
        })
        if not env.get("ok"):
            raise RuntimeError("mantle structured error: %s" % (
                (env.get("error") or "")[:300],))
        tracker.record(
            mantle_model_id, env.get("usage"),
            timers={"total_ms": env.get("total_ms"),
                    "transport_ms": env.get("transport_ms")},
            scenario_id=CTX.scenario_id, arm=CTX.arm, mode=CTX.mode,
            role=role)
        args = env.get("tool_args")
        if not isinstance(args, dict):
            raise ValueError("structured output was not an object")
        return args

    return fn


def _tmp_key():
    fd, path = tempfile.mkstemp(prefix="mm_short_")
    os.close(fd)
    os.unlink(path)
    return path
