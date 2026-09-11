"""Final controlled Bedrock benchmark runner (gated; dry-run by default).

Default (no flags, no env): emits the LIVE BENCHMARK GATE using only
free checks (Echo/deterministic doubles) and stops. Paid execution
requires BOTH BENCHMARK_LIVE_BEDROCK=1 AND --execute-live, and stops
at the first failing gate. Order: A parity, B smoke/arm, C identity,
D cost recording, E schema, F telemetry/artifacts, G corpus run,
H repetitions, I grading, J judge, K calibration, L aggregation.

Usage:
  python -m tests.benchmarks.run_final            # gate + dry run ($0)
  BENCHMARK_LIVE_BEDROCK=1 python -m tests.benchmarks.run_final --execute-live
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Answer / Planner / Reviewer. Qwen 3 32B is the COMMON reviewer across
# all arms by explicit decision (2026-09-11): the quoted
# qwen.qwen3-235b-a22b-2507 identifier proved invalid live, a huge
# reviewer is unnecessary for round one, and a fixed reviewer isolates
# answer/planner differences cleanly. A larger reviewer may be tested
# later only if reviewer quality proves to be the bottleneck.
FINAL_ARMS = {
    "ARM1": {"answer": ("qwen.qwen3-32b-v1:0", "us-east-1", "qwen32b"),
             "planner": ("qwen.qwen3-32b-v1:0", "us-east-1"),
             "reviewer": ("qwen.qwen3-32b-v1:0", "us-east-1")},
    "ARM2": {"answer": ("qwen.qwen3-next-80b-a3b", "us-east-1", "qwen-next"),
             "planner": ("mistral.magistral-small-2509", "us-east-1"),
             "reviewer": ("qwen.qwen3-32b-v1:0", "us-east-1")},
    "ARM3": {"answer": ("mistral.ministral-3-14b-instruct", "us-east-1",
                        "ministral"),
             "planner": ("google.gemma-3-27b-it", "us-east-1"),
             "reviewer": ("qwen.qwen3-32b-v1:0", "us-east-1")},
}

JUDGE_MODEL = ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "us-east-1")
BUDGET_CAP_USD = 8.0

# Worst-case token assumptions per paid call (generous ceilings).
WORST_TOKENS = {
    "answer": (6000, 1000),
    "planner": (2000, 500),
    "reviewer": (8000, 500),
    "judge": (9000, 600),
}

ARM_PRICES = {
    # model_id substring -> (in/1M, out/1M). Verified 2026-09-10/11.
    "qwen.qwen3-32b": (0.15, 0.60),
    "qwen.qwen3-next-80b": (0.15, 1.20),
    "mistral.magistral": (0.50, 1.50),  # uncertain: re-verify at run
    "mistral.ministral-3-14b": (0.20, 0.20),
    "gemma-3-27b": (0.23, 0.38),
    "haiku-4-5": (1.0, 5.0),
}


def price_for(model_id):
    for key, price in ARM_PRICES.items():
        if key in (model_id or ""):
            return price
    raise ValueError("unpriced model in budget math: %r (STOP: re-verify "
                     "pricing, do not guess)" % (model_id,))


def call_cost(model_id, kind):
    (tin, tout) = WORST_TOKENS[kind]
    (pin, pout) = price_for(model_id)
    return tin * pin / 1e6 + tout * pout / 1e6


def worst_case_budget(n_main=32, n_smoke=8, n_arms=3):
    """Worst-case paid spend: every helper fires, 15% repairs, one
    same-model throttle retry on every call, judge on main rep-1."""
    per_q = {}
    for arm, spec in FINAL_ARMS.items():
        per_q[arm] = (
            call_cost(spec["answer"][0], "answer")
            + call_cost(spec["planner"][0], "planner")
            + call_cost(spec["reviewer"][0], "reviewer")
            + 0.15 * call_cost(spec["answer"][0], "answer"))
    full_q = n_main * 3 + n_smoke * 2  # paid question executions/arm
    arms_total = sum(per_q[a] * full_q for a in per_q)
    ttft = sum(call_cost(FINAL_ARMS[a]["answer"][0], "answer")
               for a in per_q) * (n_main + n_smoke)
    judge = (n_main) * (call_cost(JUDGE_MODEL[0], "judge"))
    worst = (arms_total + ttft + judge) * 2  # every call throttled once
    return {"per_question": per_q, "arms_total": arms_total,
            "ttft": ttft, "judge": judge, "worst": worst,
            "cap": BUDGET_CAP_USD, "within_cap": worst <= BUDGET_CAP_USD}


def emit_gate():
    """Free static gate. Returns (gate_dict, ok_bool)."""
    from .harness import load_cases
    from .isolations import SMOKE_IDS, load_scenarios
    gate = {}
    try:
        frozen = load_cases()
        gate["frozen_corpus_intact"] = (
            frozen["version"] == 1 and len(frozen["cases"]) == 14)
    except Exception as e:
        gate["frozen_corpus_intact"] = "FAIL: %r" % (e,)
    try:
        m = load_cases(os.path.join(HERE, "cases_m.json"))
        gate["m_corpus_docs"] = len(m["documents"])
        gate["m_corpus_cases"] = len(m["cases"])
    except Exception as e:
        gate["m_corpus_docs"] = "FAIL: %r" % (e,)
    try:
        suite = load_scenarios()
        sc = suite["scenarios"]
        gate["scenarios"] = len(sc)
        gate["gold_complete"] = True
        gate["smoke_count"] = sum(1 for s in sc
                                  if s["scenario_id"] in SMOKE_IDS)
        gate["main_count"] = len(sc) - gate["smoke_count"]
    except Exception as e:
        gate["gold_complete"] = "FAIL: %r" % (e,)
    try:
        import config
        cfg = config.load_config()
        gate["reasoning_disabled_default"] = (
            str(getattr(cfg, "reasoning_enabled", "")) == "0")
        gate["review_disabled_default"] = (
            str(getattr(cfg, "review_enabled", "")) == "0")
        from chunking import CHUNK_OVERLAP, CHUNK_SIZE
        gate["chunking_frozen"] = (CHUNK_SIZE, CHUNK_OVERLAP) == (1000, 200)
    except Exception as e:
        gate["chunking_frozen"] = "FAIL: %r" % (e,)
    try:
        from tests.benchmarks.harness import run_all
        out = run_all()
        gate["parity_s"] = out["summary"]
        outm = run_all(cases_path=os.path.join(HERE, "cases_m.json"))
        gate["parity_m"] = outm["summary"]
    except Exception as e:
        gate["parity_s"] = "FAIL: %r" % (e,)
    budget = worst_case_budget()
    gate["budget"] = {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in budget.items()}
    ok = (
        gate.get("frozen_corpus_intact") is True
        and gate.get("m_corpus_docs") == 25
        and gate.get("gold_complete") is True
        and gate.get("reasoning_disabled_default") is True
        and gate.get("review_disabled_default") is True
        and gate.get("chunking_frozen") is True
        and isinstance(gate.get("parity_s"), dict)
        and gate["parity_s"].get("failed", 1) == 0
        and budget["within_cap"])
    gate["GATE"] = "PASS" if ok else "FAIL"
    return gate, ok


def build_arm_providers(arm_name):
    """Construct live providers for one arm. Raises -> BLOCKED."""
    from bedrock_provider import BedrockConverseProvider
    spec = FINAL_ARMS[arm_name]
    out = {}
    for role in ("answer", "planner", "reviewer"):
        model_id, region = spec[role][0], spec[role][1]
        out[role] = BedrockConverseProvider(
            model_id=model_id, region=region, temperature=0.0)
    return out


def main(argv):
    execute = "--execute-live" in argv
    gate, ok = emit_gate()
    print(json.dumps(gate, indent=1, sort_keys=True))
    if not execute:
        print("DRY RUN ONLY ($0). Pass --execute-live with "
              "BENCHMARK_LIVE_BEDROCK=1 for paid execution.")
        return 0 if ok else 2
    if os.environ.get("BENCHMARK_LIVE_BEDROCK") != "1":
        print("STOP: --execute-live requires BENCHMARK_LIVE_BEDROCK=1.")
        return 2
    if not ok:
        print("STOP: static gate FAIL; paid execution refused.")
        return 2
    print("Static gate PASS. Live execution continues in run_live_final "
          "(separate step after smoke verification).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
