"""Targeted remediation validation (short01-remediation-v1).

Validates the two declared post-diagnosis changes BEFORE any full
benchmark. Exactly 9 paid inference calls, never retried:

  A. 3x GPT-OSS answer calls (max_tokens=512, OSS answer role only)
     on scenarios empty under the old 100-token cap.
  B. 3x Qwen3-32B planner calls (timeout_s=15).
  C. 3x Qwen3-235B reviewer calls (timeout_s=15).

Usage:
  DRY_RUN=1 python tests/benchmarks/run_remediation.py  # $0, doubles
  python tests/benchmarks/run_remediation.py            # LIVE ($1 cap)

New file only; run_short.py stays frozen as the short01 historical
runner. Artifacts go to results/20260911-remediation01/ and stay
uncommitted. The 20260911-short01 directory is never touched.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from tests.benchmarks import isolations as ISO  # noqa: E402
from tests.benchmarks.harness import EchoLLM, load_cases  # noqa: E402
from tests.benchmarks.mantle_live import (  # noqa: E402
    BENCHMARK_REVISION, CTX, DEFAULT_ANSWER_MAX_TOKENS,
    OSS_ANSWER_MAX_TOKENS, OSS_RUNTIME_IDS, STRUCTURED_TIMEOUT_S,
    BudgetExceeded, MantleChatProvider, SpendTracker,
    mantle_structured_fn,
)

OSS_MODEL = "openai.gpt-oss-120b-1:0"
PLANNER_MODEL = "qwen.qwen3-32b"
REVIEWER_MODEL = "qwen.qwen3-235b-a22b-2507-v1:0"

# Previously empty under the old 100-token OSS cap (answer isolation).
OSS_SCENARIOS = ["S04-sla", "S06-loan", "S16-compound"]
PLANNER_SCENARIOS = ["S01-normal-lockin", "S04-sla", "S06-loan"]
REVIEWER_SCENARIOS = ["S01-normal-lockin", "S04-sla", "S06-loan"]

# Hard stop: 9 tiny calls must never approach this.
VALIDATION_BUDGET_CAP_USD = 1.0


class StopRun(Exception):
    pass


def utc_stamp():
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def project_costs(tracker):
    """Worst-case projection for exactly the 9 planned calls."""
    oss = tracker.worst_call_cost("openai.gpt-oss-120b",
                                  in_tok=1000, out_tok=512)
    plan = tracker.worst_call_cost("qwen.qwen3-32b")
    rev = tracker.worst_call_cost("qwen.qwen3-235b-a22b-2507-v1:0")
    return {"oss_answer_each": oss, "planner_each": plan,
            "reviewer_each": rev,
            "total_worst": 3 * oss + 3 * plan + 3 * rev}


def frozen_arm1_answers(outdir_short):
    """Existing short01 ARM1 E2E answers (no new calls to refreeze)."""
    path = os.path.join(outdir_short, "trace-arm1.jsonl")
    frozen = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("scenario_id") and row.get("answer") is not None:
                frozen[row["scenario_id"]] = row["answer"]
    return frozen


def main():
    assert BENCHMARK_REVISION == "short01-remediation-v1"
    assert OSS_ANSWER_MAX_TOKENS == 512
    assert STRUCTURED_TIMEOUT_S == 15
    assert OSS_MODEL in OSS_RUNTIME_IDS
    dry = os.environ.get("DRY_RUN") == "1"
    outdir = os.path.join(HERE, "results", "20260911-remediation01")
    os.makedirs(outdir, exist_ok=True)
    short_dir = os.path.join(HERE, "results", "20260911-short01")

    tracker = SpendTracker(cap_usd=VALIDATION_BUDGET_CAP_USD)
    proj = project_costs(tracker)
    print("COST PROJECTION (worst case, 9 calls): %s"
          % json.dumps(proj, sort_keys=True))
    if proj["total_worst"] > VALIDATION_BUDGET_CAP_USD:
        raise StopRun("projected $%.4f exceeds $%.2f cap: STOP"
                      % (proj["total_worst"], VALIDATION_BUDGET_CAP_USD))

    transport = None
    if not dry:
        key = os.environ.get("SSH_KEY", os.path.expandvars(
            r"${TEMP}\mm_run"))
        key = os.path.expandvars(key)
        from tests.benchmarks.mantle_live import SSHTransport
        refresh = ["aws", "ec2-instance-connect", "send-ssh-public-key",
                   "--profile", "rag4i-dev", "--region", "us-east-1",
                   "--instance-id", "i-07dea54f4144b8fd0",
                   "--availability-zone", "us-east-1c",
                   "--instance-os-user", "ec2-user",
                   "--ssh-public-key", "file://" + key + ".pub"]
        transport = SSHTransport(key, refresh_cmd=refresh)

    suite = ISO.load_scenarios()
    docs = load_cases(os.path.join(HERE, "cases_m.json"))["documents"]
    by_id = {s["scenario_id"]: s for s in suite["scenarios"]}
    for sid in OSS_SCENARIOS + PLANNER_SCENARIOS + REVIEWER_SCENARIOS:
        assert sid in by_id, sid
    frozen = {} if dry else frozen_arm1_answers(short_dir)
    if not dry:
        for sid in REVIEWER_SCENARIOS:
            assert sid in frozen, "missing frozen ARM1 answer for " + sid

    from backend import EMPTY_RESPONSE_MESSAGE

    records = []

    def note(rec):
        rec["benchmark_revision"] = BENCHMARK_REVISION
        records.append(rec)
        ISO.write_trace(os.path.join(outdir, "trace-remediation.jsonl"),
                        rec)

    # ---- A. GPT-OSS answers: ONE non-stream call per scenario ----
    for sid in OSS_SCENARIOS:
        sc = by_id[sid]
        snap = ISO.snapshot_evidence(sc, docs)
        CTX.arm, CTX.mode, CTX.role, CTX.scenario_id = \
            ("REM-OSS", "R", "answer", sid)
        t0 = time.monotonic()
        try:
            if dry:
                out = ISO.run_answer_isolation(EchoLLM(), snap,
                                               stream_probe=False)
            else:
                out = ISO.run_answer_isolation(
                    MantleChatProvider(OSS_MODEL, transport, tracker),
                    snap, stream_probe=False)
            wall_ms = max(0, int((time.monotonic() - t0) * 1000))
            diag = out.get("diag") or {}
            note({"check": "oss-answer", "scenario_id": sid,
                  "question": sc["query"],
                  "answer_excerpt": (out.get("answer") or "")[:2000],
                  "non_empty": bool((out.get("answer") or "").strip()),
                  "empty_response_message":
                      out.get("answer") == EMPTY_RESPONSE_MESSAGE,
                  "finish_reason": diag.get("finish_reason"),
                  "has_reasoning": diag.get("has_reasoning"),
                  "usage_in": diag.get("usage_in"),
                  "usage_out": diag.get("usage_out"),
                  "max_tokens": diag.get("max_tokens"),
                  "wall_ms": wall_ms, "error": None})
        except Exception as e:  # no retry: record and move on
            wall_ms = max(0, int((time.monotonic() - t0) * 1000))
            note({"check": "oss-answer", "scenario_id": sid,
                  "question": sc["query"], "answer_excerpt": "",
                  "non_empty": False, "empty_response_message": False,
                  "finish_reason": None, "has_reasoning": None,
                  "usage_in": None, "usage_out": None,
                  "max_tokens": OSS_ANSWER_MAX_TOKENS,
                  "wall_ms": wall_ms,
                  "error": "%s: %s" % (type(e).__name__, str(e)[:200])})
        finally:
            CTX.reset()

    # ---- B. Planner (Qwen32, 15s) ----
    for sid in PLANNER_SCENARIOS:
        sc = by_id[sid]
        snap = ISO.snapshot_evidence(sc, docs)
        CTX.arm, CTX.mode, CTX.role, CTX.scenario_id = \
            ("REM-PLAN", "R", "planner", sid)
        try:
            from query_planner import (DeterministicPlanner, PlanCache,
                                       LLMQueryPlanner)
            if dry:
                pb = {"planner": DeterministicPlanner(),
                      "cache": PlanCache(), "provider": "d", "model": "n",
                      "timeout_s": STRUCTURED_TIMEOUT_S,
                      "_meta_base": {}}
            else:
                pb = {"planner": LLMQueryPlanner(
                    mantle_structured_fn(PLANNER_MODEL, transport,
                                         tracker, role="planner"),
                    timeout_s=STRUCTURED_TIMEOUT_S, provider="mantle",
                    model=PLANNER_MODEL),
                    "cache": PlanCache(), "provider": "mantle",
                    "model": PLANNER_MODEL,
                    "timeout_s": STRUCTURED_TIMEOUT_S}
            store_files = sorted({e.get("file_name")
                                  for e in snap["evidence"]
                                  if e.get("file_name")})
            pout = ISO.run_planner_isolation(pb, snap, store_files)
            note({"check": "planner", "scenario_id": sid,
                  "schema_valid": pout.get("schema_valid"),
                  "failure": pout.get("failure"),
                  "call_failure": pout.get("call_failure"),
                  "latency_ms": pout.get("latency_ms"), "error": None})
        except Exception as e:
            note({"check": "planner", "scenario_id": sid,
                  "schema_valid": False, "failure": None,
                  "call_failure": None, "latency_ms": None,
                  "error": "%s: %s" % (type(e).__name__, str(e)[:200])})
        finally:
            CTX.reset()

    # ---- C. Reviewer (Qwen235, 15s, frozen ARM1 answers) ----
    for sid in REVIEWER_SCENARIOS:
        sc = by_id[sid]
        snap = ISO.snapshot_evidence(sc, docs)
        CTX.arm, CTX.mode, CTX.role, CTX.scenario_id = \
            ("REM-REV", "R", "reviewer", sid)
        try:
            from answer_reviewer import (DeterministicReviewer,
                                         LLMAnswerReviewer)
            if dry:
                rb = {"reviewer": DeterministicReviewer(),
                      "provider": "d", "model": "n",
                      "timeout_s": STRUCTURED_TIMEOUT_S,
                      "_meta_base": {}}
                frozen_ans = "Evidence says: dry run."
            else:
                rb = {"reviewer": LLMAnswerReviewer(
                    mantle_structured_fn(REVIEWER_MODEL, transport,
                                         tracker, role="reviewer"),
                    timeout_s=STRUCTURED_TIMEOUT_S, provider="mantle",
                    model=REVIEWER_MODEL),
                    "provider": "mantle", "model": REVIEWER_MODEL,
                    "timeout_s": STRUCTURED_TIMEOUT_S}
                frozen_ans = frozen[sid]
            rout = ISO.run_reviewer_isolation(rb, snap, frozen_ans,
                                              "normal")
            note({"check": "reviewer", "scenario_id": sid,
                  "schema_valid": rout.get("schema_valid"),
                  "failure": rout.get("failure"),
                  "call_failure": rout.get("call_failure"),
                  "latency_ms": rout.get("latency_ms"),
                  "verdict": (rout.get("verdict") or {}).get("verdict"),
                  "error": None})
        except Exception as e:
            note({"check": "reviewer", "scenario_id": sid,
                  "schema_valid": False, "failure": None,
                  "call_failure": None, "latency_ms": None,
                  "verdict": None,
                  "error": "%s: %s" % (type(e).__name__, str(e)[:200])})
        finally:
            CTX.reset()

    # ---- artifacts (bounded excerpts only, never secrets) ----
    if not dry:
        assert tracker.calls == 9, \
            "expected exactly 9 paid calls, got %d" % tracker.calls
        tracker.save(os.path.join(outdir, "spend-remediation.json"))
    revision = {
        "benchmark_revision": BENCHMARK_REVISION,
        "declared": "AFTER root-cause diagnosis, BEFORE the full "
                    "benchmark. short01 results preserved untouched.",
        "changes": {
            "oss_answer_max_tokens": OSS_ANSWER_MAX_TOKENS,
            "oss_answer_scope": sorted(OSS_RUNTIME_IDS),
            "other_answer_max_tokens": DEFAULT_ANSWER_MAX_TOKENS,
            "planner_timeout_s": STRUCTURED_TIMEOUT_S,
            "reviewer_timeout_s": STRUCTURED_TIMEOUT_S,
            "trace_additions": ["diag{finish_reason,has_reasoning,"
                                "usage_in,usage_out,max_tokens}",
                                "call_failure"],
        },
        "why": {
            "oss_512": "OSS reasoning+answer share one completion "
                       "budget; empties correlated 1:1 with "
                       "usage_out==100 (old cap).",
            "timeout_15": "server inference 1.4-2.8s; fresh-SSH "
                          "round-trip 6.2-7.7s; 5s timed out "
                          "client-side while workers succeeded.",
            "call_failure": "validator verdicts on None masked "
                            "TIMEOUT as schema/malformed.",
        },
        "unchanged": ["corpus", "gold/scenarios", "retrieval/k/"
                      "threshold/embeddings/chunking", "model matrix",
                      "model IDs", "endpoint", "IAM", "prompts",
                      "planner/reviewer schemas", "scoring rubric",
                      "judge (not run)"],
    }
    with open(os.path.join(outdir, "revision.json"), "w",
              encoding="utf-8") as f:
        json.dump(revision, f, indent=1, sort_keys=True)

    def _lat(check):
        return [r.get("latency_ms", r.get("wall_ms")) for r in records
                if r.get("check") == check
                and isinstance(r.get("latency_ms", r.get("wall_ms")),
                               int)]

    def _wall(check):
        return [r.get("wall_ms") for r in records
                if r.get("check") == check
                and isinstance(r.get("wall_ms"), int)]

    latency = {c: {"latency_ms": _lat(c), "wall_ms": _wall(c)}
               for c in ("oss-answer", "planner", "reviewer")}
    with open(os.path.join(outdir, "latency.json"), "w",
              encoding="utf-8") as f:
        json.dump(latency, f, indent=1, sort_keys=True)
    cost = {"projection_worst_usd": proj,
            "spent_estimate_usd":
                round(tracker.spent_estimate, 6) if not dry else 0.0,
            "paid_calls": tracker.calls if not dry else 0,
            "by_model": tracker.by_model if not dry else {},
            "dry_run": dry}
    with open(os.path.join(outdir, "cost.json"), "w",
              encoding="utf-8") as f:
        json.dump(cost, f, indent=1, sort_keys=True)

    oss_recs = [r for r in records if r.get("check") == "oss-answer"]
    plan_recs = [r for r in records if r.get("check") == "planner"]
    rev_recs = [r for r in records if r.get("check") == "reviewer"]
    report = {
        "benchmark_revision": BENCHMARK_REVISION, "dry_run": dry,
        "oss": {"cases": len(oss_recs),
                "non_empty": sum(1 for r in oss_recs if r.get("non_empty")),
                "empty_response_messages": sum(
                    1 for r in oss_recs if r.get("empty_response_message")),
                "finish_reasons": [r.get("finish_reason")
                                   for r in oss_recs],
                "usage_out": [r.get("usage_out") for r in oss_recs],
                "errors": [r.get("error") for r in oss_recs
                           if r.get("error")]},
        "planner": {"cases": len(plan_recs),
                    "schema_valid": sum(
                        1 for r in plan_recs if r.get("schema_valid")),
                    "call_failures": [r.get("call_failure")
                                      for r in plan_recs],
                    "latency_ms": _lat("planner")},
        "reviewer": {"cases": len(rev_recs),
                     "schema_valid": sum(
                         1 for r in rev_recs if r.get("schema_valid")),
                     "call_failures": [r.get("call_failure")
                                       for r in rev_recs],
                     "latency_ms": _lat("reviewer")},
    }
    with open(os.path.join(outdir, "report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=1, sort_keys=True)

    leaked = []
    import re
    for name in sorted(os.listdir(outdir)):
        if not name.endswith(".json"):
            continue
        blob = open(os.path.join(outdir, name), encoding="utf-8").read()
        if re.findall(r"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16})", blob):
            leaked.append(name)
    if leaked:
        raise StopRun("CREDENTIALS IN ARTIFACTS: %s" % leaked)
    print(json.dumps(report, indent=1, sort_keys=True))
    return report


if __name__ == "__main__":
    try:
        main()
    except StopRun as e:
        print("STOP: %s" % e)
        raise SystemExit(3)
    except BudgetExceeded as e:
        print("STOP (budget): %s" % e)
        raise SystemExit(4)
