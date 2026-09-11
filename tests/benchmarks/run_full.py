"""Full controlled benchmark runner (Mumbai Mantle, 5 arms, 40 cases).

Revision short01-remediation-v1 (OSS answer 512 via provider default;
planner/reviewer STRUCTURED_TIMEOUT_S). run_short.py stays frozen as
the short01 historical runner; this file is the full-run vehicle.

Usage:
  DRY_RUN=1 python tests/benchmarks/run_full.py              # $0, doubles
  SCENARIOS=S01-normal-lockin python tests/benchmarks/run_full.py  # subset
  SSH_KEY=%TEMP%\\mm_diag python tests/benchmarks/run_full.py      # LIVE

LIVE paid run additionally needs BENCHMARK_LIVE_BEDROCK=1 (paid guard)
and honors the $8 SpendTracker cap BEFORE every call. Repetitions come
from the frozen suite: repetition==main -> 3, smoke -> 2. Traces are
append-only: re-running resumes (recorded scenario/rep pairs skip).

Artifacts: tests/benchmarks/results/<date>/final01/ (uncommitted).
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from tests.benchmarks import isolations as ISO  # noqa: E402
from tests.benchmarks.harness import EchoLLM, load_cases, run_case  # noqa: E402
from tests.benchmarks.mantle_live import (  # noqa: E402
    ARMS, BENCHMARK_REVISION, BUDGET_CAP_USD, CTX, MANTLE_IDS, PRICES,
    STRUCTURED_TIMEOUT_S, BudgetExceeded, MantleChatProvider,
    SpendTracker, mantle_structured_fn,
)

RUN_ID = "final01"

ANSWER_MODELS = ["qwen.qwen3-235b-a22b-2507-v1:0",
                 "openai.gpt-oss-120b-1:0",
                 "mistral.mistral-large-3-675b-instruct",
                 "mistral.devstral-2-123b"]
PLANNER_MODEL = "qwen.qwen3-32b-v1:0"
PLANNER_MANTLE = "qwen.qwen3-32b"
REVIEWER_MODELS = [("qwen.qwen3-235b-a22b-2507-v1:0", "R235"),
                   ("qwen.qwen3-32b", "R32")]

ACCOUNT_WIDE_MARKERS = ("Operation not allowed", "NoCredentials",
                        "credential", "throttl", "TooManyRequests",
                        "InternalServer", "ServiceUnavailable",
                        "AccessDenied", "not authorized")


class StopRun(Exception):
    pass


def _pid_alive(pid):
    """True if a process with this PID exists (stale-lock check)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_live_lock(outdir):
    """Single-writer guard: at most one live run per run directory.

    Writes outdir/.live.lock (PID). A lock held by a LIVE process
    refuses startup; a stale lock (PID dead, e.g. after a kill) is
    overwritten with a warning. Best-effort release on clean exit;
    stale locks are always safe to supersede, never deleted blindly.
    """
    path = os.path.join(outdir, ".live.lock")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                prior = json.load(f)
            old_pid = int(prior.get("pid", -1))
        except Exception:
            old_pid = -1
        if old_pid > 0 and _pid_alive(old_pid):
            raise StopRun(
                "another live benchmark holds %s (pid %d): STOP, "
                "refusing concurrent write to the same run directory"
                % (path, old_pid))
        print("WARNING: superseding stale live lock (pid %s)" % (old_pid,))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime())}, f)
    return path


def release_live_lock(path):
    try:
        with open(path, encoding="utf-8") as f:
            prior = json.load(f)
        if int(prior.get("pid", -1)) == os.getpid():
            os.remove(path)
    except Exception:
        pass


def check_account_wide(err_text):
    low = (err_text or "").lower()
    return any(m.lower() in low for m in ACCOUNT_WIDE_MARKERS)


def pick_key():
    """Throwaway EIC keypair path. Validated, never guessed silently."""
    import subprocess
    cands = []
    if os.environ.get("SSH_KEY"):
        cands.append(os.path.expandvars(os.environ["SSH_KEY"]))
    for name in ("mm_diag", "mm_run"):
        cands.append(os.path.join(os.path.expandvars("${TEMP}"), name))
    for key in cands:
        if not key or not os.path.exists(key) \
                or not os.path.exists(key + ".pub"):
            continue
        try:
            proc = subprocess.run(
                ["ssh-keygen", "-y", "-f", key], capture_output=True,
                text=True, timeout=30)
        except Exception:
            continue
        if proc.returncode == 0 and "ssh-" in (proc.stdout or ""):
            return key
    raise StopRun("no usable SSH key (checked SSH_KEY/mm_diag/mm_run): "
                  "generate one and retry; no paid calls made")


# Credible worst-case ceilings per paid call, grounded in short01
# MEASURED maxima (in<=900, out<=224 incl. OSS reasoning) with wide
# headroom. Generic 6000/1000 defaults would overstate ~2x and false-
# trip the $8 gate. Caps still bind: OSS answers can emit at most 512
# out; structured calls at most 800 out (remote request ceilings).
WORST_TOKENS = {
    "answer": (2000, 512),
    "planner": (2000, 800),
    "reviewer": (3000, 800),
}


def _worst(mantle_id, kind):
    tin, tout = WORST_TOKENS[kind]
    tracker = SpendTracker()
    per_in, per_out = tracker.price_for(mantle_id)
    return tin * per_in / 1e6 + tout * per_out / 1e6


def worst_projection(tracker, executions):
    """Worst-case USD for the planned executions (no calls made)."""
    per_exec = 0.0
    for mid in sorted(set(ANSWER_MODELS + ["qwen.qwen3-32b-v1:0"])):
        m = MANTLE_IDS.get(mid, mid)
        per_exec += 2 * _worst(m, "answer")
    per_exec += _worst(PLANNER_MANTLE, "planner")
    for rmid, _ in REVIEWER_MODELS:
        per_exec += _worst(rmid, "reviewer")
    for arm in sorted(ARMS):
        spec = ARMS[arm]
        ans_m = MANTLE_IDS.get(spec["answer"], spec["answer"])
        rev_m = MANTLE_IDS.get(spec["reviewer"], spec["reviewer"])
        per_exec += 2 * _worst(ans_m, "answer") + _worst(rev_m, "reviewer")
    return per_exec * executions


def main():
    dry = os.environ.get("DRY_RUN") == "1"
    live = os.environ.get("BENCHMARK_LIVE_BEDROCK") == "1"
    if not dry and not live:
        print("STOP: live run requires BENCHMARK_LIVE_BEDROCK=1 "
              "(or DRY_RUN=1 for the free pre-flight).")
        raise SystemExit(2)
    datestr = time.strftime("%Y%m%d", time.gmtime())
    # Dry/live separation: dry validation must NEVER write into the
    # live run directory (a past dry run contaminated final01 with
    # Echo rows that resume logic could not distinguish from live).
    outdir = os.path.join(HERE, "results", datestr,
                          RUN_ID if not dry else RUN_ID + "-DRY")
    os.makedirs(outdir, exist_ok=True)
    lock_path = None
    if not dry:
        lock_path = acquire_live_lock(outdir)

    from tests.benchmarks import mantle_live as ML
    transport = None
    tracker = SpendTracker(cap_usd=BUDGET_CAP_USD)
    if not dry:
        key = pick_key()
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
    assert len(docs) == 25, len(docs)
    scenarios = list(suite["scenarios"])
    assert len(scenarios) == 40, len(scenarios)
    only = [s.strip() for s in
            os.environ.get("SCENARIOS", "").split(",") if s.strip()]
    if only:
        wanted = set(only)
        known = {s["scenario_id"] for s in scenarios}
        assert wanted <= known, sorted(wanted - known)
        scenarios = [s for s in scenarios if s["scenario_id"] in wanted]

    def reps_for(sc):
        return 3 if sc.get("repetition") == "main" else 2

    total_exec = sum(reps_for(s) for s in scenarios)
    proj = worst_projection(tracker, total_exec)
    print("WORST-CASE PROJECTION: $%.4f over %d executions "
          "(cap $%.2f)" % (proj, total_exec, BUDGET_CAP_USD))
    if not dry and proj > BUDGET_CAP_USD:
        raise StopRun("worst-case $%.4f exceeds $%.2f cap: STOP, no calls "
                      "made" % (proj, BUDGET_CAP_USD))

    if not dry:
        with open(os.path.join(outdir, "config_snapshot.json"), "w",
                  encoding="utf-8") as f:
            import hashlib
            json.dump({
                "benchmark_revision": BENCHMARK_REVISION,
                "arms": ARMS, "mantle_ids": MANTLE_IDS,
                "prices": PRICES, "region": ML.REGION,
                "endpoint": "bedrock-mantle.ap-south-1.api.aws/v1",
                "scenarios": [s["scenario_id"] for s in
                              suite["scenarios"]],
                "chunk": [s["scenario_id"] for s in scenarios],
                "repetitions": {s["scenario_id"]: reps_for(s)
                                for s in scenarios},
                "seed": 74154, "dry_run": False,
                "structured_timeout_s": STRUCTURED_TIMEOUT_S,
                "oss_answer_max_tokens": 512,
                "other_answer_max_tokens": 100,
                "judge": "NOT_RUN",
                "cases_m_sha256": hashlib.sha256(
                    open(os.path.join(HERE, "cases_m.json"),
                         "rb").read()).hexdigest(),
                "scenarios_sha256": hashlib.sha256(
                    open(os.path.join(HERE, "scenarios_v1.json"),
                         "rb").read()).hexdigest(),
                "corpus_snapshot": ISO.corpus_snapshot_id(docs),
                "access": "EIC key auto-refresh (auth only)"}, f,
                indent=1, sort_keys=True)

    def _read_jsonl(path):
        rows = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        return rows

    def _done_sets():
        from collections import Counter
        a = Counter()
        a_arms = set()
        for r in _read_jsonl(os.path.join(outdir, "trace-answer.jsonl")):
            if r.get("error"):
                continue  # transport-failed samples stay re-runnable
            a[(r.get("scenario_id"), r.get("run"))] += 1
            a_arms.add((r.get("scenario_id"), r.get("run"),
                        r.get("arm")))
        b = {(r.get("scenario_id"), r.get("run")) for r in
             _read_jsonl(os.path.join(outdir, "trace-planner.jsonl"))
             if not r.get("error")}
        c = {(r.get("scenario_id"), r.get("arm"), r.get("run")) for r in
             _read_jsonl(os.path.join(outdir, "trace-reviewer.jsonl"))
             if not r.get("error")}
        d = set()
        for arm in sorted(ARMS):
            for r in _read_jsonl(os.path.join(
                    outdir, "trace-%s.jsonl" % arm.lower())):
                if (r.get("status") or "") == "error":
                    continue
                d.add((arm, r.get("scenario_id"), r.get("run")))
        return a, a_arms, b, c, d

    spend_path = os.path.join(outdir, "spend.json")
    if not dry and os.path.exists(spend_path):
        tracker.load(spend_path)

    def _save_spend():
        if not dry:
            tracker.save(spend_path)

    ans_done, ans_arms, plan_done, rev_done, e2e_done = _done_sets()
    control_path = os.path.join(outdir, "control.json")
    control_have = {}
    if os.path.exists(control_path):
        try:
            with open(control_path, encoding="utf-8") as f:
                for c in json.load(f):
                    control_have[c.get("scenario_id")] = c
        except Exception:
            control_have = {}
    control_complete = {s["scenario_id"] for s in scenarios} \
        <= set(control_have)

    answer_models = sorted(set(ANSWER_MODELS + ["qwen.qwen3-32b-v1:0"]))
    results = []
    failures = []

    def fail_stop(where, err):
        text = "%s: %s" % (type(err).__name__, str(err)[:300])
        if check_account_wide(text):
            raise StopRun("ACCOUNT-WIDE failure at %s: %s" % (where, text))
        return text

    # ---- CONTROL (Scenario 0 parity gate) ----
    # Gate definition (repo-frozen meaning of "parity"): the versioned
    # S (14) + M (26) harness suites pass with zero failures, AND the
    # deterministic baseline completes with zero ERRORS across the 40
    # benchmark scenarios. Per-scenario pass/fail against the
    # single-workflow harness expectation is RECORDED only: ambiguous
    # scenarios (e.g. S32: 2-doc "Summarize the lease.") carry gold
    # clarification semantics that the harness expect-schema cannot
    # express; scenario-gold conformance is measured by the benchmark
    # grader (grade_deterministic), never by this gate.
    from tests.benchmarks.harness import run_all as _run_all
    gate_parity_s = _run_all()["summary"]
    gate_parity_m = _run_all(
        cases_path=os.path.join(HERE, "cases_m.json"))["summary"]
    if gate_parity_s.get("failed", 1) != 0 or \
            gate_parity_m.get("failed", 1) != 0:
        raise StopRun("CONTROL PARITY FAIL: frozen S/M suites "
                      "green-required (S=%s M=%s): STOP, no paid calls"
                      % (gate_parity_s, gate_parity_m))
    print("PARITY SUITES PASS: S %d/%d, M %d/%d (0 failed)"
          % (gate_parity_s.get("passed", 0), gate_parity_s.get("total", 0),
             gate_parity_m.get("passed", 0),
             gate_parity_m.get("total", 0)))
    control = [control_have[k] for k in sorted(control_have)]
    if not control_complete:
        control = dict(control_have)
        try:
            from llm_provider import LMStudioProvider
            qwen_local = LMStudioProvider()
            control_live_ok = qwen_local.is_reachable()
        except Exception:
            qwen_local, control_live_ok = None, False
        for sc in scenarios:
            if sc["scenario_id"] in control:
                continue
            case = {"id": sc["scenario_id"], "category": "normal",
                    "query": sc["query"], "documents": sc["documents"],
                    "expect": {"workflow": sc["expected_workflow"]}}
            if sc.get("session"):
                case["session_documents"] = sc["session"].get("documents",
                                                              [])
                if sc["session"].get("bind"):
                    case["bind"] = sc["session"]["bind"]
                if sc["session"].get("prior_turns"):
                    case["context"] = sc["session"]["prior_turns"]
            try:
                res = run_case(case, docs)
                entry = {"scenario_id": sc["scenario_id"],
                         "status": res.get("status"),
                         "failures": res.get("failures", [])}
            except Exception as e:
                entry = {"scenario_id": sc["scenario_id"],
                         "status": "error", "error": str(e)[:200]}
            if control_live_ok and not dry:
                try:
                    live = run_case(case, docs, llm=qwen_local,
                                    model_key="qwen-local")
                    entry["control_live"] = {
                        "status": live.get("status"),
                        "failures": live.get("failures", [])}
                except Exception as e:
                    entry["control_live"] = {"status": "error",
                                             "error": str(e)[:200]}
            else:
                entry["control_live"] = {"status": "blocked",
                                         "reason": "LM Studio unreachable"}
            control[sc["scenario_id"]] = entry
        with open(os.path.join(outdir, "control.json"), "w",
                  encoding="utf-8") as f:
            json.dump([control[k] for k in sorted(control)], f, indent=1,
                      sort_keys=True)
        control = [control[k] for k in sorted(control)]
    det_err = [c for c in control if c.get("status") == "error"]
    if det_err:
        raise StopRun("CONTROL STABILITY FAIL: %d/%d deterministic cases "
                      "errored (e.g. %s): STOP, no paid calls"
                      % (len(det_err), len(control),
                         det_err[0].get("scenario_id")))
    det_pass = sum(1 for c in control if c.get("status") == "pass")
    print("CONTROL STABLE: %d/%d deterministic pass, %d non-error "
          "informational (%s), 0 errors"
          % (det_pass, len(control), len(control) - det_pass,
             ",".join(c.get("scenario_id", "?") for c in control
                      if c.get("status") not in ("pass", "error"))
             or "none"))

    # ---- MODES A/B/C per scenario per rep ----
    snapshots = {}
    for sc in scenarios:
        sid = sc["scenario_id"]
        try:
            snapshots[sid] = ISO.snapshot_evidence(sc, docs)
        except Exception as e:
            failures.append({"where": "snapshot " + sid,
                             "error": str(e)[:200]})
            continue
        snap = snapshots[sid]
        for rep in range(1, reps_for(sc) + 1):
            if ans_done.get((sid, rep), 0) >= 5 and (sid, rep) in \
                    plan_done and (sid, "REVIEWER-R235", rep) in \
                    rev_done and (sid, "REVIEWER-R32", rep) in rev_done:
                continue
            # Mode A: identical evidence for every answer model.
            # Resume-safe: recorded (model, rep) pairs are reloaded
            # from disk, never re-called (paid calls must not repeat).
            answers = {}
            fresh = []
            evidence_ids = [e.get("chunk_id") for e in snap["evidence"]]
            if ans_done.get((sid, rep), 0) >= 5 or \
                    any((sid, rep, "ANS-" + m.split(".")[0]) in ans_arms
                        for m in answer_models):
                for r in _read_jsonl(os.path.join(
                        outdir, "trace-answer.jsonl")):
                    if r.get("scenario_id") != sid or \
                            r.get("run") != rep or r.get("error"):
                        continue
                    for m in answer_models:
                        if "ANS-" + m.split(".")[0] == r.get("arm") \
                                and m not in answers:
                            answers[m] = {"answer": r.get("final", ""),
                                          "evidence_ids": evidence_ids,
                                          "generation_ms": None,
                                          "ttft_ms": None, "diag": None,
                                          "_recorded": True}
            for mid in answer_models:
                if mid in answers:
                    continue
                fresh.append(mid)
                CTX.arm, CTX.mode, CTX.role, CTX.scenario_id = \
                    ("ANS-%s" % mid.split(".")[0], "A", "answer", sid)
                try:
                    if dry:
                        out = ISO.run_answer_isolation(EchoLLM(), snap)
                    else:
                        out = ISO.run_answer_isolation(
                            MantleChatProvider(mid, transport, tracker),
                            snap)
                    if out["evidence_ids"] != evidence_ids:
                        raise StopRun("EVIDENCE DIVERGED in mode A for "
                                      + sid)
                    answers[mid] = out
                except StopRun:
                    raise
                except Exception as e:
                    failures.append({"where": "A %s %s r%d"
                                     % (sid, mid, rep),
                                     "error": fail_stop("A", e)})
                    answers[mid] = {"answer": "", "error": str(e)[:200]}
                finally:
                    CTX.reset()
            # Mode B: planner isolation (Qwen32B, once per scenario/rep).
            berr = None
            if (sid, rep) in plan_done:
                pout = None  # recorded: no re-call, no duplicate trace
            else:
                try:
                    CTX.arm, CTX.mode, CTX.role, CTX.scenario_id = \
                        ("PLAN", "B", "planner", sid)
                    from query_planner import (DeterministicPlanner,
                                               LLMQueryPlanner, PlanCache)
                    if dry:
                        pb = {"planner": DeterministicPlanner(),
                              "cache": PlanCache(), "provider": "d",
                              "model": "n",
                              "timeout_s": STRUCTURED_TIMEOUT_S,
                              "_meta_base": {}}
                    else:
                        from query_planner import LLMQueryPlanner as _P
                        pb = {"planner": _P(
                            mantle_structured_fn(
                                PLANNER_MANTLE, transport,
                                tracker, role="planner"),
                            timeout_s=STRUCTURED_TIMEOUT_S,
                            provider="mantle", model=PLANNER_MANTLE),
                            "cache": PlanCache(), "provider": "mantle",
                            "model": PLANNER_MANTLE,
                            "timeout_s": STRUCTURED_TIMEOUT_S}
                    store_files = sorted({e.get("file_name")
                                          for e in snap["evidence"]
                                          if e.get("file_name")})
                    pout = ISO.run_planner_isolation(pb, snap,
                                                     store_files)
                except StopRun:
                    raise
                except Exception as e:
                    pout = {"schema_valid": False,
                            "failure": str(e)[:100],
                            "plan": None, "latency_ms": None,
                            "call_failure": None}
                    berr = "%s: %s" % (type(e).__name__, str(e)[:200])
                    failures.append({"where": "B %s r%d" % (sid, rep),
                                     "error": fail_stop("B", e)})
                finally:
                    CTX.reset()
            # Mode C: frozen answer (ARM1's), both reviewer models.
            frozen = (answers.get(ARMS["ARM1"]["answer"]) or {}).get(
                "answer", "")
            for rmid, tag in REVIEWER_MODELS:
                if (sid, "REVIEWER-" + tag, rep) in rev_done:
                    continue
                rerr = None
                try:
                    CTX.arm, CTX.mode, CTX.role, CTX.scenario_id = \
                        ("REV-" + tag, "C", "reviewer", sid)
                    from answer_reviewer import (DeterministicReviewer,
                                                 LLMAnswerReviewer)
                    if dry:
                        rb = {"reviewer": DeterministicReviewer(),
                              "provider": "d", "model": "n",
                              "timeout_s": STRUCTURED_TIMEOUT_S,
                              "_meta_base": {}}
                    else:
                        rb = {"reviewer": LLMAnswerReviewer(
                            mantle_structured_fn(rmid, transport,
                                                 tracker, role="reviewer"),
                            timeout_s=STRUCTURED_TIMEOUT_S,
                            provider="mantle", model=rmid),
                            "provider": "mantle", "model": rmid,
                            "timeout_s": STRUCTURED_TIMEOUT_S}
                    rout = ISO.run_reviewer_isolation(
                        rb, snap, frozen, "normal")
                except StopRun:
                    raise
                except Exception as e:
                    rout = {"schema_valid": False,
                            "failure": str(e)[:100],
                            "verdict": None, "latency_ms": None,
                            "call_failure": None}
                    rerr = "%s: %s" % (type(e).__name__, str(e)[:200])
                    failures.append({"where": "C %s %s r%d"
                                     % (sid, tag, rep),
                                     "error": fail_stop("C", e)})
                finally:
                    CTX.reset()
                ISO.write_trace(
                    os.path.join(outdir, "trace-reviewer.jsonl"),
                    {"scenario_id": sid, "arm": "REVIEWER-" + tag,
                     "run": rep, "question_id": sid, "router": "n/a",
                     "planner": None,
                     "retrieval": {"evidence_ids": evidence_ids},
                     "evidence_ids": evidence_ids,
                     "answer": frozen[:2000],
                     "citation_guard": None, "reviewer": rout,
                     "repair": None, "final": frozen[:2000],
                     "timings": {"reviewer_latency_ms":
                                 rout.get("latency_ms")},
                     "tokens": None, "cost": None,
                     "deterministic_grade": ISO.grade_deterministic(
                         sc, frozen,
                         [{"file_name": e.get("file_name"),
                           "content": e.get("content", "")}
                          for e in snap["evidence"]],
                         {"workflow": "normal"}, snap["evidence"],
                         reviewer_out=rout),
                     "qualitative_grade": "NOT_RUN",
                     "error": rerr,
                     "failure_category": None if rout.get("schema_valid")
                     else "SCHEMA_FAILURE", "severity": sc["severity"]})
            # Mode A traces + grades (fresh models only; recorded
            # rows already on disk are never duplicated).
            for mid, out in answers.items():
                if mid not in fresh:
                    continue
                if "error" in out and "answer" not in out:
                    continue
                srcs = [{"file_name": e.get("file_name"),
                         "content": e.get("content", "")}
                        for e in snap["evidence"]]
                grade = ISO.grade_deterministic(
                    sc, out.get("answer", ""), srcs, {"workflow": "normal"},
                    snap["evidence"])
                ISO.write_trace(
                    os.path.join(outdir, "trace-answer.jsonl"),
                    {"scenario_id": sid,
                     "arm": "ANS-" + mid.split(".")[0],
                     "run": rep, "question_id": sid, "router": "n/a",
                     "planner": None,
                     "retrieval": {"evidence_ids": evidence_ids},
                     "evidence_ids": evidence_ids,
                     "answer": out.get("answer", ""),
                     "diag": out.get("diag"),
                     "citation_guard": None, "reviewer": None,
                     "repair": None, "final": out.get("answer", ""),
                     "timings": {"generation_ms": out.get("generation_ms"),
                                 "ttft_ms": out.get("ttft_ms")},
                     "tokens": None, "cost": None,
                     "deterministic_grade": grade,
                     "qualitative_grade": "NOT_RUN",
                     "error": out.get("error"),
                     "failure_category": None if grade["grounded"]
                     else ";".join(grade["failure_categories"][:2]),
                     "severity": sc["severity"]})
            # Mode B trace (skipped when already recorded).
            if pout is not None:
                ISO.write_trace(
                    os.path.join(outdir, "trace-planner.jsonl"),
                    {"scenario_id": sid, "arm": "PLANNER", "run": rep,
                     "question_id": sid, "router": "n/a", "planner": pout,
                 "retrieval": {"evidence_ids": evidence_ids},
                 "evidence_ids": evidence_ids, "answer": "",
                 "citation_guard": None, "reviewer": None, "repair": None,
                 "final": "",
                 "timings": {"planner_latency_ms": pout.get("latency_ms")},
                 "tokens": None, "cost": None,
                 "deterministic_grade": {"planner_schema_valid":
                                         pout.get("schema_valid")},
                 "qualitative_grade": "NOT_RUN",
                 "error": berr,
                 "failure_category": None if pout.get("schema_valid")
                 else "SCHEMA_FAILURE", "severity": sc["severity"]})
            _save_spend()

    # ---- MODE D: full E2E per arm per rep ----
    for arm in sorted(ARMS):
        spec = ARMS[arm]
        for sc in scenarios:
            sid = sc["scenario_id"]
            for rep in range(1, reps_for(sc) + 1):
                if (arm, sid, rep) in e2e_done:
                    continue
                try:
                    CTX.arm, CTX.mode, CTX.role, CTX.scenario_id = \
                        (arm, "D", "e2e", sid)
                    from query_planner import (LLMQueryPlanner,
                                               PlanCache, query_planned)
                    from answer_reviewer import (LLMAnswerReviewer,
                                                 query_reviewed)
                    from tests.benchmarks import metrics as metrics_mod
                    store, bind_sid, context = ISO.build_store(sc, docs)
                    if dry:
                        from query_planner import DeterministicPlanner
                        from answer_reviewer import DeterministicReviewer
                        pbundle = {"planner": DeterministicPlanner(),
                                   "cache": PlanCache(), "provider": "d",
                                   "model": "n",
                                   "timeout_s": STRUCTURED_TIMEOUT_S,
                                   "_meta_base": {}}
                        rbundle = {"reviewer": DeterministicReviewer(),
                                   "provider": "d", "model": "n",
                                   "timeout_s": STRUCTURED_TIMEOUT_S,
                                   "_meta_base": {}}
                        llm = EchoLLM()
                    else:
                        pm = MANTLE_IDS[spec["planner"]] \
                            if spec["planner"] in MANTLE_IDS \
                            else spec["planner"]
                        rm = MANTLE_IDS[spec["reviewer"]] \
                            if spec["reviewer"] in MANTLE_IDS \
                            else spec["reviewer"]
                        pbundle = {
                            "planner": LLMQueryPlanner(
                                mantle_structured_fn(
                                    pm, transport, tracker, role="planner"),
                                timeout_s=STRUCTURED_TIMEOUT_S,
                                provider="mantle", model=pm),
                            "cache": PlanCache(), "provider": "mantle",
                            "model": pm,
                            "timeout_s": STRUCTURED_TIMEOUT_S}
                        rbundle = {
                            "reviewer": LLMAnswerReviewer(
                                mantle_structured_fn(
                                    rm, transport, tracker, role="reviewer"),
                                timeout_s=STRUCTURED_TIMEOUT_S,
                                provider="mantle", model=rm),
                            "provider": "mantle", "model": rm,
                            "timeout_s": STRUCTURED_TIMEOUT_S}
                        llm = MantleChatProvider(spec["answer"], transport,
                                                 tracker)
                    import time as _t
                    _t0 = _t.monotonic()
                    answer, sources, info, reasoning, review = \
                        query_reviewed(
                            sc["query"], vector_store=store,
                            llm_provider=llm,
                            conversation_context=context,
                            session_id=bind_sid,
                            planner_bundle=pbundle,
                            reviewer_bundle=rbundle)
                    wall_ms = max(0, int((_t.monotonic() - _t0) * 1000))
                    repair_n = 1 if (review or {}).get(
                        "repair_attempted") else 0
                    metrics = metrics_mod.compute(
                        answer, sources, info, {"wall_ms": wall_ms},
                        evidence_text=" ".join(
                            (s or {}).get("content", "") for s in sources),
                        model_key="echo", reasoning=reasoning,
                        review=review)
                    res = {"id": sid,
                           "status": "pass" if not ISO.grade_deterministic(
                               sc, answer or "", sources or [], info or {},
                               (snapshots.get(sid) or {}).get("evidence",
                                                              []),
                               repair_count=repair_n)["failures"]
                           else "fail",
                           "answer": answer, "sources": sources,
                           "info": info, "metrics": metrics,
                           "reasoning": reasoning, "review": review}
                except StopRun:
                    raise
                except Exception as e:
                    res = {"id": sid, "status": "error",
                           "error": str(e)[:200]}
                    repair_n = 0
                    failures.append({"where": "D %s %s r%d"
                                     % (arm, sid, rep),
                                     "error": fail_stop("D", e)})
                finally:
                    CTX.reset()
                grade = None
                try:
                    grade = ISO.grade_deterministic(
                        sc, res.get("answer", "") or "",
                        res.get("sources", []) or [],
                        res.get("info", {}) or [],
                        (snapshots.get(sid) or {}).get("evidence", []),
                        repair_count=repair_n)
                except Exception as e:
                    grade = {"grounded": False,
                             "failures": [str(e)[:200]],
                             "failure_categories": ["OTHER"]}
                ISO.write_trace(
                    os.path.join(outdir, "trace-%s.jsonl" % arm.lower()),
                    {"scenario_id": sid, "arm": arm, "run": rep,
                     "question_id": sid, "status": res.get("status"),
                     "router": (res.get("info") or {}).get("workflow"),                     "planner": res.get("reasoning"),
                     "retrieval": {"source_files":
                                   (res.get("metrics") or {}).get(
                                       "source_files", [])},
                     "evidence_ids": [e.get("chunk_id") for e in
                                      (snapshots.get(sid) or {}).get(
                                          "evidence", [])],
                     "answer": res.get("answer", "") or "",
                     "citation_guard": {
                         "flagged": (res.get("metrics") or {}).get(
                             "guard_flagged"),
                         "total": (res.get("metrics") or {}).get(
                             "guard_total")},
                     "reviewer": res.get("review"),
                     "repair": {"attempted": repair_n},
                     "final": res.get("answer", "") or "",
                     "timings": res.get("metrics", {}),
                     "tokens": {"in": (res.get("metrics") or {}).get(
                         "tokens_in"),
                         "out": (res.get("metrics") or {}).get(
                             "tokens_out"),
                         "method": (res.get("metrics") or {}).get(
                             "token_method")},
                     "cost": (res.get("metrics") or {}).get("cost_usd"),
                     "deterministic_grade": grade,
                     "qualitative_grade": "NOT_RUN",
                     "failure_category": None if grade.get("grounded")
                     else ";".join(
                         grade.get("failure_categories", [])[:2]),
                     "severity": sc["severity"]})
                _save_spend()

    # ---- calibration shell (18% of E2E, never scored) ----
    pool = []
    for arm in sorted(ARMS):
        path = os.path.join(outdir, "trace-%s.jsonl" % arm.lower())
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    pool.append(json.loads(line))
                except Exception:
                    pass
    packet = ISO.calibration_packet(pool, seed=74154, frac=0.18)
    with open(os.path.join(outdir, "calibration_packet.json"), "w",
              encoding="utf-8") as f:
        json.dump(packet, f, indent=1, sort_keys=True)

    # ---- secret scan over all artifacts ----
    leaked = []
    for name in sorted(os.listdir(outdir)):
        if not name.endswith((".json", ".jsonl")):
            continue
        blob = open(os.path.join(outdir, name), encoding="utf-8").read()
        import re
        hits = re.findall(r"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16})", blob)
        if hits:
            leaked.append(name)
    if leaked:
        raise StopRun("CREDENTIALS IN ARTIFACTS: %s" % leaked)

    summary = {"run": RUN_ID, "benchmark_revision": BENCHMARK_REVISION,
               "scenarios": len(scenarios),
               "executions": total_exec,
               "arms": sorted(ARMS), "failures": failures,
               "spend_estimate_usd": round(tracker.spent_estimate, 4),
               "paid_calls": tracker.calls,
               "retries": tracker.retries,
               "by_model": tracker.by_model,
               "judge": "NOT_RUN",
               "judge_validation_status": "PENDING"}
    _save_spend()
    # Completeness from trace counts (subset runs report pending).
    exp_ids = {(s["scenario_id"], rep) for s in scenarios
               for rep in range(1, reps_for(s) + 1)}
    have_p = {(r.get("scenario_id"), r.get("run")) for r in
              _read_jsonl(os.path.join(outdir, "trace-planner.jsonl"))}
    have_c = {(r.get("scenario_id"), r.get("run")) for r in
              _read_jsonl(os.path.join(outdir, "trace-reviewer.jsonl"))}
    a_rep_ok = all(
        sum(1 for r in _read_jsonl(os.path.join(
            outdir, "trace-answer.jsonl"))
            if r.get("scenario_id") == s["scenario_id"]
            and r.get("run") == rep) >= 5
        for s in scenarios for rep in range(1, reps_for(s) + 1))
    e2e_missing = {}
    for arm in sorted(ARMS):
        have_e = {(r.get("scenario_id"), r.get("run")) for r in
                  _read_jsonl(os.path.join(
                      outdir, "trace-%s.jsonl" % arm.lower()))}
        e2e_missing[arm] = sorted(exp_ids - have_e)
    summary["complete"] = bool(
        a_rep_ok and exp_ids <= have_p and exp_ids <=
        {x for x in have_c} and not any(e2e_missing.values()))
    summary["pending"] = {
        "answer_reps_missing": sorted(exp_ids - {
            (r.get("scenario_id"), r.get("run")) for r in
            _read_jsonl(os.path.join(outdir, "trace-answer.jsonl"))
            if sum(1 for q in _read_jsonl(os.path.join(
                outdir, "trace-answer.jsonl"))
                if q.get("scenario_id") == r.get("scenario_id")
                and q.get("run") == r.get("run")) >= 5}),
        "planner": sorted(exp_ids - have_p),
        "reviewer": sorted(exp_ids - {x for x in have_c}),
        "e2e": e2e_missing,
    }
    with open(os.path.join(outdir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    with open(os.path.join(outdir, "teardown.txt"), "w",
              encoding="utf-8") as f:
        f.write("production untouched: DictStore in-memory only; "
                "no vectors/S3/RDS writes by construction.\n")
    if lock_path:
        release_live_lock(lock_path)
    return summary


if __name__ == "__main__":
    try:
        summary = main()
    except StopRun as e:
        print("STOP: %s" % e)
        raise SystemExit(3)
    except BudgetExceeded as e:
        print("STOP (budget): %s" % e)
        raise SystemExit(4)
    print(json.dumps(summary, indent=1, sort_keys=True))
