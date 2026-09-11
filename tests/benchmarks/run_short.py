"""Short exploratory benchmark runner (Mumbai Mantle, 5 arms, 12 cases).

Usage:
  DRY_RUN=1 python tests/benchmarks/run_short.py   # $0, Echo doubles
  python tests/benchmarks/run_short.py             # LIVE paid run ($8 cap)

New file only; nothing existing is modified. Stops on the first
account-wide failure, budget breach, hard-gate violation, or any
model substitution. Artifacts go to tests/benchmarks/results/<run-id>/
and stay uncommitted.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from tests.benchmarks import isolations as ISO
from tests.benchmarks.harness import EchoLLM, load_cases, run_case
from tests.benchmarks.mantle_live import (
    ARMS, BUDGET_CAP_USD, CTX, MANTLE_IDS, PRICES, BudgetExceeded,
    MantleChatProvider, SpendTracker, mantle_structured_fn,
)
SHORT_IDS = ["S01-normal-lockin", "S04-sla", "S06-loan",
             "S11-exact-clause", "S16-compound", "S21-compare-leases",
             "S25-summary-handbook", "S28-extract-notice", "S18-neardup",
             "S34-stale", "S36-followup", "S40-session"]
assert len(SHORT_IDS) == 12 and len(set(SHORT_IDS)) == 12

ANSWER_MODELS = ["qwen.qwen3-235b-a22b-2507-v1:0",
                 "openai.gpt-oss-120b-1:0",
                 "mistral.mistral-large-3-675b-instruct",
                 "mistral.devstral-2-123b"]
PLANNER_MODEL = "qwen.qwen3-32b-v1:0"
REVIEWER_MODELS = ["qwen.qwen3-235b-a22b-2507-v1:0",
                   "qwen.qwen3-32b-v1:0"]

ACCOUNT_WIDE_MARKERS = ("Operation not allowed", "NoCredentials",
                        "credential", "throttl", "TooManyRequests",
                        "InternalServer", "ServiceUnavailable",
                        "AccessDenied", "not authorized")


def utc_stamp():
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


class StopRun(Exception):
    pass


def check_account_wide(err_text):
    low = (err_text or "").lower()
    return any(m.lower() in low for m in ACCOUNT_WIDE_MARKERS)


def scan_secrets(obj):
    import re
    blob = json.dumps(obj, default=str)
    hits = re.findall(r"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|"
                      r"xox[bap]-[0-9A-Za-z-]+)", blob)
    return hits


def main():
    dry = os.environ.get("DRY_RUN") == "1"
    run_id = time.strftime("%Y%m%d", time.gmtime()) + "-short01"
    outdir = os.path.join(HERE, "results", run_id)
    os.makedirs(outdir, exist_ok=True)

    from tests.benchmarks import mantle_live as ML
    transport = None
    tracker = SpendTracker()
    if not dry:
        key = os.environ.get("SSH_KEY", os.path.expandvars(
            r"${TEMP}\mm_run"))
        key = os.path.expandvars(key)
        from tests.benchmarks.mantle_live import SSHTransport
        # EIC pushes expire after 60s; the transport re-pushes automatically
        # (auth plumbing only, no infra/IAM/quota change). Recorded here
        # so the exact access path is auditable from the config snapshot.
        refresh = ["aws", "ec2-instance-connect", "send-ssh-public-key",
                   "--profile", "rag4i-dev", "--region", "us-east-1",
                   "--instance-id", "i-07dea54f4144b8fd0",
                   "--availability-zone", "us-east-1c",
                   "--instance-os-user", "ec2-user",
                   "--ssh-public-key", "file://" + key + ".pub"]
        transport = SSHTransport(key, refresh_cmd=refresh)
        cfg_path = os.path.join(outdir, "config_snapshot.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"arms": ARMS, "mantle_ids": MANTLE_IDS,
                       "prices": PRICES, "region": ML.REGION,
                       "endpoint": "bedrock-mantle.ap-south-1.api.aws/v1",
                       "scenarios": SHORT_IDS, "repetitions": 1,
                       "seed": 74154, "dry_run": False,
                       "access": "EIC key auto-refresh (auth only)"}, f,
                      indent=1, sort_keys=True)

    suite = ISO.load_scenarios()
    docs = load_cases(os.path.join(HERE, "cases_m.json"))["documents"]
    scenarios = [s for s in suite["scenarios"]
                 if s["scenario_id"] in SHORT_IDS]
    assert len(scenarios) == 12, len(scenarios)
    only = [s.strip() for s in
            os.environ.get("SCENARIOS", "").split(",") if s.strip()]
    if only:
        wanted = set(only)
        assert wanted <= set(SHORT_IDS), sorted(wanted - set(SHORT_IDS))
        scenarios = [s for s in scenarios if s["scenario_id"] in wanted]
    corpus_id = ISO.corpus_snapshot_id(docs)

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
        a = {}
        for r in _read_jsonl(os.path.join(outdir, "trace-answer.jsonl")):
            a.setdefault(r.get("scenario_id"), 0)
            a[r.get("scenario_id")] += 1
        b = {r.get("scenario_id") for r in
             _read_jsonl(os.path.join(outdir, "trace-planner.jsonl"))}
        c = set()
        for r in _read_jsonl(os.path.join(outdir, "trace-reviewer.jsonl")):
            c.add((r.get("scenario_id"), r.get("arm")))
        d = set()
        for arm in ["ARM1", "ARM2", "ARM3", "ARM4", "ARM5"]:
            for r in _read_jsonl(os.path.join(
                    outdir, "trace-%s.jsonl" % arm.lower())):
                d.add((arm, r.get("scenario_id")))
        return a, b, c, d

    spend_path = os.path.join(outdir, "spend.json")
    if not dry and os.path.exists(spend_path):
        tracker.load(spend_path)

    def _save_spend():
        if not dry:
            tracker.save(spend_path)

    ans_done, plan_done, rev_done, e2e_done = _done_sets()
    control_path = os.path.join(outdir, "control.json")
    control_complete = False
    if os.path.exists(control_path):
        try:
            with open(control_path, encoding="utf-8") as f:
                have = {c.get("scenario_id") for c in json.load(f)}
            control_complete = set(SHORT_IDS) <= have
        except Exception:
            control_complete = False

    answer_models = sorted(set(ANSWER_MODELS + ["qwen.qwen3-32b-v1:0"]))
    # Mode A needs all 5 answer models incl. Qwen32B (Arm5 shares OSS).
    results = []
    failures = []

    def fail_stop(where, err):
        text = "%s: %s" % (type(err).__name__, str(err)[:300])
        if check_account_wide(text):
            raise StopRun("ACCOUNT-WIDE failure at %s: %s" % (where, text))
        return text

    # ---- CONTROL: deterministic baseline (free) + live local Qwen ----
    # Skipped when a previous chunk already recorded all 12 scenarios.
    control = []
    if control_complete:
        with open(control_path, encoding="utf-8") as f:
            control = json.load(f)
    else:
        try:
            from llm_provider import LMStudioProvider
            qwen_local = LMStudioProvider()
            control_live_ok = qwen_local.is_reachable()
        except Exception:
            qwen_local, control_live_ok = None, False
        for sc in scenarios:
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
            control.append(entry)
        with open(os.path.join(outdir, "control.json"), "w",
                  encoding="utf-8") as f:
            json.dump(control, f, indent=1, sort_keys=True)

    # ---- MODES A/B/C per scenario ----
    # Resume: fully-recorded scenarios are skipped (traces are append-only).
    snapshots = {}
    for sc in scenarios:
        sid = sc["scenario_id"]
        if ans_done.get(sid, 0) >= 5 and sid in plan_done and \
                (sid, "REVIEWER-R235") in rev_done and \
                (sid, "REVIEWER-R32") in rev_done:
            try:
                snapshots[sid] = ISO.snapshot_evidence(sc, docs)
            except Exception:
                pass
            continue
        try:
            snapshots[sid] = ISO.snapshot_evidence(sc, docs)
        except Exception as e:
            failures.append({"where": "snapshot " + sid,
                             "error": str(e)[:200]})
            continue
        snap = snapshots[sid]
        # Mode A: identical evidence for every answer model.
        answers = {}
        evidence_ids = [e.get("chunk_id") for e in snap["evidence"]]
        for mid in answer_models:
            CTX.arm, CTX.mode, CTX.role, CTX.scenario_id = \
                ("ANS-%s" % mid.split(".")[0], "A", "answer", sid)
            try:
                if dry:
                    out = ISO.run_answer_isolation(EchoLLM(), snap)
                else:
                    out = ISO.run_answer_isolation(
                        MantleChatProvider(mid, transport, tracker), snap)
                if out["evidence_ids"] != evidence_ids:
                    raise StopRun("EVIDENCE DIVERGED in mode A for " + sid)
                answers[mid] = out
            except StopRun:
                raise
            except Exception as e:
                failures.append({"where": "A %s %s" % (sid, mid),
                                 "error": fail_stop("A", e)})
                answers[mid] = {"answer": "", "error": str(e)[:200]}
            finally:
                CTX.reset()
        # Mode B: planner isolation (Qwen32B, once per scenario).
        try:
            CTX.arm, CTX.mode, CTX.role, CTX.scenario_id = \
                ("PLAN", "B", "planner", sid)
            from query_planner import DeterministicPlanner, PlanCache
            if dry:
                pb = {"planner": DeterministicPlanner(),
                      "cache": PlanCache(), "provider": "d", "model": "n",
                      "timeout_s": 5, "_meta_base": {}}
            else:
                from query_planner import LLMQueryPlanner
                pb = {"planner": LLMQueryPlanner(
                    mantle_structured_fn("qwen.qwen3-32b", transport,
                                         tracker, role="planner"),
                    timeout_s=5, provider="mantle",
                    model="qwen.qwen3-32b"),
                    "cache": PlanCache(), "provider": "mantle",
                    "model": "qwen.qwen3-32b", "timeout_s": 5}
            store_files = sorted({e.get("file_name")
                                  for e in snap["evidence"]
                                  if e.get("file_name")})
            pout = ISO.run_planner_isolation(pb, snap, store_files)
        except StopRun:
            raise
        except Exception as e:
            pout = {"schema_valid": False, "failure": str(e)[:100],
                    "plan": None, "latency_ms": None}
            failures.append({"where": "B %s" % sid,
                             "error": fail_stop("B", e)})
        finally:
            CTX.reset()
        # Mode C: frozen answer (ARM1's), both reviewer models.
        frozen = (answers.get(ARMS["ARM1"]["answer"]) or {}).get(
            "answer", "")
        for rmid, tag in (("qwen.qwen3-235b-a22b-2507-v1:0", "R235"),
                          ("qwen.qwen3-32b", "R32")):
            try:
                CTX.arm, CTX.mode, CTX.role, CTX.scenario_id = \
                    ("REV-" + tag, "C", "reviewer", sid)
                from answer_reviewer import DeterministicReviewer
                if dry:
                    rb = {"reviewer": DeterministicReviewer(),
                          "provider": "d", "model": "n", "timeout_s": 5,
                          "_meta_base": {}}
                else:
                    from answer_reviewer import LLMAnswerReviewer
                    rb = {"reviewer": LLMAnswerReviewer(
                        mantle_structured_fn(rmid, transport, tracker,
                                             role="reviewer"),
                        timeout_s=5, provider="mantle", model=rmid),
                        "provider": "mantle", "model": rmid,
                        "timeout_s": 5}
                rout = ISO.run_reviewer_isolation(
                    rb, snap, frozen, "normal")
            except StopRun:
                raise
            except Exception as e:
                rout = {"schema_valid": False, "failure": str(e)[:100],
                        "verdict": None, "latency_ms": None}
                failures.append({"where": "C %s %s" % (sid, tag),
                                 "error": fail_stop("C", e)})
            finally:
                CTX.reset()
            ISO.write_trace(
                os.path.join(outdir, "trace-reviewer.jsonl"),
                {"scenario_id": sid, "arm": "REVIEWER-" + tag, "run": 1,
                 "question_id": sid, "router": "n/a", "planner": None,
                 "retrieval": {"evidence_ids": evidence_ids},
                 "evidence_ids": evidence_ids, "answer": frozen[:2000],
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
                 "failure_category": None if rout.get("schema_valid")
                 else "SCHEMA_FAILURE", "severity": sc["severity"]})
        # Mode A traces + grades per answer model.
        for mid, out in answers.items():
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
                {"scenario_id": sid, "arm": "ANS-" + mid.split(".")[0],
                 "run": 1, "question_id": sid, "router": "n/a",
                 "planner": None,
                 "retrieval": {"evidence_ids": evidence_ids},
                 "evidence_ids": evidence_ids,
                 "answer": out.get("answer", ""),
                 "citation_guard": None, "reviewer": None, "repair": None,
                 "final": out.get("answer", ""),
                 "timings": {"generation_ms": out.get("generation_ms"),
                             "ttft_ms": out.get("ttft_ms")},
                 "tokens": None, "cost": None,
                 "deterministic_grade": grade,
                 "qualitative_grade": "NOT_RUN",
                 "failure_category": None if grade["grounded"]
                 else ";".join(grade["failure_categories"][:2]),
                 "severity": sc["severity"]})
        # Mode B trace.
        ISO.write_trace(
            os.path.join(outdir, "trace-planner.jsonl"),
            {"scenario_id": sid, "arm": "PLANNER", "run": 1,
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
                  "failure_category": None if pout.get("schema_valid")
                  else "SCHEMA_FAILURE", "severity": sc["severity"]})
        _save_spend()

    # ---- MODE D: full E2E per arm (direct runners: full grade data) ----
    # Resume: recorded (arm, scenario) pairs are skipped.
    for arm in ["ARM1", "ARM2", "ARM3", "ARM4", "ARM5"]:
        spec = ARMS[arm]
        for sc in scenarios:
            sid = sc["scenario_id"]
            if (arm, sid) in e2e_done:
                continue
            try:
                CTX.arm, CTX.mode, CTX.role, CTX.scenario_id = \
                    (arm, "D", "e2e", sid)
                from query_planner import (LLMQueryPlanner, PlanCache,
                                           query_planned)
                from answer_reviewer import (LLMAnswerReviewer,
                                             query_reviewed)
                from tests.benchmarks import metrics as metrics_mod
                store, bind_sid, context = ISO.build_store(sc, docs)
                if dry:
                    from query_planner import DeterministicPlanner
                    from answer_reviewer import DeterministicReviewer
                    pbundle = {"planner": DeterministicPlanner(),
                               "cache": PlanCache(), "provider": "d",
                               "model": "n", "timeout_s": 5,
                               "_meta_base": {}}
                    rbundle = {"reviewer": DeterministicReviewer(),
                               "provider": "d", "model": "n",
                               "timeout_s": 5, "_meta_base": {}}
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
                            mantle_structured_fn(pm, transport, tracker,
                                                 role="planner"),
                            timeout_s=5, provider="mantle", model=pm),
                        "cache": PlanCache(), "provider": "mantle",
                        "model": pm, "timeout_s": 5}
                    rbundle = {
                        "reviewer": LLMAnswerReviewer(
                            mantle_structured_fn(rm, transport, tracker,
                                                 role="reviewer"),
                            timeout_s=5, provider="mantle", model=rm),
                        "provider": "mantle", "model": rm, "timeout_s": 5}
                    llm = MantleChatProvider(spec["answer"], transport,
                                             tracker)
                import time as _t
                _t0 = _t.monotonic()
                answer, sources, info, reasoning, review = query_reviewed(
                    sc["query"], vector_store=store, llm_provider=llm,
                    conversation_context=context, session_id=bind_sid,
                    planner_bundle=pbundle, reviewer_bundle=rbundle)
                wall_ms = max(0, int((_t.monotonic() - _t0) * 1000))
                repair_n = 1 if (review or {}).get(
                    "repair_attempted") else 0
                metrics = metrics_mod.compute(
                    answer, sources, info, {"wall_ms": wall_ms},
                    evidence_text=" ".join(
                        (s or {}).get("content", "") for s in sources),
                    model_key="echo", reasoning=reasoning, review=review)
                res = {"id": sid,
                       "status": "pass" if not ISO.grade_deterministic(
                           sc, answer or "", sources or [], info or {},
                           (snapshots.get(sid) or {}).get("evidence", []),
                           repair_count=repair_n)["failures"] else "fail",
                       "answer": answer, "sources": sources, "info": info,
                       "metrics": metrics, "reasoning": reasoning,
                       "review": review}
            except StopRun:
                raise
            except Exception as e:
                res = {"id": sid, "status": "error",
                       "error": str(e)[:200]}
                repair_n = 0
                failures.append({"where": "D %s %s" % (arm, sid),
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
                grade = {"grounded": False, "failures": [str(e)[:200]],
                         "failure_categories": ["OTHER"]}
            ISO.write_trace(
                os.path.join(outdir, "trace-%s.jsonl" % arm.lower()),
                {"scenario_id": sid, "arm": arm, "run": 1,
                 "question_id": sid,
                 "router": (res.get("info") or {}).get("workflow"),
                 "planner": res.get("reasoning"),
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
                     "out": (res.get("metrics") or {}).get("tokens_out"),
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
    for arm in ["ARM1", "ARM2", "ARM3", "ARM4", "ARM5"]:
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

    summary = {"run": "short01", "scenarios": len(scenarios),
               "arms": ["ARM1", "ARM2", "ARM3", "ARM4", "ARM5"],
               "failures": failures,
               "spend_estimate_usd": round(tracker.spent_estimate, 4),
               "paid_calls": tracker.calls,
               "by_model": tracker.by_model,
               "judge": "NOT_RUN",
               "judge_validation_status": "PENDING"}
    _save_spend()
    # Completeness across all expected units (12 scenarios x modes).
    exp_a = {(s, m) for s in SHORT_IDS for m in
             ["qwen.qwen3-235b-a22b-2507-v1:0",
              "openai.gpt-oss-120b-1:0",
              "mistral.mistral-large-3-675b-instruct",
              "mistral.devstral-2-123b", "qwen.qwen3-32b-v1:0"]}
    have_a = set()
    for r in _read_jsonl(os.path.join(outdir, "trace-answer.jsonl")):
        have_a.add((r.get("scenario_id"), None))
    # Count-based check (arm labels collide for qwen models by design).
    from collections import Counter
    a_counts = Counter(
        r.get("scenario_id") for r in
        _read_jsonl(os.path.join(outdir, "trace-answer.jsonl")))
    a_ok = all(a_counts.get(s, 0) >= 5 for s in SHORT_IDS)
    p_ok = {r.get("scenario_id") for r in
            _read_jsonl(os.path.join(outdir, "trace-planner.jsonl"))} \
        >= set(SHORT_IDS)
    c_rows = _read_jsonl(os.path.join(outdir, "trace-reviewer.jsonl"))
    c_ok = all(sum(1 for r in c_rows if r.get("scenario_id") == s) >= 2
               for s in SHORT_IDS)
    e_ok = True
    for arm in ["ARM1", "ARM2", "ARM3", "ARM4", "ARM5"]:
        have = {r.get("scenario_id") for r in _read_jsonl(
            os.path.join(outdir, "trace-%s.jsonl" % arm.lower()))}
        if not set(SHORT_IDS) <= have:
            e_ok = False
    summary["complete"] = bool(a_ok and p_ok and c_ok and e_ok)
    summary["pending"] = {
        "answer": sorted(set(SHORT_IDS) - {
            s for s in SHORT_IDS if a_counts.get(s, 0) >= 5}),
        "planner": sorted(set(SHORT_IDS) - {
            r.get("scenario_id") for r in _read_jsonl(
                os.path.join(outdir, "trace-planner.jsonl"))}),
        "reviewer": sorted(set(SHORT_IDS) - {
            s for s in SHORT_IDS
            if sum(1 for r in c_rows if r.get("scenario_id") == s) >= 2}),
        "e2e": {arm: sorted(set(SHORT_IDS) - {
            r.get("scenario_id") for r in _read_jsonl(os.path.join(
                outdir, "trace-%s.jsonl" % arm.lower()))})
            for arm in ["ARM1", "ARM2", "ARM3", "ARM4", "ARM5"]},
    }
    with open(os.path.join(outdir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    with open(os.path.join(outdir, "teardown.txt"), "w",
              encoding="utf-8") as f:
        f.write("production untouched: DictStore in-memory only; "
                "no vectors/S3/RDS writes by construction.\n")
    return summary


if __name__ == "__main__":
    from tests.benchmarks.mantle_live import MANTLE_IDS
    try:
        summary = main()
    except StopRun as e:
        print("STOP: %s" % e)
        raise SystemExit(3)
    except BudgetExceeded as e:
        print("STOP (budget): %s" % e)
        raise SystemExit(4)
    print(json.dumps(summary, indent=1, sort_keys=True))
