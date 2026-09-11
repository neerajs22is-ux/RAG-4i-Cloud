"""Final controlled benchmark: isolation runners, grader, judge.

Free unless a live Bedrock provider is passed in. Modes:
- ANSWER ISOLATION (REASONING off, REVIEW off, frozen evidence snapshot)
- PLANNER ISOLATION (REASONING on, REVIEW off)
- REVIEWER ISOLATION (REASONING off, REVIEW on, frozen Q/A/E triple)
- FULL E2E (both on; reviewer bundle only where the arm has one)

No app code is modified here; all runners call the frozen code paths
with explicit bundles. No silent substitution: missing/unavailable
models raise and the arm/role is recorded BLOCKED.
"""

import hashlib
import json
import os
import random
import time

HERE = os.path.dirname(os.path.abspath(__file__))

JUDGE_SCHEMA_VERSION = 1
JUDGE_DIMS = ("clarity", "completeness", "synthesis", "usefulness")
JUDGE_RATIONALE_CHARS = 500

FAILURE_TAXONOMY = (
    "WRONG_DOCUMENT", "WRONG_VERSION", "MISSED_REQUESTED_ASPECT",
    "UNSUPPORTED_CLAIM", "BAD_COMPARISON", "BAD_SYNTHESIS",
    "BAD_REFUSAL", "BAD_CLARIFICATION", "RETRIEVAL_FAILURE",
    "PLANNER_FAILURE", "OVER_ESCALATION", "UNDER_ESCALATION",
    "REVIEWER_RUBBER_STAMP", "REVIEWER_FALSE_REPAIR", "SCHEMA_FAILURE",
    "TIMEOUT", "MODEL_ERROR", "QUOTA_ERROR", "ACCESS_ERROR",
    "SCOPE_VIOLATION", "OTHER",
)

# scenario_ids whose behavior is fully deterministic (no paid repeat
# value beyond smoke coverage).
SMOKE_IDS = (
    "S01-normal-lockin", "S04-sla", "S11-exact-clause",
    "S31-ambiguous", "S32-ambiguous-lease", "S37-greeting",
    "S38-refusal", "S39-refusal-2",
)


def load_scenarios(path=None):
    path = path or os.path.join(HERE, "scenarios_v1.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    required = set(data.get("required_suite_keys", []))
    assert required, "scenarios file missing required_suite_keys"
    for sc in data["scenarios"]:
        missing = required - set(sc.keys())
        if missing:
            raise ValueError(
                "scenario %s missing gold keys: %s"
                % (sc.get("scenario_id"), sorted(missing)))
    ids = [s["scenario_id"] for s in data["scenarios"]]
    assert len(ids) == len(set(ids)), "duplicate scenario_id"
    return data


def corpus_snapshot_id(documents):
    blob = json.dumps(documents, sort_keys=True).encode("utf-8")
    return "corpus-" + hashlib.sha256(blob).hexdigest()[:12]


def build_store(scenario, documents):
    """DictStore exactly like harness.run_case (fresh sid per call)."""
    from .harness import DictStore
    import secrets
    docs = []
    for key in scenario["documents"]:
        for chunk in documents[key]["chunks"]:
            docs.append({"file": documents[key]["file"],
                         "page": chunk.get("page", 0),
                         "text": chunk["text"]})
    sess = scenario.get("session") or {}
    bind_sid = None
    if sess.get("documents"):
        indexing_sid = secrets.token_hex(16)
        for entry in sess["documents"]:
            for chunk in entry["chunks"]:
                docs.append({"file": entry["file"],
                             "page": chunk.get("page", 0),
                             "text": chunk["text"],
                             "scope": "session", "sid": indexing_sid})
        bind_sid = indexing_sid if sess.get("bind", "own") == "own" \
            else secrets.token_hex(16)
    context = [dict(m) for m in (sess.get("prior_turns") or [])]
    return DictStore(docs), bind_sid, context


def snapshot_evidence(scenario, documents, config=None):
    """Frozen retrieval, run ONCE per scenario. Returns evidence dict."""
    import config as config_mod
    from backend import retrieve_documents
    store, bind_sid, context = build_store(scenario, documents)
    cfg = config or config_mod.load_config()
    retrieved = retrieve_documents(
        scenario["query"], config=cfg, vector_store=store,
        session_id=bind_sid)
    evidence = []
    for s in retrieved:
        evidence.append({
            "chunk_id": s.get("chunk_id"),
            "file_name": s.get("file_name"),
            "page": s.get("page"),
            "document_id": s.get("document_id"),
            "score": s.get("score"),
            "content": s.get("content", ""),
        })
    files = sorted({e["file_name"] for e in evidence if e["file_name"]})
    return {"evidence": evidence, "files": files, "session_id": bind_sid,
            "context": context, "query": scenario["query"]}


def _support_template(question, evidence):
    from answer_support import assess_support
    from backend import _select_template
    support = assess_support(question, [
        {"content": e.get("content", "")} for e in evidence])
    return support, _select_template(support.get("level"))


def run_answer_model(answer_provider, snapshot, config=None):
    """One streaming answer call on frozen evidence (TTFT probe)."""
    _, template = _support_template(snapshot["query"],
                                    snapshot["evidence"])
    context_text = "\n\n---\n\n".join(
        e.get("content", "") for e in snapshot["evidence"])
    t0 = time.monotonic()
    first_ms = None
    pieces = []
    for piece in answer_provider.generate_stream(
            context_text, snapshot["query"], template):
        if first_ms is None:
            first_ms = max(0, int((time.monotonic() - t0) * 1000))
        pieces.append(piece)
    total_ms = max(0, int((time.monotonic() - t0) * 1000))
    if first_ms is None:
        first_ms = total_ms
    return "".join(pieces), {"ttft_ms": first_ms, "total_ms": total_ms,
                             "template": template[:60]}


def run_answer_isolation(answer_provider, snapshot, config=None,
                         stream_probe=True):
    """ANSWER ISOLATION: identical Q/E/chunks/ordering for every model."""
    import time as _t
    t0 = _t.monotonic()
    from backend import generate_answer
    import config as config_mod
    cfg = config or config_mod.load_config()
    support, template = _support_template(snapshot["query"],
                                          snapshot["evidence"])
    context_text = "\n\n---\n\n".join(
        e.get("content", "") for e in snapshot["evidence"])
    answer = generate_answer(snapshot["query"], [
        {"content": e.get("content", ""),
         "file_name": e.get("file_name"), "page": e.get("page"),
         "chunk_id": e.get("chunk_id"),
         "document_id": e.get("document_id"),
         "score": e.get("score")}
        for e in snapshot["evidence"]],
        config=cfg, llm_provider=answer_provider,
        prompt_template=template)
    gen_ms = max(0, int((_t.monotonic() - t0) * 1000))
    ttft_ms = None
    if stream_probe:
        _, stream_info = run_answer_model(answer_provider, snapshot,
                                          config=cfg)
        ttft_ms = stream_info["ttft_ms"]
    return {"answer": answer, "support_level": support.get("level"),
            "template": template[:60], "generation_ms": gen_ms,
            "ttft_ms": ttft_ms,
            "evidence_ids": [e.get("chunk_id")
                             for e in snapshot["evidence"]]}


def run_planner_isolation(planner_bundle, snapshot, store_files):
    """PLANNER ISOLATION: same Q/metadata/files/presence for every model."""
    t0 = time.monotonic()
    planner = planner_bundle["planner"]
    out = planner.plan(
        snapshot["query"], available_files=store_files,
        session_id=snapshot.get("session_id"),
        context=snapshot.get("context") or [])
    # Planners return (payload, call_meta); payload may be UNVALIDATED.
    if isinstance(out, tuple) and len(out) == 2:
        raw, call_meta = out
    else:
        raw, call_meta = out, {}
    latency_ms = (call_meta or {}).get("latency_ms")
    if latency_ms is None:
        latency_ms = max(0, int((time.monotonic() - t0) * 1000))
    from query_planner import validate_plan
    plan, failure = validate_plan(
        raw, available_files=store_files,
        session_id=snapshot.get("session_id"))
    return {"raw_type": type(raw).__name__, "plan": plan,
            "failure": failure, "latency_ms": latency_ms,
            "schema_valid": plan is not None and failure is None}


def run_reviewer_isolation(reviewer_bundle, snapshot, answer,
                           workflow, plan_meta=None):
    """REVIEWER ISOLATION: frozen (Q,A,E) triple, only reviewer varies."""
    t0 = time.monotonic()
    reviewer = reviewer_bundle["reviewer"]
    out = reviewer.review(
        snapshot["query"], answer, snapshot["evidence"],
        workflow=workflow,
        available_files=snapshot.get("files"),
        session_id=snapshot.get("session_id"), plan_meta=plan_meta)
    # Reviewers return (payload, call_meta); payload may be UNVALIDATED.
    if isinstance(out, tuple) and len(out) == 2:
        raw, call_meta = out
    else:
        raw, call_meta = out, {}
    latency_ms = (call_meta or {}).get("latency_ms")
    if latency_ms is None:
        latency_ms = max(0, int((time.monotonic() - t0) * 1000))
    from answer_reviewer import validate_verdict
    verdict, failure = validate_verdict(raw)
    return {"verdict": verdict, "failure": failure,
            "latency_ms": latency_ms,
            "schema_valid": verdict is not None and failure is None}


def grade_deterministic(scenario, answer, sources, info, evidence,
                        planner_out=None, reviewer_out=None,
                        repair_count=0):
    """Deterministic grading against gold. Returns grade dict."""
    from answer_support import assess_support, citation_guard
    failures = []
    cats = []
    ans_low = (answer or "").lower()
    for fact in scenario.get("required_facts", []):
        if fact.lower() not in ans_low:
            failures.append("missing required fact: %s" % fact[:60])
            cats.append("MISSED_REQUESTED_ASPECT")
    for banned in scenario.get("forbidden_claims", []):
        if banned.lower() in ans_low:
            failures.append("forbidden claim in answer: %s" % banned[:60])
            cats.append("UNSUPPORTED_CLAIM")
    cited = {(s or {}).get("file_name") for s in (sources or [])}
    for doc in scenario.get("expected_documents", []):
        if doc not in cited:
            failures.append("missing expected document: %s" % doc)
            cats.append("WRONG_DOCUMENT")
    if scenario.get("expected_workflow") and \
            (info or {}).get("workflow") != scenario["expected_workflow"]:
        failures.append("workflow %s != %s" % (
            (info or {}).get("workflow"), scenario["expected_workflow"]))
        cats.append("OTHER")
    exp_support = scenario.get("expected_support_level")
    if exp_support and exp_support != "direct" and sources:
        got = assess_support(scenario["query"], sources)["level"]
        if got != exp_support and not (exp_support == "direct"):
            failures.append("support %s != %s" % (got, exp_support))
            cats.append("OTHER")
    marker = scenario.get("expected_clarification")
    if marker:
        if marker not in (answer or ""):
            failures.append("missing clarification behavior")
            cats.append("BAD_CLARIFICATION")
    if scenario.get("expected_scope") == "session" and not sources:
        failures.append("session evidence missing")
        cats.append("SCOPE_VIOLATION")
    guard = citation_guard(answer or "", sources or [])
    if planner_out is not None and not planner_out.get("schema_valid"):
        failures.append("planner schema invalid: %s"
                        % planner_out.get("failure"))
        cats.append("SCHEMA_FAILURE")
    if reviewer_out is not None and not reviewer_out.get("schema_valid"):
        failures.append("reviewer schema invalid: %s"
                        % reviewer_out.get("failure"))
        cats.append("SCHEMA_FAILURE")
    if repair_count > 1:
        failures.append("repair bound violated: %d" % repair_count)
        cats.append("OTHER")
    # Rubber-stamp: PASS while deterministic grading found grounding gap.
    rubber = False
    if reviewer_out and (reviewer_out.get("verdict") or {}).get(
            "verdict") == "pass":
        if any(c in ("MISSED_REQUESTED_ASPECT", "UNSUPPORTED_CLAIM",
                     "WRONG_DOCUMENT", "WRONG_VERSION")
               for c in cats):
            rubber = True
            cats.append("REVIEWER_RUBBER_STAMP")
    grounded = not failures
    if scenario.get("severity") == "critical" and failures:
        cats.append("CRITICAL")
    return {"grounded": grounded, "failures": failures,
            "failure_categories": sorted(set(cats)),
            "guard_flagged": guard["count"], "guard_total": guard["total"],
            "rubber_stamp": rubber, "repair_count": repair_count}


JUDGE_PROMPT = (
    "You are a blind answer-quality judge. You do not know which model "
    "produced which answer. Score ONLY clarity, completeness of "
    "explanation, synthesis quality, and usefulness against the evidence. "
    "Return ONE JSON object with exactly these keys: schema_version (=1), "
    "overall_score (0-10), dimensions ({{clarity, completeness, synthesis, "
    "usefulness}}, each 0-10), critical_failure (boolean), rationale "
    "(<=500 chars). Return ONLY the JSON object.\n"
    "Question: {question}\nEvidence:\n{evidence}\n"
    "Candidate answers (randomized labels):\n{candidates}\n"
)


def build_judge_input(question, evidence_text, labeled_answers):
    ev = (evidence_text or "")[:4000]
    lines = []
    for label in sorted(labeled_answers):
        lines.append("[%s]\n%s" % (label, (labeled_answers[label]
                                           or "")[:2000]))
    return JUDGE_PROMPT.format(question=(question or "")[:500], evidence=ev,
                               candidates="\n".join(lines))


def validate_judge(raw):
    if not isinstance(raw, dict):
        return None, "schema"
    if set(raw.keys()) != {"schema_version", "overall_score",
                           "dimensions", "critical_failure", "rationale"}:
        return None, "schema"
    if raw.get("schema_version") != JUDGE_SCHEMA_VERSION:
        return None, "schema"
    if not isinstance(raw.get("overall_score"), (int, float)):
        return None, "schema"
    if not 0 <= float(raw["overall_score"]) <= 10:
        return None, "schema"
    dims = raw.get("dimensions")
    if not isinstance(dims, dict) or set(dims.keys()) != set(JUDGE_DIMS):
        return None, "schema"
    for dim in JUDGE_DIMS:
        if not isinstance(dims[dim], (int, float)):
            return None, "schema"
        if not 0 <= float(dims[dim]) <= 10:
            return None, "schema"
    if not isinstance(raw.get("critical_failure"), bool):
        return None, "schema"
    rationale = raw.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        return None, "schema"
    if len(rationale) > JUDGE_RATIONALE_CHARS:
        return None, "schema"
    return {"schema_version": JUDGE_SCHEMA_VERSION,
            "overall_score": float(raw["overall_score"]),
            "dimensions": {d: float(dims[d]) for d in JUDGE_DIMS},
            "critical_failure": raw["critical_failure"],
            "rationale": rationale.strip()}, None


def run_judge(judge_provider, question, evidence_text, labeled_answers,
              seed=74154):
    """Blind judge call. Labels are pre-randomized by the caller."""
    if judge_provider is None:
        raise ValueError("judge provider is required (no substitution).")
    prompt = build_judge_input(question, evidence_text, labeled_answers)
    raw_text = judge_provider.generate(
        prompt, question, "{context}\n\n{question}")
    try:
        raw = json.loads(raw_text[raw_text.index("{"):
                                  raw_text.rindex("}") + 1])
    except (ValueError, IndexError):
        return None, "malformed"
    return validate_judge(raw)


def calibration_packet(results, seed=74154, frac=0.18):
    """Anonymized 15-20% sample shell. Scores are NEVER fabricated."""
    rng = random.Random(seed)
    pool = [r for r in results if r.get("answer")]
    n = max(1, min(len(pool), round(len(pool) * frac)))
    if n / max(1, len(pool)) < 0.15 and len(pool) >= 7:
        n = max(n, 7)
    sample = rng.sample(pool, n)
    return {"calibration_status": "PENDING_HUMAN",
            "seed": seed, "sample_size": n,
            "population": len(pool),
            "items": [{"scenario_id": r.get("scenario_id"),
                       "arm": r.get("arm"), "run": r.get("run"),
                       "question": r.get("question"),
                       "evidence_excerpt": (r.get("evidence_excerpt")
                                            or "")[:400],
                       "answer": r.get("answer"),
                       "human_overall_score": None,
                       "human_critical_failure": None} for r in sample]}


def write_trace(path, record):
    """One JSONL trace record. Bounded excerpts only, never secrets."""
    safe = dict(record)
    ev = safe.get("evidence_excerpt")
    if isinstance(ev, str):
        safe["evidence_excerpt"] = ev[:1500]
    if isinstance(safe.get("answer"), str):
        safe["answer"] = safe["answer"][:4000]
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe, sort_keys=True) + "\n")
