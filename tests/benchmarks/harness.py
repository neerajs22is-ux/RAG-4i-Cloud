"""Deterministic benchmark runner (doubles by default; live opt-in)."""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))

REQUIRED_CASE_KEYS = {"id", "category", "query", "documents", "expect"}
REQUIRED_EXPECT_KEYS = {"workflow"}
CATEGORIES = {"normal", "conversational", "partial", "unsupported",
              "followup", "comparison", "extraction", "summary",
              "ambiguous", "session-upload"}


class Doc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})
        self.id = None


def live_bedrock_llm():
    """Bedrock answer provider for live benchmark comparison, or None.

    Enabled ONLY with BENCHMARK_LIVE_BEDROCK=1 plus a valid bedrock
    configuration (LLM_PROVIDER=bedrock, ANSWER_MODEL_ID set). The normal
    suite never enables it: no live AWS calls, no credentials needed.
    Callers pass the result as run_all(llm=...) with an explicit
    model_key (e.g. "sonnet-4.6") for cost math.
    """
    if os.environ.get("BENCHMARK_LIVE_BEDROCK") != "1":
        return None
    import config as config_mod
    from bedrock_provider import BedrockConverseProvider

    cfg = config_mod.load_config()
    if (getattr(cfg, "llm_provider", "") or "").lower() != "bedrock":
        raise ValueError(
            "BENCHMARK_LIVE_BEDROCK=1 requires LLM_PROVIDER=bedrock.")
    if not (getattr(cfg, "answer_model_id", "") or "").strip():
        raise ValueError(
            "BENCHMARK_LIVE_BEDROCK=1 requires ANSWER_MODEL_ID.")
    return BedrockConverseProvider(
        model_id=cfg.answer_model_id,
        region=cfg.bedrock_region,
        temperature=cfg.llm_temperature,
    )


def live_bedrock_planner():
    """Bedrock reasoning bundle for live benchmark comparison, or None.

    Same gate as live_bedrock_llm plus REASONING_ENABLED=1,
    REASONING_PROVIDER=bedrock and REASONING_MODEL_ID. Raises loudly on
    misconfiguration; never substitutes another model. Normal suite
    never enables it.
    """
    if os.environ.get("BENCHMARK_LIVE_BEDROCK") != "1":
        return None
    import config as config_mod
    from query_planner import (PlanCache, build_llm_planner,
                               reasoning_enabled)

    cfg = config_mod.load_config()
    if not reasoning_enabled(cfg):
        raise ValueError(
            "BENCHMARK_LIVE_BEDROCK=1 with planner requires "
            "REASONING_ENABLED=1.")
    if (getattr(cfg, "reasoning_provider", "") or "").lower() != "bedrock":
        raise ValueError(
            "BENCHMARK_LIVE_BEDROCK=1 with planner requires "
            "REASONING_PROVIDER=bedrock.")
    if not (getattr(cfg, "reasoning_model_id", "") or "").strip():
        raise ValueError(
            "BENCHMARK_LIVE_BEDROCK=1 with planner requires "
            "REASONING_MODEL_ID.")
    planner = build_llm_planner(cfg)
    return {"planner": planner, "cache": PlanCache(),
            "provider": planner.provider, "model": planner.model,
            "timeout_s": planner.timeout_s,
            "_meta_base": {"reasoning_used": True,
                           "reasoning_provider": planner.provider,
                           "reasoning_model": planner.model}}


def live_bedrock_reviewer():
    """Bedrock reviewer bundle for live benchmark comparison, or None.

    Same gate as live_bedrock_llm plus REVIEW_ENABLED=1,
    REVIEW_PROVIDER=bedrock and REVIEW_MODEL_ID. Raises loudly on
    misconfiguration; never substitutes another model. Normal suite
    never enables it.
    """
    if os.environ.get("BENCHMARK_LIVE_BEDROCK") != "1":
        return None
    import config as config_mod
    from answer_reviewer import build_llm_reviewer, review_enabled

    cfg = config_mod.load_config()
    if not review_enabled(cfg):
        raise ValueError(
            "BENCHMARK_LIVE_BEDROCK=1 with reviewer requires "
            "REVIEW_ENABLED=1.")
    if (getattr(cfg, "review_provider", "") or "").lower() != "bedrock":
        raise ValueError(
            "BENCHMARK_LIVE_BEDROCK=1 with reviewer requires "
            "REVIEW_PROVIDER=bedrock.")
    if not (getattr(cfg, "review_model_id", "") or "").strip():
        raise ValueError(
            "BENCHMARK_LIVE_BEDROCK=1 with reviewer requires "
            "REVIEW_MODEL_ID.")
    reviewer = build_llm_reviewer(cfg)
    return {"reviewer": reviewer,
            "provider": reviewer.provider, "model": reviewer.model,
            "timeout_s": reviewer.timeout_s,
            "_meta_base": {"review_used": True,
                           "review_provider": reviewer.provider,
                           "review_model": reviewer.model}}


# Live benchmark arms (exact IDs documented in LIVE_ARMS.md; IDs here
# are explicit and loud, never silent defaults). run_live_arm builds
# providers from these IDs directly so no .env mutation is needed and
# a BLOCKED arm raises instead of substituting another model.
LIVE_ARMS = {
    "A": {"answer_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
          "answer_region": "us-east-1", "answer_key": "haiku-4.5",
          "reasoning_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
          "reasoning_region": "us-east-1",
          "reviewer_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
          "reviewer_region": "us-east-1"},
    "B": {"answer_model": "us.anthropic.claude-sonnet-5",
          "answer_region": "us-east-1", "answer_key": "sonnet-5",
          "reasoning_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
          "reasoning_region": "us-east-1",
          "reviewer_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
          "reviewer_region": "us-east-1"},
    "C": {"answer_model": "in.openai.gpt-5.6-luna",
          "answer_region": "ap-south-1", "answer_key": "luna",
          "reasoning_model": "in.openai.gpt-5.6-luna",
          "reasoning_region": "ap-south-1",
          "reviewer_model": "in.openai.gpt-5.6-luna",
          "reviewer_region": "ap-south-1"},
}


def run_live_arm(arm_name, cases_path=None):
    """Run one paid live arm end-to-end (answer+reasoning+reviewer).

    Requires BENCHMARK_LIVE_BEDROCK=1 in the environment (paid calls);
    raises otherwise, and raises (never substitutes) when any arm model
    is unreachable, denied, or missing access. cases_path selects the
    S (default, frozen) or M corpus file. Failures are recorded
    per-case by run_all; no cross-arm fallback exists by construction.
    """
    if os.environ.get("BENCHMARK_LIVE_BEDROCK") != "1":
        raise ValueError(
            "run_live_arm requires BENCHMARK_LIVE_BEDROCK=1 (paid calls).")
    if arm_name not in LIVE_ARMS:
        raise ValueError(
            "unknown live arm %r (expected one of %s)."
            % (arm_name, sorted(LIVE_ARMS)))
    import config as config_mod
    from answer_reviewer import build_llm_reviewer
    from bedrock_provider import BedrockConverseProvider
    from query_planner import PlanCache, build_llm_planner

    arm = LIVE_ARMS[arm_name]
    llm = BedrockConverseProvider(
        model_id=arm["answer_model"], region=arm["answer_region"],
        temperature=0.0)
    reasoning_cfg = config_mod.AppConfig(
        reasoning_enabled="1", reasoning_provider="bedrock",
        reasoning_model_id=arm["reasoning_model"],
        reasoning_region=arm["reasoning_region"])
    reasoning = build_llm_planner(reasoning_cfg)
    planner_bundle = {"planner": reasoning, "cache": PlanCache(),
                      "provider": reasoning.provider,
                      "model": reasoning.model,
                      "timeout_s": reasoning.timeout_s,
                      "_meta_base": {"reasoning_used": True,
                                     "reasoning_provider":
                                         reasoning.provider,
                                     "reasoning_model": reasoning.model}}
    review_cfg = config_mod.AppConfig(
        review_enabled="1", review_provider="bedrock",
        review_model_id=arm["reviewer_model"],
        review_region=arm["reviewer_region"])
    reviewer = build_llm_reviewer(review_cfg)
    reviewer_bundle = {"reviewer": reviewer,
                       "provider": reviewer.provider,
                       "model": reviewer.model,
                       "timeout_s": reviewer.timeout_s,
                       "_meta_base": {"review_used": True,
                                      "review_provider": reviewer.provider,
                                      "review_model": reviewer.model}}
    return run_all(cases_path=cases_path, llm=llm,
                   model_key=arm["answer_key"],
                   planner=planner_bundle, reviewer=reviewer_bundle)


class DictStore:
    """Deterministic word-overlap store (no embeddings, no network).

    Documents may carry "scope"/"sid" (session rows); filtering mirrors
    the provider rule: persistent always visible, session rows only on
    a matching session_id. Omitting both keeps legacy persistent-only.
    """

    def __init__(self, documents, score=0.9):
        self._docs = list(documents or [])
        self._score = score

    def _visible(self, d, session_id):
        if d.get("scope") == "session":
            return d.get("sid") is not None and d.get("sid") == session_id
        return True

    def _as_doc(self, d, i):
        return Doc(d["text"], {"source_path": "/bench/" + d["file"],
                               "source": "/bench/" + d["file"],
                               "file_name": d["file"],
                               "page": d.get("page", 0),
                               "document_id": "bench-" + d["file"],
                               "chunk_id": f"bench-{i}"})

    def search(self, query, k=5, session_id=None):
        words = [w for w in (query or "").lower().split() if len(w) >= 4]
        out = []
        for i, d in enumerate(self._docs):
            if not self._visible(d, session_id):
                continue
            low = d["text"].lower()
            if any(w in low for w in words):
                out.append((self._as_doc(d, i), self._score))
        return out[:k]

    def chunks_for_source(self, file_name, limit=8, session_id=None):
        out = []
        for i, d in enumerate(self._docs):
            if d["file"] == file_name and self._visible(d, session_id):
                out.append((self._as_doc(d, i), None))
            if len(out) >= limit:
                break
        return out

    def list_sources(self, limit=50, session_id=None):
        seen = []
        for d in self._docs:
            if d["file"] not in seen and self._visible(d, session_id):
                seen.append(d["file"])
        return [{"file_name": f, "document_id": "bench-" + f}
                for f in seen[:limit]]

    def get_status(self):
        return {"ready": bool(self._docs)}


class EchoLLM:
    """Deterministic stand-in: answers quote evidence (grounded by design)."""

    def __init__(self):
        self.calls = 0

    def generate(self, context, question, prompt_template):
        self.calls += 1
        head = (context or "")[:400].replace("\n", " ")
        return f"Evidence says: {head}"

    def generate_stream(self, context, question, prompt_template):
        yield self.generate(context, question, prompt_template)

    def is_reachable(self, timeout=3.0):
        return True

    @property
    def describe(self):
        return "Echo"


def load_cases(path=None):
    """Load + validate the versioned case file (ValueError on defect)."""
    path = path or os.path.join(HERE, "cases.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data.get("version"), int):
        raise ValueError("cases.json: missing integer version")
    if not isinstance(data.get("documents"), dict):
        raise ValueError("cases.json: missing documents map")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases.json: missing cases list")
    ids = set()
    for c in cases:
        if not REQUIRED_CASE_KEYS.issubset(c):
            raise ValueError(f"case missing keys: {c.get('id')}")
        if c["id"] in ids:
            raise ValueError(f"duplicate case id: {c['id']}")
        ids.add(c["id"])
        if c["category"] not in CATEGORIES:
            raise ValueError(f"unknown category: {c['id']}")
        if not REQUIRED_EXPECT_KEYS.issubset(c["expect"]):
            raise ValueError(f"case missing expect.workflow: {c['id']}")
        for key in c["documents"]:
            if key not in data["documents"]:
                raise ValueError(f"unknown document key: {key}")
        for entry in c.get("session_documents") or []:
            if not isinstance(entry, dict) or not entry.get("file") \
                    or not isinstance(entry.get("chunks"), list) \
                    or not entry["chunks"]:
                raise ValueError(f"bad session_documents: {c['id']}")
            for chunk in entry["chunks"]:
                if not isinstance(chunk, dict) or not chunk.get("text"):
                    raise ValueError(f"bad session chunk: {c['id']}")
        if c.get("bind") not in (None, "other"):
            raise ValueError(f"bad bind: {c['id']}")
    return data


def check_expectations(answer, sources, info, expect):
    """Deterministic property checks (never exact model wording)."""
    exp = expect or {}
    failures = []
    if exp.get("workflow") and info.get("workflow") != exp["workflow"]:
        failures.append(f"workflow {info.get('workflow')} != {exp['workflow']}")
    if len(sources) < int(exp.get("min_sources", 0)):
        failures.append(f"only {len(sources)} sources")
    files = {(s or {}).get("file_name") for s in sources}
    for f in exp.get("must_cite", []):
        if f not in files:
            failures.append(f"missing citation: {f}")
    for f in exp.get("must_not_cite", []):
        if f in files:
            failures.append(f"leaked citation: {f}")
    blob = ((answer or "")
            + " ".join((s or {}).get("content", "") for s in sources)).lower()
    for term in exp.get("must_include_any", []):
        if term.lower() not in blob:
            failures.append(f"missing term: {term}")
    for term in exp.get("must_exclude", []):
        if term.lower() in (answer or "").lower():
            failures.append(f"forbidden term in answer: {term}")
    if exp.get("support") and sources:
        from answer_support import assess_support
        level = assess_support(exp.get("support_query", ""), sources)["level"]
        if level != exp["support"]:
            failures.append(f"support {level} != {exp['support']}")
    return failures


def run_case(case, documents, llm=None, model_key="echo", planner=None,
             reviewer=None):
    """Run one case deterministically.

    Cases with "session_documents" index them under a fresh session id
    and bind it for retrieval, unless "bind" is "other" (a different
    fresh id: the cross-session probe). Cases with "requires" stay
    skipped (future gates). planner is an optional query_planner bundle
    (or the string "deterministic" for the no-LLM planner); without it
    the frozen 4.12 path runs. reviewer is an optional
    answer_reviewer bundle (or the string "deterministic" for the
    no-LLM pass-only reviewer); without it no review runs. When a
    reviewer is given, the reviewed path runs (planner bundle honored
    when also given) and review telemetry lands per-result under
    "review" (metadata only).
    """
    from query_planner import DeterministicPlanner, PlanCache, query_planned
    from workflows import query_workflow
    from .metrics import compute
    import config

    if case.get("requires"):
        return {"id": case["id"], "status": "skipped",
                "reason": "requires-%s" % (case['requires'],)}
    import secrets
    docs = []
    for key in case["documents"]:
        for chunk in documents[key]["chunks"]:
            docs.append({"file": documents[key]["file"],
                         "page": chunk.get("page", 0),
                         "text": chunk["text"]})
    indexing_sid = None
    bind_sid = None
    if case.get("session_documents"):
        indexing_sid = secrets.token_hex(16)
        for entry in case["session_documents"]:
            for chunk in entry["chunks"]:
                docs.append({"file": entry["file"],
                             "page": chunk.get("page", 0),
                             "text": chunk["text"],
                             "scope": "session", "sid": indexing_sid})
        bind_sid = secrets.token_hex(16) if case.get("bind") == "other" \
            else indexing_sid
    store = DictStore(docs)
    llm = llm or EchoLLM()
    context = [m for m in (case.get("context") or [])]
    t0 = time.monotonic()
    reasoning = None
    review = None
    try:
        if reviewer is None and planner is None:
            answer, sources, info = query_workflow(
                case["query"], config=config.load_config(),
                vector_store=store, llm_provider=llm,
                conversation_context=context, session_id=bind_sid)
        elif reviewer is None:
            if planner == "deterministic":
                bundle = {"planner": DeterministicPlanner(),
                          "cache": PlanCache(),
                          "provider": "deterministic", "model": "none",
                          "timeout_s": 5,
                          "_meta_base": {"reasoning_used": False}}
            else:
                bundle = planner
            answer, sources, info, reasoning = query_planned(
                case["query"], config=config.load_config(),
                vector_store=store, llm_provider=llm,
                conversation_context=context, session_id=bind_sid,
                bundle=bundle)
        else:
            from answer_reviewer import (DeterministicReviewer,
                                         query_reviewed)
            if planner is None or planner == "deterministic":
                if planner == "deterministic":
                    planner_bundle = {
                        "planner": DeterministicPlanner(),
                        "cache": PlanCache(),
                        "provider": "deterministic", "model": "none",
                        "timeout_s": 5,
                        "_meta_base": {"reasoning_used": False}}
                else:
                    planner_bundle = None
            else:
                planner_bundle = planner
            if reviewer == "deterministic":
                reviewer_bundle = {
                    "reviewer": DeterministicReviewer(),
                    "provider": "deterministic", "model": "none",
                    "timeout_s": 5,
                    "_meta_base": {"review_used": False}}
            else:
                reviewer_bundle = reviewer
            answer, sources, info, reasoning, review = query_reviewed(
                case["query"], config=config.load_config(),
                vector_store=store, llm_provider=llm,
                conversation_context=context, session_id=bind_sid,
                planner_bundle=planner_bundle,
                reviewer_bundle=reviewer_bundle)
    except Exception as e:  # harness reports, never raises
        return {"id": case["id"], "status": "error", "error": repr(e)[:200]}
    wall_ms = max(0, int((time.monotonic() - t0) * 1000))
    evidence = " ".join((s or {}).get("content", "") for s in sources)
    metrics = compute(answer, sources, info, {"wall_ms": wall_ms},
                      evidence_text=evidence, model_key=model_key,
                      reasoning=reasoning, review=review)
    failures = check_expectations(answer, sources, info, case.get("expect"))
    result = {"id": case["id"], "status": "pass" if not failures else "fail",
              "failures": failures, "metrics": metrics}
    if reasoning is not None:
        result["reasoning"] = {k: reasoning.get(k) for k in (
            "reasoning_used", "escalation_reason",
            "planner_failure_category", "planner_cached",
            "planner_latency_ms") if k in reasoning}
    if review is not None:
        result["review"] = {k: review.get(k) for k in (
            "review_used", "review_verdict", "review_trigger",
            "review_failure_category", "repair_attempted",
            "repair_succeeded", "review_latency_ms",
            "guard_before_flagged", "guard_before_total",
            "guard_after_flagged", "guard_after_total") if k in review}
    return result


def run_all(cases_path=None, llm=None, model_key="echo", planner=None,
            reviewer=None):
    """Run every case; returns {"results": [...], "summary": {...}}.

    planner=None reproduces the frozen baseline; pass "deterministic"
    or a query_planner bundle to measure the planned dimension. Planner
    telemetry lands per-result under "reasoning" (metadata only).
    reviewer=None means no review; pass "deterministic" or an
    answer_reviewer bundle to measure the reviewed dimension. Review
    telemetry lands per-result under "review" (metadata only), and the
    summary carries aggregate review counters (all counts, no text).
    Synthetic benchmark data only.
    """
    data = load_cases(cases_path)
    results = [run_case(c, data["documents"], llm=llm, model_key=model_key,
                        planner=planner, reviewer=reviewer)
               for c in data["cases"]]
    done = [r for r in results if r["status"] != "skipped"]
    summary = {"total": len(results),
               "passed": sum(1 for r in done if r["status"] == "pass"),
               "failed": sum(1 for r in done if r["status"] != "pass"),
               "skipped": sum(1 for r in results if r["status"] == "skipped"),
               "cases_version": data["version"]}
    if reviewer is not None:
        reviewed = [r for r in done if (r.get("metrics") or {}).get(
            "review_used")]
        repairs = [r for r in reviewed if (r.get("metrics") or {}).get(
            "repair_attempted")]
        succeeded = [r for r in repairs if (r.get("metrics") or {}).get(
            "repair_succeeded")]
        guard_fixed = 0
        for r in succeeded:
            m = r.get("metrics") or {}
            before = m.get("guard_before_flagged")
            after = m.get("guard_after_flagged")
            if isinstance(before, int) and isinstance(after, int) \
                    and after < before:
                guard_fixed += 1
        summary["review_used"] = len(reviewed)
        summary["repair_attempted"] = len(repairs)
        summary["repair_succeeded"] = len(succeeded)
        summary["guard_improved"] = guard_fixed
    return {"results": results, "summary": summary}
