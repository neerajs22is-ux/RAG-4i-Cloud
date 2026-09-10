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


def run_case(case, documents, llm=None, model_key="echo"):
    """Run one case deterministically.

    Cases with "session_documents" index them under a fresh session id
    and bind it for retrieval, unless "bind" is "other" (a different
    fresh id: the cross-session probe). Cases with "requires" stay
    skipped (future gates).
    """
    from workflows import query_workflow
    from .metrics import compute
    import config

    if case.get("requires"):
        return {"id": case["id"], "status": "skipped",
                "reason": f"requires-{case['requires']}"}
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
    try:
        answer, sources, info = query_workflow(
            case["query"], config=config.load_config(), vector_store=store,
            llm_provider=llm, conversation_context=context,
            session_id=bind_sid)
    except Exception as e:  # harness reports, never raises
        return {"id": case["id"], "status": "error", "error": repr(e)[:200]}
    wall_ms = max(0, int((time.monotonic() - t0) * 1000))
    evidence = " ".join((s or {}).get("content", "") for s in sources)
    metrics = compute(answer, sources, info, {"wall_ms": wall_ms},
                      evidence_text=evidence, model_key=model_key)
    failures = check_expectations(answer, sources, info, case.get("expect"))
    return {"id": case["id"], "status": "pass" if not failures else "fail",
            "failures": failures, "metrics": metrics}


def run_all(cases_path=None, llm=None, model_key="echo"):
    """Run every case; returns {"results": [...], "summary": {...}}."""
    data = load_cases(cases_path)
    results = [run_case(c, data["documents"], llm=llm, model_key=model_key)
               for c in data["cases"]]
    done = [r for r in results if r["status"] != "skipped"]
    summary = {"total": len(results),
               "passed": sum(1 for r in done if r["status"] == "pass"),
               "failed": sum(1 for r in done if r["status"] != "pass"),
               "skipped": sum(1 for r in results if r["status"] == "skipped"),
               "cases_version": data["version"]}
    return {"results": results, "summary": summary}
