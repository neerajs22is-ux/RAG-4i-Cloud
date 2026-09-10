"""Answer reviewer / bounded repair (Phase 5E).

Core rule: the models may become smarter; the evidence system does not
become less deterministic.

Pipeline position (additive only):

    retrieval (existing, k=5 / threshold 0.3 / MiniLM)
    -> answer generation (existing prompts, temperature, provider)
    -> citation guard (existing, authoritative for grounding)
    -> conditional reviewer (NEW, gated by REVIEW_ENABLED)
    -> validated verdict (deterministic validator, untrusted boundary)
    -> optional ONE deterministic gap-fill retrieval
    -> existing answer regeneration path (evidence set only)
    -> citation guard again
    -> final answer

Rules enforced here:
  - REVIEW_ENABLED=0 (default): no reviewer object is built, no network
    call happens, behavior is identical to the pre-5E path.
  - The reviewer NEVER generates user-facing prose, never retrieves,
    never executes tools/SQL/filesystem, never modifies storage,
    vectors, sessions, scope, or filters. It proposes ONLY a validated
    structured verdict (pass / repair / insufficient).
  - Reviewer input is bounded: effective question, generated answer,
    retrieved/cited evidence excerpts, validated Plan v1 metadata when
    available, workflow, bounded document metadata. Never transcripts,
    paths, secrets, unrelated documents, hidden state, or the corpus.
  - Invocation is deterministic: citation-guard flag, comparison,
    summary, or explicit REVIEW_ALWAYS_IF_CONFIGURED. Ordinary answers
    are not reviewed by default.
  - Only verdict == "repair" can trigger repair, at most ONE cycle,
    through the SAME deterministic retrieval with provider-side scope
    enforcement (persistent OR current session). INSUFFICIENT never
    searches; failures preserve the original answer.

Gap definitions (exact):
  - missing_aspect: an explicitly requested aspect of the user's
    question that is not addressed by the answer using the retrieved
    evidence.
  - unsupported_claim: a materially factual claim in the answer for
    which no retrieved chunk provides support.
  - Style, wording, verbosity, and formatting preferences are NEVER
    evidence gaps.

Pure helpers wherever possible (fully unit-testable without Streamlit
or AWS). Provider-touching constructors fail loudly when explicitly
enabled but misconfigured.
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

REVIEW_SCHEMA_VERSION = 1

VERDICTS = frozenset({"pass", "repair", "insufficient"})

REVIEW_KEYS = frozenset({"schema_version", "verdict", "missing_aspects",
                         "unsupported_claims",
                         "suggested_followup_queries"})

# Reviewer failure categories (closed set; metadata only).
REVIEW_FAILURES = frozenset({"disabled", "unavailable", "credentials",
                             "timeout", "malformed", "schema",
                             "unsupported", "model_error"})

# Invocation triggers (closed set; deterministic policy below).
REVIEW_TRIGGERS = frozenset({"citation_guard", "comparison", "summary",
                             "always", "none"})

# Bounds (hard; validator rejects anything beyond these).
MAX_MISSING_ASPECTS = 4
MAX_UNSUPPORTED_CLAIMS = 4
MAX_FOLLOWUP_QUERIES = 3
MAX_ASPECT_CHARS = 200
MAX_CLAIM_CHARS = 300
MAX_FOLLOWUP_QUERY_CHARS = 200
MIN_FOLLOWUP_QUERY_CHARS = 3
MAX_QUESTION_CHARS = 500
MAX_ANSWER_CHARS = 2000
MAX_EVIDENCE_CHUNKS = 5
MAX_EVIDENCE_CHARS = 1200
MAX_DOC_FILES = 20
MAX_PLAN_REASON_CHARS = 120

# Forbidden output keys (defense in depth; exact key set already
# rejects them, but they are named explicitly so tests and audits can
# assert the reviewer never emits answer text, paths, or secrets).
FORBIDDEN_VERDICT_KEYS = frozenset({
    "answer", "repaired_answer", "final_answer", "generated_answer",
    "text", "response", "file_path", "source_path", "path", "filepath",
    "secret", "secrets", "api_key", "apikey", "password", "token",
})

WORKFLOW_ALLOWLIST = frozenset({"normal", "comparison", "extraction",
                                "summary"})


def _normalize(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _looks_like_path(text):
    """True for strings carrying filesystem path structure."""
    t = text or ""
    if "/" in t or "\\" in t:
        return True
    low = t.lower()
    if ".." in t and ("/" in t or "\\" in t or ".pdf" in low):
        return True
    return False


def _pdf_names(text):
    return [m.lower() for m in
            re.findall(r"[\w][\w\-]*\.pdf", text or "", re.IGNORECASE)]


# ---------- verdict validation (deterministic, strict) ---------- #

def validate_verdict(raw):
    """Validate an UNTRUSTED reviewer verdict.

    Returns (canonical|None, failure|None). Anything malformed,
    unbounded, invented, or out-of-allowlist is rejected -- never
    repaired. Non-dict input is "malformed"; dicts violating the
    schema are "schema".
    """
    if not isinstance(raw, dict):
        return None, "malformed"
    if set(raw.keys()) != REVIEW_KEYS:
        return None, "schema"
    if not isinstance(raw.get("schema_version"), int) or \
            raw.get("schema_version") != REVIEW_SCHEMA_VERSION:
        return None, "schema"
    verdict = raw.get("verdict")
    if verdict not in VERDICTS:
        return None, "schema"
    # Explicit forbidden-key defense (exact set already rejects).
    for key in FORBIDDEN_VERDICT_KEYS:
        if key in raw:
            return None, "schema"
    missing = raw.get("missing_aspects")
    if not isinstance(missing, list) or len(missing) > MAX_MISSING_ASPECTS:
        return None, "schema"
    clean_missing = []
    for item in missing:
        if not isinstance(item, str):
            return None, "schema"
        text = item.strip()
        if not text or len(text) > MAX_ASPECT_CHARS:
            return None, "schema"
        if _looks_like_path(text):
            return None, "schema"
        if text not in clean_missing:
            clean_missing.append(text)
    unsupported = raw.get("unsupported_claims")
    if not isinstance(unsupported, list) or \
            len(unsupported) > MAX_UNSUPPORTED_CLAIMS:
        return None, "schema"
    clean_unsupported = []
    for item in unsupported:
        if not isinstance(item, str):
            return None, "schema"
        text = item.strip()
        if not text or len(text) > MAX_CLAIM_CHARS:
            return None, "schema"
        if _looks_like_path(text):
            return None, "schema"
        if text not in clean_unsupported:
            clean_unsupported.append(text)
    followups = raw.get("suggested_followup_queries")
    if not isinstance(followups, list) or \
            len(followups) > MAX_FOLLOWUP_QUERIES:
        return None, "schema"
    clean_followups = []
    for item in followups:
        if not isinstance(item, str):
            return None, "schema"
        text = item.strip()
        if not text or len(text) > MAX_FOLLOWUP_QUERY_CHARS:
            return None, "schema"
        if _looks_like_path(text):
            return None, "schema"
        if text not in clean_followups:
            clean_followups.append(text)
    # Consistency: pass carries no gaps and no followups.
    if verdict == "pass":
        if clean_missing or clean_unsupported or clean_followups:
            return None, "schema"
    return {"schema_version": REVIEW_SCHEMA_VERSION,
            "verdict": verdict,
            "missing_aspects": clean_missing,
            "unsupported_claims": clean_unsupported,
            "suggested_followup_queries": clean_followups}, None


def validate_followup_queries(queries, *, available_files=None):
    """Filter UNTRUSTED suggested queries to safe retrieval intents.

    Returns (valid_list, rejected_count). Valid queries are stripped,
    bounded, free of path structure, and (when available_files is
    given) reference only known .pdf names. Scope is never altered
    here: callers pass the same session_id to the existing retrieval.
    """
    valid = []
    rejected = 0
    known = set((f or "").lower() for f in (available_files or []))
    for item in queries or []:
        if not isinstance(item, str):
            rejected += 1
            continue
        text = item.strip()
        if len(text) < MIN_FOLLOWUP_QUERY_CHARS or \
                len(text) > MAX_FOLLOWUP_QUERY_CHARS:
            rejected += 1
            continue
        if _looks_like_path(text):
            rejected += 1
            continue
        names = _pdf_names(text)
        if names and known:
            if any(n not in known for n in names):
                rejected += 1
                continue
        if text and text not in valid:
            valid.append(text)
        elif not text:
            rejected += 1
    return valid, rejected


# ---------- bounded reviewer input ---------- #

def build_reviewer_input(question, answer, evidence, *, workflow=None,
                         available_files=None, plan_meta=None,
                         session_bound=False):
    """Bounded reviewer context (metadata + excerpts, never secrets).

    Includes only: effective question (bounded), generated answer
    (bounded), retrieved/cited evidence excerpts (bounded, file_name +
    page labels only, never source_path), validated Plan v1 metadata
    when available (workflow/escalation only, never plan text), the
    workflow name, and bounded document filenames. Never transcripts,
    raw paths, session ids, secrets, unrelated documents, or corpus.
    """
    files = list(available_files or [])[:MAX_DOC_FILES]
    q = re.sub(r"\s+", " ", str(question or "")).strip()
    q = q[:MAX_QUESTION_CHARS]
    a = re.sub(r"\s+", " ", str(answer or "")).strip()
    a = a[:MAX_ANSWER_CHARS]
    chunks = []
    for s in (evidence or [])[:MAX_EVIDENCE_CHUNKS]:
        if not isinstance(s, dict):
            continue
        content = re.sub(r"\s+", " ", str(s.get("content") or "")).strip()
        content = content[:MAX_EVIDENCE_CHARS]
        fname = str(s.get("file_name") or "unknown")[:80]
        page = s.get("page")
        label = fname + (" p. " + str(page) if page is not None else "")
        chunks.append("[" + label + "] " + content)
    plan_bits = []
    if isinstance(plan_meta, dict):
        for key in ("workflow", "escalation_reason"):
            val = plan_meta.get(key)
            if isinstance(val, str) and val:
                plan_bits.append(key + "=" + val[:MAX_PLAN_REASON_CHARS])
    lines = [
        "Review the answer against the evidence as ONE JSON object. "
        "Never write the answer itself.",
        "Schema (exact keys): schema_version=1, verdict in "
        "[pass, repair, insufficient], missing_aspects (0-4 short "
        "phrases), unsupported_claims (0-4 short phrases), "
        "suggested_followup_queries (0-3 short retrieval queries).",
        "missing_aspect: an explicitly requested aspect not addressed "
        "by the answer using the evidence. unsupported_claim: a "
        "materially factual claim with no supporting chunk. Style, "
        "wording, verbosity, and formatting are never gaps.",
        "Question: " + q,
        "Answer: " + a,
        "Workflow: " + str(workflow or "normal")[:20],
        "Known documents: " + (", ".join(files) if files else "(none)"),
        "Session documents bound: " + ("yes" if session_bound else "no"),
    ]
    if plan_bits:
        lines.append("Plan: " + ", ".join(plan_bits))
    if chunks:
        lines.append("Evidence:")
        lines.extend(chunks[:MAX_EVIDENCE_CHUNKS])
    lines.append("Return ONLY the JSON object.")
    return "\n".join(lines)


def supports_native_structured_output():
    """Structured-output capability actually verified in this env.

    Status: langchain-aws is NOT installed in this environment, so
    native Bedrock outputConfig support is UNVERIFIED here; the
    adapter below uses the documented forced-tool-calling path
    (with_structured_output), which works across versions. Every
    reviewer verdict still passes the deterministic validator
    regardless of which structured path produced it.
    """
    return {"mode": "forced-tool-calling",
            "native_outputConfig": "unverified",
            "validator": "deterministic (always)"}


def bedrock_reviewer_structured_fn(chat_model):
    """Adapt a chat model to structured_fn(prompt, schema) -> dict.

    Uses forced tool calling (with_structured_output); the result
    MUST be a dict or a ValueError marks it malformed downstream. No
    answer text is ever produced through this path.
    """

    def call(prompt_text, schema):
        runner = chat_model.with_structured_output(schema)
        result = runner.invoke(prompt_text)
        if not isinstance(result, dict):
            raise ValueError(
                "Reviewer returned non-dict structured output.")
        return result

    return call


# ---------- reviewers ---------- #

class DeterministicReviewer:
    """No-op reviewer: always passes with zero latency.

    Produces a canonical pass verdict so the enabled-but-deterministic
    path runs byte-identically to the pre-5E path (no repair). Same
    (verdict, call_meta) protocol as LLMAnswerReviewer. Never calls
    any model; never generates answer text by construction.
    """

    def review(self, question, answer, evidence, *, workflow=None,
               available_files=None, session_id=None, plan_meta=None):
        from document_scope import is_valid_session_id

        t0 = time.monotonic()
        meta = {"provider": "deterministic", "model": "none",
                "latency_ms": 0, "failure": None,
                "tokens_in": None, "tokens_out": None}
        if session_id is not None and not is_valid_session_id(session_id):
            meta = dict(meta)
            meta.update(failure="schema",
                        latency_ms=_ms(t0, time.monotonic()))
            return None, meta
        verdict = {"schema_version": REVIEW_SCHEMA_VERSION,
                   "verdict": "pass",
                   "missing_aspects": [],
                   "unsupported_claims": [],
                   "suggested_followup_queries": []}
        return verdict, meta


class LLMAnswerReviewer:
    """Proposes structured verdicts via a reviewer model. Never answers.

    structured_fn(prompt_text, schema_dict) -> dict is injected
    (Bedrock adapter above in production, fakes in tests). Output is
    ALWAYS untrusted: callers must run validate_verdict before use.
    Has no answer-generation methods by construction.
    """

    REVIEW_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer"},
            "verdict": {"type": "string"},
            "missing_aspects": {"type": "array",
                                "items": {"type": "string"}},
            "unsupported_claims": {"type": "array",
                                   "items": {"type": "string"}},
            "suggested_followup_queries": {"type": "array",
                                           "items": {"type": "string"}},
        },
        "required": sorted(REVIEW_KEYS),
    }

    def __init__(self, structured_fn, *, timeout_s=5, provider="?",
                 model="?"):
        if not callable(structured_fn):
            raise ValueError("LLMAnswerReviewer needs a structured_fn.")
        try:
            timeout = float(timeout_s)
        except (TypeError, ValueError):
            raise ValueError("timeout_s must be numeric.")
        if timeout <= 0:
            raise ValueError("timeout_s must be positive.")
        self._fn = structured_fn
        self.timeout_s = timeout
        self.provider = provider or "?"
        self.model = model or "?"

    def review(self, question, answer, evidence, *, workflow=None,
               available_files=None, session_id=None, plan_meta=None):
        """Return (raw_dict|None, meta). Raw output is UNVALIDATED."""
        from document_scope import is_valid_session_id

        t0 = time.monotonic()
        meta = {"provider": self.provider, "model": self.model,
                "latency_ms": None, "failure": None,
                "tokens_in": None, "tokens_out": None}
        if session_id is not None and not is_valid_session_id(session_id):
            meta["failure"] = "schema"
            meta["latency_ms"] = _ms(t0, time.monotonic())
            return None, meta
        prompt = build_reviewer_input(
            question, answer, evidence, workflow=workflow,
            available_files=available_files, plan_meta=plan_meta,
            session_bound=session_id is not None)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._fn, prompt,
                                     self.REVIEW_JSON_SCHEMA)
                raw = future.result(timeout=self.timeout_s)
        except TimeoutError:
            meta["failure"] = "timeout"
            meta["latency_ms"] = _ms(t0, time.monotonic())
            return None, meta
        except Exception as e:
            meta["failure"] = _classify_reviewer_error(e)
            meta["latency_ms"] = _ms(t0, time.monotonic())
            return None, meta
        meta["latency_ms"] = _ms(t0, time.monotonic())
        tokens = _usage_of(raw)
        if tokens is not None:
            meta["tokens_in"], meta["tokens_out"] = tokens
        if not isinstance(raw, dict):
            meta["failure"] = "malformed"
            return None, meta
        return raw, meta


def _usage_of(raw):
    return None


def _classify_reviewer_error(err):
    name = type(err).__name__
    text = (name + " " + str(err)[:200]).lower()
    if "noregion" in text or "nocredentials" in name.lower() \
            or "partialcredentials" in name.lower() \
            or "no credentials" in text:
        return "credentials"
    if "timeout" in text or isinstance(err, TimeoutError):
        return "timeout"
    if "langchain-aws" in text or "boto3" in text \
            or "not installed" in text:
        return "unsupported"
    if "validation" in name.lower() or "not found" in text \
            or "unknown model" in text or "model" in name.lower():
        return "model_error"
    if "throttl" in text:
        return "unavailable"
    return "unavailable"


def _ms(a, b):
    try:
        return max(0, int((float(b) - float(a)) * 1000))
    except (TypeError, ValueError):
        return 0


# ---------- invocation policy (deterministic) ---------- #

def should_review(guard, workflow, *, review_always=False):
    """Decide whether the reviewer may run (no model call here).

    Returns (run: bool, trigger: str). Priority is deterministic:
    citation_guard > comparison > summary > always > none. System
    answers without evidence are never reviewed (callers check that
    first); this gate only sees the guard dict and workflow name.
    """
    count = 0
    try:
        count = int((guard or {}).get("count") or 0)
    except (TypeError, ValueError):
        count = 0
    if count > 0:
        return True, "citation_guard"
    if (workflow or "normal") == "comparison":
        return True, "comparison"
    if (workflow or "normal") == "summary":
        return True, "summary"
    if review_always:
        return True, "always"
    return False, "none"


def _is_system_answer(answer):
    text = (answer or "").strip()
    for prefix in ("I could not find", "I don't have enough",
                   "The model returned", "The knowledge base",
                   "An error occurred", "Quick check before I answer:",
                   "To proceed, I need"):
        if text.startswith(prefix):
            return True
    return False


def _empty_review(trigger="none", failure=None):
    return {"review_used": False, "review_provider": None,
            "review_model": None, "review_latency_ms": None,
            "review_verdict": None, "review_failure_category": failure,
            "review_trigger": trigger, "repair_attempted": False,
            "repair_succeeded": False, "review_tokens_in": None,
            "review_tokens_out": None, "review_cost_usd": None,
            "guard_before_flagged": None, "guard_before_total": None,
            "guard_after_flagged": None, "guard_after_total": None}


# ---------- configuration / bundling ---------- #

def review_enabled(config=None):
    if config is None:
        return False
    return str(getattr(config, "review_enabled", "0") or "0").lower() in (
        "1", "true", "yes", "on")


def review_always_enabled(config=None):
    if config is None:
        return False
    return str(getattr(config, "review_always", "0") or "0").lower() in (
        "1", "true", "yes", "on")


def build_llm_reviewer(config=None):
    """Build the LLM reviewer from config (loud on misconfiguration)."""
    from bedrock_provider import BedrockConverseProvider

    cfg = config
    if cfg is None:
        from config import get_config
        cfg = get_config()
    provider = (getattr(cfg, "review_provider", "") or "").strip().lower()
    if provider != "bedrock":
        raise ValueError(
            "REVIEW_PROVIDER supports only 'bedrock' in Phase 5E; "
            "got %r." % (provider,))
    model_id = (getattr(cfg, "review_model_id", "") or "").strip()
    if not model_id:
        raise ValueError("REVIEW_MODEL_ID is required when review "
                         "is enabled; no model is ever chosen silently.")
    try:
        timeout = float(getattr(cfg, "review_timeout_s", 5) or 5)
    except (TypeError, ValueError):
        raise ValueError("REVIEW_TIMEOUT_S must be numeric.")
    chat_model = BedrockConverseProvider(
        model_id=model_id,
        region=(getattr(cfg, "review_region", "us-east-1")
                or "us-east-1"))._chat_model()
    structured = bedrock_reviewer_structured_fn(chat_model)
    return LLMAnswerReviewer(structured, timeout_s=timeout,
                             provider="bedrock", model=model_id)


def get_reviewer_bundle(config=None):
    """Bundle for reviewed execution, or None when review is disabled.

    Disabled (default) returns None WITHOUT building anything: no
    provider objects, no network, deterministic path untouched.
    """
    if not review_enabled(config):
        return None
    reviewer = build_llm_reviewer(config)
    return {"reviewer": reviewer,
            "provider": reviewer.provider, "model": reviewer.model,
            "timeout_s": reviewer.timeout_s,
            "_meta_base": {"review_used": True,
                           "review_provider": reviewer.provider,
                           "review_model": reviewer.model}}


def get_session_reviewer_bundle(state, config=None):
    """Session-cached bundle (rebuilds when fingerprint changes)."""
    bundle = get_reviewer_bundle(config)
    if bundle is None:
        if isinstance(state, dict):
            state.pop("_reviewer_bundle", None)
        return None
    fingerprint = (bundle["provider"], bundle["model"],
                   bundle["timeout_s"])
    if isinstance(state, dict):
        cached = state.get("_reviewer_bundle")
        if isinstance(cached, dict) and \
                cached.get("fingerprint") == fingerprint:
            return cached["bundle"]
        state["_reviewer_bundle"] = {"fingerprint": fingerprint,
                                     "bundle": bundle}
    return bundle


# ---------- deterministic evidence merge ---------- #

def merge_evidence(original, new_lists):
    """Union original + gap-fill hits by chunk_id (max score wins).

    Sorted by score descending (None scores last), stable for ties.
    No re-scoring, no re-chunking, no scope change: inputs already
    carry provider-side filtering.
    """
    best = {}
    order = []
    for s in (original or []):
        cid = (s or {}).get("chunk_id") or id(s)
        if cid not in best:
            best[cid] = s
            order.append(cid)
    for hits in (new_lists or []):
        for s in (hits or []):
            cid = (s or {}).get("chunk_id") or id(s)
            if cid not in best:
                best[cid] = s
                order.append(cid)
                continue
            try:
                old_score = best[cid].get("score")
                new_score = s.get("score")
                old_f = float(old_score) if old_score is not None else None
                new_f = float(new_score) if new_score is not None else None
            except (TypeError, ValueError):
                continue
            if new_f is not None and (old_f is None or new_f > old_f):
                best[cid] = s

    def _sort_key(cid):
        s = best[cid] or {}
        v = s.get("score")
        try:
            f = float(v) if v is not None else None
        except (TypeError, ValueError):
            f = None
        # None scores last; stable for ties via original order.
        return (f is not None, f if f is not None else 0.0)

    ranked = sorted(order, key=_sort_key, reverse=True)
    return [best[cid] for cid in ranked]


# ---------- reviewed execution (existing runners authoritative) ---------- #

def _resolve_store(config, vector_store):
    if vector_store is not None:
        return vector_store
    from embeddings import get_embedding_provider
    from vector_store import get_vector_store

    cfg = config
    if cfg is None:
        from config import get_config
        cfg = get_config()
    return get_vector_store(cfg, get_embedding_provider(cfg))


def _available_files(vector_store, session_id):
    from workflows import known_file_names
    try:
        return [n for n in known_file_names(vector_store,
                                            session_id=session_id) if n]
    except Exception as e:
        logger.warning("Reviewer file listing failed: %s", e)
        return []


def _effective_of(query_text, conversation_context):
    from answer_support import resolve_effective_question
    try:
        effective, _clarified = resolve_effective_question(
            query_text, conversation_context)
    except Exception:
        effective = query_text
    return effective or query_text


def _guard_of(answer, retrieved):
    from answer_support import citation_guard
    try:
        return citation_guard(answer or "", retrieved or [])
    except Exception as e:
        logger.warning("Reviewer guard computation failed: %s", e)
        return {"flagged": [], "count": 0, "total": 0}


def _run_gap_fill(valid_queries, *, config, vector_store, session_id):
    """One deterministic retrieval pass per validated query.

    Uses the SAME backend.retrieve_documents mechanism (same k,
    threshold, scope). Returns (new_lists, succeeded_any).
    """
    from backend import retrieve_documents

    new_lists = []
    succeeded = False
    for q in valid_queries:
        try:
            if session_id is None:
                hits = retrieve_documents(q, config=config,
                                          vector_store=vector_store)
            else:
                hits = retrieve_documents(q, config=config,
                                          vector_store=vector_store,
                                          session_id=session_id)
        except Exception as e:
            logger.warning("Reviewer gap-fill retrieval failed: %s", e)
            continue
        succeeded = True
        new_lists.append(hits or [])
    return new_lists, succeeded


def _regenerate_answer(query_text, effective, merged, *, workflow,
                       config, vector_store, llm_provider,
                       conversation_context):
    """Regenerate via the EXISTING answer path (evidence set only).

    Prompts, temperature, provider, sanitizer, and workflow semantics
    are unchanged; only the deterministic evidence set differs.
    Returns (answer|None, failure|None). None answer means preserve
    the original (regeneration unavailable).
    """
    if not merged:
        return None, "empty-evidence"
    try:
        if workflow == "comparison":
            return _regenerate_comparison(
                query_text, merged, config, llm_provider), None
        if workflow == "summary":
            return _regenerate_summary(
                query_text, merged, config, llm_provider), None
        if workflow == "extraction":
            return _regenerate_extraction(
                query_text, merged), None
        return _regenerate_normal(
            effective, merged, config, llm_provider), None
    except Exception as e:
        logger.warning("Reviewer regeneration failed: %s", e)
        return None, "regeneration"


def _regenerate_normal(effective, merged, config, llm_provider):
    from answer_support import DIRECT, assess_support, detect_broad_scope
    from backend import (OVERVIEW_SUPPORT, PARTIAL_SUPPORT,
                         generate_answer)

    support = assess_support(effective or "", merged)
    broad, _target = detect_broad_scope(effective or "")
    if broad:
        level = OVERVIEW_SUPPORT
    else:
        level = None if support.get("level") == DIRECT else PARTIAL_SUPPORT
    return generate_answer(effective or "", merged, config=config,
                           llm_provider=llm_provider,
                           support_level=level)


def _regenerate_comparison(query_text, merged, config, llm_provider):
    from llm_provider import get_llm_provider
    from output_safety import is_empty_response, strip_think_blocks
    from workflows import COMPARISON_PROMPT_TEMPLATE, _evidence_block

    seen = []
    for s in merged or []:
        fname = (s or {}).get("file_name")
        if fname and fname not in seen:
            seen.append(fname)
    targets = seen[:2] if len(seen) >= 2 else seen
    if not targets:
        raise ValueError("no comparison targets in merged evidence")
    blocks = []
    for t in sorted(targets):
        part = [s for s in merged
                if ((s or {}).get("file_name") or "").lower()
                == (t or "").lower()]
        blocks.append(_evidence_block(t, part))
    # New evidence from files outside the original pair joins as its
    # own blocks (deterministic, sorted) rather than being dropped.
    extra_files = sorted({(s or {}).get("file_name") for s in merged
                          if (s or {}).get("file_name") not in targets})
    for t in extra_files:
        if not t:
            continue
        part = [s for s in merged if (s or {}).get("file_name") == t]
        blocks.append(_evidence_block(t, part))
    context = "\n\n".join(blocks)
    provider = llm_provider
    if provider is None:
        cfg = config
        if cfg is None:
            from config import get_config
            cfg = get_config()
        provider = get_llm_provider(cfg)
    raw = provider.generate(context, query_text,
                            COMPARISON_PROMPT_TEMPLATE)
    answer = strip_think_blocks(raw)
    if is_empty_response(raw):
        from backend import EMPTY_RESPONSE_MESSAGE
        answer = EMPTY_RESPONSE_MESSAGE
    return answer


def _regenerate_summary(query_text, merged, config, llm_provider):
    from llm_provider import get_llm_provider
    from output_safety import is_empty_response, strip_think_blocks
    from workflows import (MAX_SUMMARY_CHUNKS, SUMMARY_PROMPT_TEMPLATE,
                           _evidence_block)

    counts = {}
    order = []
    for s in merged or []:
        fname = (s or {}).get("file_name") or "unknown"
        counts[fname] = counts.get(fname, 0) + 1
        if fname not in order:
            order.append(fname)
    target = None
    best = -1
    for fname in order:
        if counts.get(fname, 0) > best:
            best = counts[fname]
            target = fname
    if not target:
        raise ValueError("no summary target in merged evidence")
    filtered = [s for s in merged
                if ((s or {}).get("file_name") or "unknown") == target]
    filtered = filtered[:MAX_SUMMARY_CHUNKS]
    if not filtered:
        raise ValueError("no summary evidence after merge")
    context = _evidence_block(target, filtered)
    provider = llm_provider
    if provider is None:
        cfg = config
        if cfg is None:
            from config import get_config
            cfg = get_config()
        provider = get_llm_provider(cfg)
    raw = provider.generate(context, query_text, SUMMARY_PROMPT_TEMPLATE)
    answer = strip_think_blocks(raw)
    if is_empty_response(raw):
        from backend import EMPTY_RESPONSE_MESSAGE
        answer = EMPTY_RESPONSE_MESSAGE
    return answer


def _regenerate_extraction(query_text, merged):
    from workflows import (build_extraction_table, detect_workflow,
                           harvest_extraction_rows)

    detected = detect_workflow(query_text or "")
    fields = list(detected.get("fields") or [])
    if not fields:
        raise ValueError("no extraction fields for regeneration")
    rows, missing = harvest_extraction_rows(merged, fields)
    if not rows:
        raise ValueError("no extraction rows after merge")
    return build_extraction_table(rows, missing)


def _review_meta_base(bundle):
    base = {"review_used": False, "review_provider": None,
            "review_model": None}
    if isinstance(bundle, dict):
        base.update({"review_provider": bundle.get("provider"),
                     "review_model": bundle.get("model")})
    return base


def _execute_reviewed(query_text, *, initial, config, vector_store,
                      llm_provider, conversation_context, session_id,
                      reviewer_bundle, workflow):
    """Shared review/repair flow for planned + streaming entries.

    initial is (answer, sources, info, reasoning_meta). Returns
    (final_answer, final_sources, final_info, reasoning_meta,
    review_meta). At most ONE gap-fill retrieval + ONE regeneration;
    never loops. Failures preserve the original answer.
    """
    answer, sources, info, reasoning = initial
    guard_before = _guard_of(answer, sources)
    meta_base = _review_meta_base(reviewer_bundle)
    meta = dict(meta_base)
    meta.update(review_latency_ms=None, review_verdict=None,
                review_failure_category=None,
                review_trigger="none",
                repair_attempted=False, repair_succeeded=False,
                review_tokens_in=None, review_tokens_out=None,
                review_cost_usd=None,
                guard_before_flagged=guard_before.get("count"),
                guard_before_total=guard_before.get("total"),
                guard_after_flagged=guard_before.get("count"),
                guard_after_total=guard_before.get("total"))
    # System answers (refusals/clarifications/errors) and empty
    # evidence are never reviewed: no generation to ground.
    if not sources or _is_system_answer(answer):
        return answer, sources, info, reasoning, meta
    always = review_always_enabled(config)
    run, trigger = should_review(guard_before, workflow,
                                 review_always=always)
    meta.update(review_trigger=trigger)
    if reviewer_bundle is None or reviewer_bundle.get("reviewer") is None:
        meta.update(review_failure_category="disabled")
        return answer, sources, info, reasoning, meta
    if not run:
        return answer, sources, info, reasoning, meta
    reviewer = reviewer_bundle["reviewer"]
    vs = _resolve_store(config, vector_store)
    files = _available_files(vs, session_id)
    effective = _effective_of(query_text, conversation_context)
    plan_meta = reasoning if isinstance(reasoning, dict) else None
    raw, call_meta = reviewer.review(
        effective, answer, sources, workflow=workflow,
        available_files=files, session_id=session_id,
        plan_meta=plan_meta)
    meta.update(review_provider=call_meta.get("provider"),
                review_model=call_meta.get("model"),
                review_latency_ms=call_meta.get("latency_ms"),
                review_tokens_in=call_meta.get("tokens_in"),
                review_tokens_out=call_meta.get("tokens_out"))
    meta.update(review_used=True)
    if raw is None:
        meta.update(
            review_failure_category=call_meta.get("failure")
            or "unavailable")
        return answer, sources, info, reasoning, meta
    verdict, failure = validate_verdict(raw)
    if failure is not None or verdict is None:
        meta.update(review_failure_category=call_meta.get("failure")
                    or failure or "schema")
        return answer, sources, info, reasoning, meta
    meta.update(review_verdict=verdict.get("verdict"))
    if verdict.get("verdict") != "repair":
        # pass / insufficient: preserve existing behavior, no search.
        return answer, sources, info, reasoning, meta
    valid_queries, _rejected = validate_followup_queries(
        verdict.get("suggested_followup_queries") or [],
        available_files=files)
    if not valid_queries:
        meta.update(repair_attempted=False, repair_succeeded=False)
        return answer, sources, info, reasoning, meta
    meta.update(repair_attempted=True)
    new_lists, succeeded = _run_gap_fill(
        valid_queries, config=config, vector_store=vs,
        session_id=session_id)
    if not succeeded or not new_lists:
        meta.update(repair_succeeded=False)
        return answer, sources, info, reasoning, meta
    merged = merge_evidence(sources, new_lists)
    if not merged:
        meta.update(repair_succeeded=False)
        return answer, sources, info, reasoning, meta
    new_answer, regen_failure = _regenerate_answer(
        query_text, effective, merged, workflow=workflow, config=config,
        vector_store=vs, llm_provider=llm_provider,
        conversation_context=conversation_context)
    if new_answer is None:
        meta.update(repair_succeeded=False)
        return answer, sources, info, reasoning, meta
    guard_after = _guard_of(new_answer, merged)
    meta.update(repair_succeeded=True,
                guard_after_flagged=guard_after.get("count"),
                guard_after_total=guard_after.get("total"))
    new_info = dict(info or {})
    new_info["retrieved"] = merged
    return new_answer, merged, new_info, reasoning, meta


def query_reviewed(query_text, config=None, vector_store=None,
                   llm_provider=None, conversation_context=None,
                   known_files=None, on_phase=None, session_id=None,
                   planner_bundle=None, reviewer_bundle=None):
    """Reviewed non-streaming runner.

    Runs the existing planned path (planner_bundle may be None for the
    frozen path), then the conditional reviewer + at most ONE repair.
    Returns (answer, sources, info, reasoning_meta, review_meta).
    With reviewer_bundle None (or deterministic pass), results equal
    the planned path exactly. Metadata-only review record included.
    """
    from query_planner import query_planned

    cfg = config
    if cfg is None:
        from config import get_config
        cfg = get_config()
    answer, sources, info, reasoning = query_planned(
        query_text, config=cfg, vector_store=vector_store,
        llm_provider=llm_provider, conversation_context=conversation_context,
        known_files=known_files, on_phase=on_phase, session_id=session_id,
        bundle=planner_bundle)
    workflow = None
    try:
        workflow = (info or {}).get("workflow") or \
            (reasoning or {}).get("workflow") or "normal"
    except Exception:
        workflow = "normal"
    if workflow not in WORKFLOW_ALLOWLIST:
        workflow = "normal"
    return _execute_reviewed(
        query_text, initial=(answer, sources, info, reasoning),
        config=cfg, vector_store=vector_store, llm_provider=llm_provider,
        conversation_context=conversation_context, session_id=session_id,
        reviewer_bundle=reviewer_bundle, workflow=workflow)


def _one_shot(text):
    yield text


def stream_reviewed_answer(query_text, config=None, vector_store=None,
                           llm_provider=None, conversation_context=None,
                           known_files=None, on_phase=None, session_id=None,
                           planner_bundle=None, reviewer_bundle=None):
    """Reviewed streaming entry. Returns (info, stream, reasoning_meta,
    review_meta).

    With reviewer_bundle None the underlying planned stream is
    returned directly (no buffering, byte-identical). When review is
    enabled, the initial stream is assembled (sanitized by the
    existing path), reviewed, and -- only on validated repair --
    regenerated once via the existing generation path; the final
    stream yields the final answer text.
    """
    from query_planner import stream_planned_answer

    cfg = config
    if cfg is None:
        from config import get_config
        cfg = get_config()
    if reviewer_bundle is None or reviewer_bundle.get("reviewer") is None:
        info, stream, reasoning = stream_planned_answer(
            query_text, config=cfg, vector_store=vector_store,
            llm_provider=llm_provider, conversation_context=conversation_context,
            known_files=known_files, on_phase=on_phase, session_id=session_id,
            bundle=planner_bundle)
        # Disabled record without any reviewer call or import.
        meta = _empty_review(trigger="none", failure="disabled")
        if isinstance(reasoning, dict):
            workflow = reasoning.get("workflow") or \
                (info or {}).get("workflow") or "normal"
            if workflow in ("comparison", "summary"):
                # Even disabled, record which trigger WOULD have fired
                # (metadata only; no reviewer call was made).
                guard = _guard_of(None, [])
                _run, trig = should_review(guard, workflow,
                                           review_always=False)
                meta.update(review_trigger=trig)
        return info, stream, reasoning, meta
    # Enabled: buffer the planned stream, then review once.
    info, stream, reasoning = stream_planned_answer(
        query_text, config=cfg, vector_store=vector_store,
        llm_provider=llm_provider, conversation_context=conversation_context,
        known_files=known_files, on_phase=on_phase, session_id=session_id,
        bundle=planner_bundle)
    if info.get("answer") is not None or info.get("failed"):
        # Decided without generation: nothing to review.
        meta = _empty_review(trigger="none", failure=None)
        try:
            workflow = (info or {}).get("workflow") or \
                (reasoning or {}).get("workflow") or "normal"
        except Exception:
            workflow = "normal"
        if workflow not in WORKFLOW_ALLOWLIST:
            workflow = "normal"
        meta.update(review_trigger="none")
        return info, stream, reasoning, meta
    try:
        assembled = "".join(list(stream))
    except Exception:
        # Underlying stream failed: propagate for caller fallback;
        # no answer exists to review.
        def _raise():
            raise
            yield ""
        meta = _empty_review(trigger="none", failure=None)
        return info, _raise(), reasoning, meta
    workflow = None
    try:
        workflow = (info or {}).get("workflow") or \
            (reasoning or {}).get("workflow") or "normal"
    except Exception:
        workflow = "normal"
    if workflow not in WORKFLOW_ALLOWLIST:
        workflow = "normal"
    final_answer, final_sources, final_info, reasoning, meta = \
        _execute_reviewed(
            query_text,
            initial=(assembled, info.get("retrieved") or [], info,
                     reasoning),
            config=cfg, vector_store=vector_store,
            llm_provider=llm_provider,
            conversation_context=conversation_context,
            session_id=session_id, reviewer_bundle=reviewer_bundle,
            workflow=workflow)
    final_info = dict(final_info or {})
    # Refresh guard note inputs: final retrieved set is authoritative.
    return final_info, _one_shot(final_answer), reasoning, meta
