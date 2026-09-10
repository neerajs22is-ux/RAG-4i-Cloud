"""Query planner / reasoning LLM layer (Phase 5D).

Core rule: the models may become smarter; the evidence system does not
become less deterministic.

Pipeline position (additive only):

    detect_workflow (existing, authoritative for workflow type)
    → requires_reasoning_plan (deterministic escalation gate)
    → planner: LLMQueryPlanner | DeterministicPlanner (proposes ONLY a
       structured plan; never generates user-facing answers)
    → validate_plan (deterministic; planner output is UNTRUSTED)
    → existing workflow/retrieval execution (unchanged machinery)

Rules enforced here:
  - REASONING_ENABLED=0 (default): no planner object is built, no
    network call happens, behavior is byte-identical to 4.12.
  - Planner input is bounded: effective question, available filenames,
    session-presence (never the session id value), bounded memory
    window. Never transcripts, paths, secrets, or chunk contents.
  - A validated plan can only AUGMENT normal retrieval forms or trigger
    a deterministic clarification; it can never reroute the workflow,
    touch providers, alter scope, or render text.
  - Every failure (credentials, provider, timeout, malformed output,
    schema) falls back to the deterministic path with a recorded
    metadata-only failure category. Invalid plans are never executed.

Pure helpers wherever possible (fully unit-testable without Streamlit
or AWS). Provider-touching constructors fail loudly when explicitly
enabled but misconfigured.
"""

import logging
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

PLAN_SCHEMA_VERSION = 1

# Bounds (hard; validator rejects anything beyond these).
MAX_PLAN_DOCUMENTS = 4
MAX_PLAN_TOPICS = 6
MAX_TOPIC_CHARS = 80
MAX_RETRIEVAL_QUERIES = 4
MAX_QUERY_CHARS = 200
MAX_CLARIFICATION_CHARS = 300
MAX_AVAILABLE_FILES = 20
MAX_MEMORY_TURNS = 2
MAX_MEMORY_CHARS = 300
MAX_QUESTION_CHARS = 500
PLAN_CACHE_SIZE = 64

WORKFLOW_ALLOWLIST = frozenset({"normal", "comparison", "extraction",
                                "summary"})

# Curated comparison dimensions (closed set; nothing invented).
COMPARISON_DIMENSIONS = frozenset({"scope", "parties", "dates", "amounts",
                                   "obligations", "notice", "renewal",
                                   "termination"})

# Escalation reasons actually emitted by the v1 gate (closed set).
ESCALATION_REASONS = frozenset({"non-document", "ordinary", "compound",
                                "complex_extraction", "ambiguous_target",
                                "multi_form_retrieval"})

# Reserved, never emitted by the v1 gate (future use only).
RESERVED_ESCALATION_REASONS = frozenset({"comparison", "other"})

# Planner failure categories (closed set; metadata only).
FAILURE_CATEGORIES = frozenset({"credentials", "unavailable", "timeout",
                                "malformed", "schema", "workflow",
                                "unsupported", "model"})

PLAN_KEYS = frozenset({"schema_version", "workflow", "documents", "topics",
                       "retrieval_queries", "fields",
                       "comparison_dimensions", "needs_clarification"})


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _content_terms(text: str):
    stop = {"a", "an", "the", "is", "are", "was", "it", "that", "this",
            "to", "of", "in", "on", "for", "do", "does", "me", "more",
            "about", "too", "there", "s", "t", "and", "or", "please",
            "my", "you", "your", "i", "what", "how", "which", "with"}
    return [w for w in re.findall(r"[a-z']+", text or "") if w not in stop]


# ---------- escalation gate (deterministic, conservative) ---------- #

def requires_reasoning_plan(query_text, intent, *, available_files=None):
    """Decide structurally whether the reasoning planner may help.

    Returns (escalate: bool, reason: str). Ordinary questions stay
    deterministic; only explicit complex patterns escalate. Never calls
    any model. `intent` is a query_router intent string.
    """
    from query_router import DOCUMENT_FOLLOWUP, DOCUMENT_QUERY
    from workflows import FIELD_VOCAB, detect_workflow

    if intent not in (DOCUMENT_QUERY, DOCUMENT_FOLLOWUP):
        return False, "non-document"
    text = _normalize(query_text)
    detected = detect_workflow(query_text)
    mentioned = detected.get("mentioned") or []
    terms = _content_terms(text)
    fields = [f for f, variants in FIELD_VOCAB.items()
              if any(v in text for v in variants)][:4]

    if detected["workflow"] == "comparison":
        # Resolved comparisons execute deterministically; only ambiguous
        # targeting benefits from planning help.
        if len(mentioned) >= 2:
            return False, "ordinary"
        return True, "ambiguous_target"
    if detected["workflow"] == "extraction":
        if len(fields) >= 3:
            return True, "complex_extraction"
        return False, "ordinary"
    if detected["workflow"] == "summary":
        if len(mentioned) == 1:
            return False, "ordinary"
        return True, "ambiguous_target"
    if _is_summary_ask_without_target(text, mentioned):
        return True, "ambiguous_target"
    # Normal workflow from here on.
    if text.count("?") >= 2:
        return True, "compound"
    if _is_compound_ask(text, terms):
        return True, "compound"
    if intent == DOCUMENT_FOLLOWUP and (
            len(fields) >= 2 or len(terms) >= 5):
        return True, "multi_form_retrieval"
    if not mentioned and _needs_document_choice(text, available_files):
        return True, "ambiguous_target"
    return False, "ordinary"


_SUMMARY_VERBS = re.compile(r"\bsummar(ize|ise|y|ies|ized|ised)\b|\boverview\b",
                          re.IGNORECASE)

_COMPOUND_VERBS = {"list", "compare", "extract", "summarize", "explain",
                   "show", "tell", "describe", "find"}


def _is_summary_ask_without_target(text: str, mentioned) -> bool:
    """Summary verb but no named file ("summarize the document")."""
    return bool(mentioned == [] and _SUMMARY_VERBS.search(text))


def _is_compound_ask(text: str, terms) -> bool:
    """Two imperative clauses joined by 'and' ("list X and compare Y")."""
    if " and " not in text:
        return False
    if len(terms) < 6:
        return False
    parts = text.split(" and ")
    hits = sum(1 for p in parts
               if _content_terms(p) and
               (set(_content_terms(p)) & _COMPOUND_VERBS
                or p.strip().endswith("?")))
    return hits >= 2


def _needs_document_choice(text: str, available_files) -> bool:
    """Bare demonstrative reference ("this document", "that file")."""
    if available_files is None:
        return False
    return has_demonstrative_reference(text)


def has_demonstrative_reference(text: str) -> bool:
    return bool(re.search(
        r"\b(this|that|these|those)\b.{0,20}\b(document|file|contract|"
        r"lease|deed|agreement|policy|paper)\b", text or ""))


# ---------- plan validation (deterministic, strict) ---------- #

def validate_plan(plan, *, available_files, session_id=None):
    """Validate an UNTRUSTED planner proposal.

    Returns (canonical_plan|None, failure_category|None). Anything
    malformed, unbounded, invented, or out-of-allowlist is rejected —
    never repaired. `available_files` is the deterministic session-aware
    file list; unknown names/paths fail. session_id is accepted for
    signature symmetry but never embedded (plans carry no secrets).
    """
    from workflows import FIELD_VOCAB

    if not isinstance(plan, dict):
        return None, "schema"
    if set(plan.keys()) != PLAN_KEYS:
        return None, "schema"
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION or \
            not isinstance(plan.get("schema_version"), int):
        return None, "schema"
    workflow = plan.get("workflow")
    if workflow not in WORKFLOW_ALLOWLIST:
        return None, "workflow"
    documents = plan.get("documents")
    if not isinstance(documents, list) or len(documents) > MAX_PLAN_DOCUMENTS:
        return None, "documents"
    known = set(available_files or [])
    clean_docs = []
    for name in documents:
        if not isinstance(name, str):
            return None, "documents"
        name = name.strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            return None, "documents"
        if name not in known:
            return None, "documents"
        if name not in clean_docs:
            clean_docs.append(name)
    topics = plan.get("topics")
    if not isinstance(topics, list) or len(topics) > MAX_PLAN_TOPICS:
        return None, "schema"
    clean_topics = []
    for topic in topics:
        if not isinstance(topic, str):
            return None, "schema"
        topic = topic.strip()
        if not topic or len(topic) > MAX_TOPIC_CHARS:
            return None, "schema"
        if topic not in clean_topics:
            clean_topics.append(topic)
    queries = plan.get("retrieval_queries")
    if not isinstance(queries, list) or len(queries) > MAX_RETRIEVAL_QUERIES:
        return None, "schema"
    clean_queries = []
    for query in queries:
        if not isinstance(query, str):
            return None, "schema"
        query = query.strip()
        if len(query) > MAX_QUERY_CHARS:
            return None, "schema"
        if query and query not in clean_queries:
            clean_queries.append(query)
    fields = plan.get("fields")
    if not isinstance(fields, list):
        return None, "fields"
    allowed_fields = set(FIELD_VOCAB.keys())
    clean_fields = []
    for field in fields:
        if field not in allowed_fields:
            return None, "fields"
        if field not in clean_fields:
            clean_fields.append(field)
    dimensions = plan.get("comparison_dimensions")
    if not isinstance(dimensions, list):
        return None, "dimensions"
    clean_dimensions = []
    for dimension in dimensions:
        if dimension not in COMPARISON_DIMENSIONS:
            return None, "dimensions"
        if dimension not in clean_dimensions:
            clean_dimensions.append(dimension)
    clarification = plan.get("needs_clarification")
    if not isinstance(clarification, dict):
        return None, "clarification"
    if set(clarification.keys()) != {"required", "question"}:
        return None, "clarification"
    required = clarification.get("required")
    question = clarification.get("question")
    if not isinstance(required, bool) or not isinstance(question, str):
        return None, "clarification"
    question = question.strip()
    if not required:
        if question != "":
            return None, "clarification"
    else:
        if not question or len(question) > MAX_CLARIFICATION_CHARS:
            return None, "clarification"
        anchors = set(clean_docs) | {t.lower() for t in clean_topics}
        low = question.lower()
        if not any(a and a in low for a in anchors):
            return None, "clarification"
    return {"schema_version": PLAN_SCHEMA_VERSION,
            "workflow": workflow,
            "documents": clean_docs,
            "topics": clean_topics,
            "retrieval_queries": clean_queries,
            "fields": clean_fields,
            "comparison_dimensions": clean_dimensions,
            "needs_clarification": {"required": required,
                                    "question": question}}, None


# ---------- planners ---------- #

class DeterministicPlanner:
    """Wraps detect_workflow with zero semantic change.

    Produces a canonical plan whose retrieval_queries are EMPTY, so the
    executor augments nothing and the 4.12 path runs byte-identically.
    needs_clarification is always negative here: existing runners own
    ambiguity handling, exactly as today.

    Same (plan, call_meta) protocol as LLMQueryPlanner; call_meta here
    is a zero-latency deterministic record.
    """

    def plan(self, query_text, *, available_files=None, session_id=None,
             context=None):
        from workflows import detect_workflow

        detected = detect_workflow(query_text or "")
        mentioned = [m for m in (detected.get("mentioned") or [])
                     if not available_files or m in (available_files or [])]
        plan = {"schema_version": PLAN_SCHEMA_VERSION,
                "workflow": detected["workflow"],
                "documents": mentioned,
                "topics": _content_terms(query_text)[:MAX_PLAN_TOPICS],
                "retrieval_queries": [],
                "fields": list(detected.get("fields") or []),
                "comparison_dimensions": [],
                "needs_clarification": {"required": False, "question": ""}}
        meta = {"provider": "deterministic", "model": "none",
                "latency_ms": 0, "failure": None,
                "tokens_in": None, "tokens_out": None}
        return plan, meta


def build_planner_prompt(query_text, *, available_files=None,
                         session_bound=False, context=None):
    """Bounded planner input (metadata only — never evidence/paths)."""
    files = list(available_files or [])[:MAX_AVAILABLE_FILES]
    snippets = []
    for message in (context or [])[-MAX_MEMORY_TURNS:]:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = re.sub(r"\s+", " ", str(message.get("content") or "")).strip()
        if text:
            snippets.append(text[:MAX_MEMORY_CHARS])
    lines = [
        "Propose a retrieval plan as ONE JSON object. Never answer the "
        "question itself.",
        "Schema (exact keys): schema_version=1, workflow in "
        "[normal, comparison, extraction, summary], documents (known "
        "filenames only), topics, retrieval_queries (0-4 short queries), "
        "fields, comparison_dimensions, needs_clarification "
        "({required, question}).",
        f"Question: {(query_text or '').strip()[:MAX_QUESTION_CHARS]}",
        "Known documents: " + (", ".join(files) if files else "(none)"),
        "Session documents bound: " + ("yes" if session_bound else "no"),
    ]
    if snippets:
        lines.append("Recent user turns: " + " | ".join(snippets))
    lines.append("Return ONLY the JSON object.")
    return "\n".join(lines)


def supports_native_structured_output():
    """Whether the installed stack offers native Bedrock JSON-schema output.

    Status: langchain-aws is NOT installed in this environment, so native
    outputConfig support is UNVERIFIED here; the adapter below uses the
    documented forced-tool-calling path (with_structured_output), which
    works across versions. Re-verify per installed langchain-aws before
    adopting the native path.
    """
    return {"mode": "forced-tool-calling",
            "native_outputConfig": "unverified",
            "validator": "deterministic (always)"}


def bedrock_structured_fn(chat_model):
    """Adapt a chat model to structured_fn(prompt, schema) -> dict.

    Uses forced tool calling (with_structured_output); the result MUST
    be a dict or a ValueError marks it malformed downstream. No answer
    text is ever produced through this path.
    """
    def call(prompt_text, schema):
        runner = chat_model.with_structured_output(schema)
        result = runner.invoke(prompt_text)
        if not isinstance(result, dict):
            raise ValueError("Planner returned non-dict structured output.")
        return result

    return call


class LLMQueryPlanner:
    """Proposes structured plans via a reasoning model. Never answers.

    structured_fn(prompt_text, schema_dict) -> dict is injected (Bedrock
    adapter above in production, fakes in tests). Output is ALWAYS
    untrusted: callers must run validate_plan before use.
    """

    PLAN_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer"},
            "workflow": {"type": "string"},
            "documents": {"type": "array", "items": {"type": "string"}},
            "topics": {"type": "array", "items": {"type": "string"}},
            "retrieval_queries": {"type": "array",
                                  "items": {"type": "string"}},
            "fields": {"type": "array", "items": {"type": "string"}},
            "comparison_dimensions": {"type": "array",
                                      "items": {"type": "string"}},
            "needs_clarification": {"type": "object"},
        },
        "required": sorted(PLAN_KEYS),
    }

    def __init__(self, structured_fn, *, timeout_s=5, provider="?",
                 model="?"):
        if not callable(structured_fn):
            raise ValueError("LLMQueryPlanner needs a structured_fn.")
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

    def plan(self, query_text, *, available_files=None, session_id=None,
             context=None):
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
        prompt = build_planner_prompt(
            query_text, available_files=available_files,
            session_bound=session_id is not None, context=context)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._fn, prompt,
                                     self.PLAN_JSON_SCHEMA)
                raw = future.result(timeout=self.timeout_s)
        except TimeoutError:
            meta["failure"] = "timeout"
            meta["latency_ms"] = _ms(t0, time.monotonic())
            return None, meta
        except Exception as e:
            meta["failure"] = _classify_planner_error(e)
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


def _classify_planner_error(err) -> str:
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
        return "model"
    if "throttl" in text:
        return "unavailable"
    return "unavailable"


def _ms(a, b):
    try:
        return max(0, int((float(b) - float(a)) * 1000))
    except (TypeError, ValueError):
        return 0


# ---------- plan cache (validated plans only, bounded) ---------- #

class PlanCache:
    """Bounded LRU of validated plans. Keys bind question + documents +
    session + provider + model, so plans never cross incompatible
    contexts. Values are canonical plans (no contents, no secrets)."""

    def __init__(self, max_entries=PLAN_CACHE_SIZE):
        self._entries = OrderedDict()
        self._max = max(1, int(max_entries))
        self._lock = threading.Lock()

    @staticmethod
    def cache_key(question, available_files, session_id, provider, model):
        files = tuple(sorted(available_files or []))
        return (PLAN_SCHEMA_VERSION, _normalize(question), files,
                session_id or "", provider or "", model or "")

    def get(self, key):
        with self._lock:
            if key not in self._entries:
                return None
            self._entries.move_to_end(key)
            return self._entries[key]

    def put(self, key, plan):
        with self._lock:
            self._entries[key] = plan
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def __len__(self):
        with self._lock:
            return len(self._entries)


def _one_shot(text):
    yield text


def _empty_reasoning(escalation_reason="ordinary"):
    return {"reasoning_used": False, "reasoning_provider": None,
            "reasoning_model": None, "planner_latency_ms": None,
            "planner_failure_category": None,
            "escalation_reason": escalation_reason,
            "planner_cached": False, "planner_tokens_in": None,
            "planner_tokens_out": None, "reasoning_cost_usd": None,
            "workflow": None}


def _deterministic_fallback(bundle, query_text, *, available_files,
                            session_id, context):
    """Validated deterministic plan (always succeeds for sane inputs)."""
    planner = DeterministicPlanner()
    raw, _call_meta = planner.plan(query_text, available_files=available_files,
                                   session_id=session_id, context=context)
    plan, failure = validate_plan(raw, available_files=available_files,
                                  session_id=session_id)
    meta = dict(bundle.get("_meta_base") or {})
    if failure is not None or plan is None:
        meta.update(planner_failure_category="schema")
        return None, meta
    return plan, meta


def _obtain_plan(bundle, query_text, *, available_files, session_id,
                 context, escalate, cache_key):
    """Cache -> (LLM | deterministic) -> validate, with loud fallback."""
    cache = bundle.get("cache")
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            meta = dict(bundle.get("_meta_base") or {})
            meta.update(planner_cached=True, reasoning_used=True)
            return hit, meta
    meta = dict(bundle.get("_meta_base") or {})
    meta.update(planner_cached=False, reasoning_used=False)
    planner = bundle.get("planner")
    raw, call_meta = planner.plan(
        query_text, available_files=available_files, session_id=session_id,
        context=context) if escalate else (None, {"failure": None,
                                                  "latency_ms": 0,
                                                  "tokens_in": None,
                                                  "tokens_out": None})
    meta.update(planner_latency_ms=call_meta.get("latency_ms"),
                planner_tokens_in=call_meta.get("tokens_in"),
                planner_tokens_out=call_meta.get("tokens_out"))
    if escalate and raw is not None:
        meta.update(reasoning_used=True)
        plan, failure = validate_plan(raw, available_files=available_files,
                                      session_id=session_id)
        if failure is None and plan is not None:
            if cache is not None:
                cache.put(cache_key, plan)
            return plan, meta
        meta.update(planner_failure_category=call_meta.get("failure")
                    or failure or "schema")
    elif escalate:
        meta.update(reasoning_used=True,
                    planner_failure_category=call_meta.get("failure")
                    or "unavailable")
    plan, fallback_meta = _deterministic_fallback(
        bundle, query_text, available_files=available_files,
        session_id=session_id, context=context)
    if meta.get("planner_failure_category") is None:
        meta.update(planner_failure_category=fallback_meta.get(
            "planner_failure_category"))
    return plan, meta


def _clarification_text(plan, available_files):
    """Deterministic clarification body from validated plan data only.

    Uses the shared clarification marker so the existing confirmation
    flow ("yes" proceeds) keeps working unchanged.
    """
    from answer_support import CLARIFICATION_MARKER

    names = list(plan.get("documents") or [])[:3]
    if not names:
        names = list(available_files or [])[:3]
    topics = list(plan.get("topics") or [])[:3]
    bits = []
    if topics:
        bits.append("about " + ", ".join(topics))
    if names:
        bits.append("in " + ", ".join(names))
    detail = (" ".join(bits) + " ") if bits else ""
    return (f"{CLARIFICATION_MARKER} I need a bit more detail {detail}"
            f"please name the document(s) or topic you mean.")


# ---------- configuration / bundling ---------- #

def reasoning_enabled(config=None) -> bool:
    if config is None:
        return False
    return str(getattr(config, "reasoning_enabled", "0") or "0").lower() in (
        "1", "true", "yes", "on")


def build_llm_planner(config=None):
    """Build the LLM planner from config (loud on misconfiguration)."""
    from bedrock_provider import BedrockConverseProvider

    cfg = config
    if cfg is None:
        from config import get_config
        cfg = get_config()
    provider = (getattr(cfg, "reasoning_provider", "") or "").strip().lower()
    if provider != "bedrock":
        raise ValueError(
            "REASONING_PROVIDER supports only 'bedrock' in Phase 5D; "
            f"got {provider!r}.")
    model_id = (getattr(cfg, "reasoning_model_id", "") or "").strip()
    if not model_id:
        raise ValueError("REASONING_MODEL_ID is required when reasoning "
                         "is enabled; no model is ever chosen silently.")
    try:
        timeout = float(getattr(cfg, "reasoning_timeout_s", 5) or 5)
    except (TypeError, ValueError):
        raise ValueError("REASONING_TIMEOUT_S must be numeric.")
    chat_model = BedrockConverseProvider(
        model_id=model_id,
        region=(getattr(cfg, "reasoning_region", "us-east-1")
                or "us-east-1"))._chat_model()
    structured = bedrock_structured_fn(chat_model)
    return LLMQueryPlanner(structured, timeout_s=timeout,
                           provider="bedrock", model=model_id)


def bedrock_structured_fn(chat_model):
    """Forced-tool structured adapter (verified path; see
    supports_native_structured_output). Returns
    structured_fn(prompt, schema) -> dict. Never produces answer text."""

    def call(prompt_text, schema):
        runner = chat_model.with_structured_output(schema)
        result = runner.invoke(prompt_text)
        if not isinstance(result, dict):
            raise ValueError(
                "Planner returned non-dict structured output.")
        return result

    return call


def get_planner_bundle(config=None, cache=None):
    """Bundle for planned execution, or None when reasoning is disabled.

    Disabled (default) returns None WITHOUT building anything: no
    provider objects, no network, deterministic path untouched.
    """
    if not reasoning_enabled(config):
        return None
    planner = build_llm_planner(config)
    return {"planner": planner,
            "cache": cache if cache is not None else PlanCache(),
            "provider": planner.provider, "model": planner.model,
            "timeout_s": planner.timeout_s,
            "_meta_base": {"reasoning_used": True,
                           "reasoning_provider": planner.provider,
                           "reasoning_model": planner.model}}


def get_session_planner_bundle(state, config=None):
    """Session-cached bundle (rebuilds when fingerprint changes)."""
    bundle = get_planner_bundle(
        config, cache=(state.get("_planner_cache")
                       if isinstance(state, dict) else None))
    if bundle is None:
        if isinstance(state, dict):
            state.pop("_planner_bundle", None)
        return None
    fingerprint = (bundle["provider"], bundle["model"],
                   bundle["timeout_s"])
    if isinstance(state, dict):
        cached = state.get("_planner_bundle")
        if isinstance(cached, dict) and \
                cached.get("fingerprint") == fingerprint:
            return cached["bundle"]
        if not isinstance(state.get("_planner_cache"), PlanCache):
            state["_planner_cache"] = PlanCache()
        bundle["cache"] = state["_planner_cache"]
        state["_planner_bundle"] = {"fingerprint": fingerprint,
                                    "bundle": bundle}
    return bundle


# ---------- planned execution (existing runners stay authoritative) ---------- #

def _intent_of(query_text, conversation_context):
    from query_router import observe_query
    try:
        return observe_query(query_text, conversation_context).intent
    except Exception:
        return "document_query"


def _available_files(vector_store, session_id):
    """Session-aware file list via the existing workflows helper."""
    from workflows import known_file_names
    try:
        return [n for n in known_file_names(vector_store,
                                            session_id=session_id) if n]
    except Exception as e:
        logger.warning("Planner file listing failed: %s", e)
        return []


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


def _plan_inputs(query_text, config, vector_store, conversation_context,
                 session_id):
    from conversation_memory import coerce_context
    from workflows import detect_workflow

    vs = _resolve_store(config, vector_store)
    detected = detect_workflow(query_text or "")
    intent = _intent_of(query_text, conversation_context)
    files = _available_files(vs, session_id)
    context = coerce_context(conversation_context)[-MAX_MEMORY_TURNS * 2:]
    return vs, detected, intent, files, context


def _execute_plan(query_text, *, bundle, detected, intent, files, context,
                   session_id, vs, base_call):
    """Shared plan/fetch/validate/fallback flow. base_call(extra_forms)
    runs the existing runners; returns (result..., reasoning_meta)."""
    escalate, reason = requires_reasoning_plan(
        query_text, intent, available_files=files)
    meta = dict((bundle or {}).get("_meta_base") or {})
    meta.update(escalation_reason=reason, workflow=detected["workflow"])
    if bundle is None or bundle.get("planner") is None:
        meta.update(_empty_reasoning(reason))
        meta.update(workflow=detected["workflow"])
        return base_call([]), meta
    key = PlanCache.cache_key(query_text, files, session_id,
                              bundle.get("provider"), bundle.get("model"))
    plan, plan_meta = _obtain_plan(
        bundle, query_text, available_files=files, session_id=session_id,
        context=context, escalate=escalate, cache_key=key)
    meta.update(plan_meta)
    meta.update(workflow=detected["workflow"])
    if plan is None:
        return base_call([]), meta
    if plan.get("workflow") != detected["workflow"]:
        # Planner must never reroute: discard, count, deterministic path.
        if meta.get("planner_failure_category") is None:
            meta.update(planner_failure_category="workflow")
        return base_call([]), meta
    if plan.get("needs_clarification", {}).get("required"):
        return ("clarify", _clarification_text(plan, files)), meta
    if detected["workflow"] == "normal":
        return base_call(list(plan.get("retrieval_queries") or [])), meta
    return base_call([]), meta


def query_planned(query_text, config=None, vector_store=None,
                  llm_provider=None, conversation_context=None,
                  known_files=None, on_phase=None, session_id=None,
                  bundle=None):
    """Planned non-streaming runner. Returns (answer, sources, info, meta).

    With bundle None (or deterministic no-op plans), results equal
    query_workflow exactly. Metadata-only reasoning record included.
    """
    from workflows import query_workflow

    cfg = config
    if cfg is None:
        from config import get_config
        cfg = get_config()
    vs, detected, intent, files, context = _plan_inputs(
        query_text, cfg, vector_store, conversation_context, session_id)

    def base_call(extra_forms):
        return query_workflow(
            query_text, config=cfg, vector_store=vs,
            llm_provider=llm_provider, conversation_context=context,
            known_files=known_files, on_phase=on_phase,
            session_id=session_id, extra_forms=extra_forms)

    outcome, meta = _execute_plan(
        query_text, bundle=bundle, detected=detected, intent=intent,
        files=files, context=context, session_id=session_id, vs=vs,
        base_call=base_call)
    if isinstance(outcome, tuple) and outcome and outcome[0] == "clarify":
        answer, info = _clarification_bundle(
            query_text, outcome[1], detected["workflow"])
        return answer, [], info, meta
    answer, sources, info = outcome
    return answer, sources, info, meta


def stream_planned_answer(query_text, config=None, vector_store=None,
                          llm_provider=None, conversation_context=None,
                          known_files=None, on_phase=None, session_id=None,
                          bundle=None):
    """Planned streaming entry. Returns (info, stream, meta)."""
    from workflows import stream_workflow_answer

    cfg = config
    if cfg is None:
        from config import get_config
        cfg = get_config()
    vs, detected, intent, files, context = _plan_inputs(
        query_text, cfg, vector_store, conversation_context, session_id)

    def base_call(extra_forms):
        return stream_workflow_answer(
            query_text, config=cfg, vector_store=vs,
            llm_provider=llm_provider, conversation_context=context,
            known_files=known_files, on_phase=on_phase,
            session_id=session_id, extra_forms=extra_forms)

    outcome, meta = _execute_plan(
        query_text, bundle=bundle, detected=detected, intent=intent,
        files=files, context=context, session_id=session_id, vs=vs,
        base_call=base_call)
    if isinstance(outcome, tuple) and outcome and outcome[0] == "clarify":
        _answer, info = _clarification_bundle(
            query_text, outcome[1], detected["workflow"])
        return info, _one_shot(_answer), meta
    info, stream = outcome
    return info, stream, meta


def _one_shot(text):
    yield text


def _clarification_timings():
    return {"retrieval_ms": 0, "support_ms": 0, "preparation_ms": 0,
            "generation_ms": 0}


def _clarification_bundle(query_text, text, workflow):
    """Deterministic clarification (template text only — never plan text)."""
    from backend import LOW_RELEVANCE_MESSAGE
    from output_safety import is_empty_response

    answer = text.strip() or LOW_RELEVANCE_MESSAGE
    if is_empty_response(answer):
        answer = LOW_RELEVANCE_MESSAGE
    info = {"answer": answer, "retrieved": [], "needs_retrieval": True,
            "support_level": None, "failed": False,
            "timings": _clarification_timings(),
            "workflow": workflow, "label": "Not enough context",
            "fallback_template": None, "fallback_question": query_text}
    return answer, info
