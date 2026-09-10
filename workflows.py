"""Advanced legal workflows: comparison, extraction, summaries (Phase 4.12).

Additive layer over the frozen RAG core. Normal questions never enter
this module's paths: `stream_workflow_answer` delegates them to
`backend.stream_answer` unchanged (same evidence, prompts, sanitizer).

Design rules (see RAG_INVARIANTS.md):
  - Detection is deterministic/structural (no LLM classification).
  - Retrieval reuses `backend.retrieve_documents` (query forms, k=5,
    threshold 0.3) plus `chunks_for_source` top-up via the provider
    abstraction (Chroma/pgvector parity preserved).
  - Generation reuses the shared LLM provider (temperature 0.0 from
    config) and ThinkStreamSanitizer; existing PROMPT_TEMPLATE /
    PARTIAL_PROMPT_TEMPLATE are never touched (new dedicated prompts
    below are the only addition).
  - Every claim/value stays traceable to retrieved evidence; the
    existing citation guard applies to generated text.
  - Sequential calls only; bounded chunk budgets (t3.micro safe).
"""

import logging
import re
import time

logger = logging.getLogger(__name__)

# Workflow types (internal intents; "normal" = existing path).
NORMAL = "normal"
COMPARISON = "comparison"
EXTRACTION = "extraction"
SUMMARY = "summary"

# Display labels for workflow answers (telemetry/feedback compatible).
COMPARISON_LABEL = "Comparison answer"
PARTIAL_COMPARISON_LABEL = "Partial comparison"
EXTRACTION_LABEL = "Structured answer"
SUMMARY_LABEL = "Document summary"

_FILENAME_RE = re.compile(r"\b[\w][\w\-]*\.pdf\b", re.IGNORECASE)

# --- deterministic cue sets (structural, not phrase lists) ---
_COMPARISON_CUES = (
    r"\bcompar(e|es|ed|ing|ison)\b",
    r"\bdiffer(ence|ences|ent|ently|s)?\b",
    r"\bversus\b", r"\b\vs\.?\b", r"\bcontrast\b",
    r"\bsame\b", r"\bdifferent\b",
)
_PLURAL_DOC_RE = re.compile(
    r"\b(these|both|the two|two)\b.{0,30}\b(documents?|contracts?|leases?|"
    r"deeds?|agreements?|files?|policies?)\b"
    r"|\bboth\b|\bthese two\b", re.IGNORECASE)
_SUMMARY_VERBS = re.compile(r"\bsummar(ize|ise|y|ies|ized|ised)\b|\boverview\b",
                            re.IGNORECASE)
_EXTRACTION_VERBS = re.compile(
    r"\b(list|extract|tabulat\w*|enumerat\w*|show|give|find)\b", re.IGNORECASE)

# Small curated legal field vocabulary (canonical -> variants).
FIELD_VOCAB = {
    "dates": ("date", "dates", "dated"),
    "deadlines": ("deadline", "deadlines", "due"),
    "renewal periods": ("renewal", "renew", "renews"),
    "notice periods": ("notice", "notices"),
    "payment terms": ("payment", "payments", "payable"),
    "amounts": ("amount", "amounts", "rs", "inr", "rupees", "price"),
    "deposits": ("deposit", "deposits"),
    "parties": ("party", "parties"),
    "obligations": ("obligation", "obligations"),
    "termination terms": ("termination", "terminate"),
    "lock-in terms": ("lock-in", "lock in", "lockin"),
}
# Topic words licensing a bare "same/differ" comparison against the only
# two indexed documents (never invented identities).
_COMPARISON_TOPIC_CUES = {
    "notice", "period", "periods", "payment", "payments", "lock-in",
    "lock", "rent", "deposit", "term", "terms", "renewal", "termination",
    "clause", "clauses", "date", "dates", "deadline", "obligation",
    "lease", "contract", "agreement", "deed",
}

CLARIFICATION_MARKER = "To proceed, I need"

# Bounded budgets (t3.micro / Qwen pilot safe).
MAX_SUMMARY_CHUNKS = 24
SUMMARY_GROUP_SIZE = 6
SUMMARY_CHUNK_CHARS = 1200
MAX_EXTRACTION_SUBQUERIES = 4
MAX_EXTRACTION_ROWS = 20

# --- NEW dedicated prompts (existing templates untouched) --- #

COMPARISON_PROMPT_TEMPLATE = """
You are an expert legal assistant for a Chartered Accountant firm.
Compare the documents using ONLY the evidence below. Each evidence block
is labelled with its source document.

Rules:
- Present a markdown table with one row per topic:
  | Topic | <first document> | <second document> |
  Use the exact document filenames from the evidence labels as headers.
- Then add a concise conclusion distinguishing similarities, differences,
  and anything missing or unclear.
- Every cell must be traceable to the evidence: quote it closely.
- If a document has no evidence for a topic, write "Not stated in the
  retrieved material" for that cell. Never invent the missing side.
- Never claim one document is "more favorable" or better unless the
  evidence explicitly establishes the basis.
- Never invent contract terms, dates, parties, or mechanics.

Evidence:
{context}

Question:
{question}
"""

SUMMARY_PART_PROMPT_TEMPLATE = """
You are an expert legal assistant for a Chartered Accountant firm.
Summarize ONLY the evidence below into short bullets of explicit terms:
parties, dates, amounts, obligations, notice and renewal provisions.

Rules:
- Bullets must quote the evidence closely; mark anything unclear as such.
- Never invent sections, clauses, dates, or parties.
- Never give legal advice or opinions.

Evidence:
{context}

Question:
{question}
"""

SUMMARY_PROMPT_TEMPLATE = """
You are an expert legal assistant for a Chartered Accountant firm.
Write an overview of the document using ONLY the evidence below.

Rules:
- Summarize explicit terms only: parties, dates, amounts, obligations,
  notice and renewal provisions. Preserve important dates and amounts.
- Distinguish what the evidence establishes from what is unclear or
  missing. Never invent sections, clauses, dates, or parties.
- Never give legal advice or opinions.
- Cite the source document and page numbers from the evidence labels
  (e.g. "lease.pdf p. 1").

Evidence:
{context}

Question:
{question}
"""


def _normalize(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    return t


def _content_terms(text: str):
    stop = {"a", "an", "the", "is", "are", "was", "it", "that", "this",
            "to", "of", "in", "on", "for", "do", "does", "me", "more",
            "about", "too", "there", "s", "t", "and", "or", "please",
            "my", "you", "your", "i", "what", "how", "which", "with"}
    return [w for w in re.findall(r"[a-z']+", text or "") if w not in stop]


def mentioned_files(text: str):
    """Filenames named in the request, order kept, deduplicated."""
    out = []
    for m in _FILENAME_RE.findall(text or ""):
        low = m.lower()
        if low not in [f.lower() for f in out]:
            out.append(m)
    return out


def detect_workflow(text: str) -> dict:
    """Structural workflow detection (no LLM, no index access).

    Returns {"workflow": NORMAL|COMPARISON|EXTRACTION|SUMMARY,
             "mentioned": [...], "fields": [...] bleed-through for
             extraction/summary disambiguation}.
    Priority: summary (most specific) > comparison > extraction.
    """
    t = _normalize(text)
    files = mentioned_files(text)
    if _SUMMARY_VERBS.search(t) and len(files) == 1:
        # "What does lease.pdf say about renewal?" has no summary verb
        # match issue: it DOESN'T match (no summar*/overview) -> normal.
        # Guard the reverse: "Summarize lease.pdf and compare..." is a
        # comparison; comparison cues win when 2 files are named.
        if not (len(files) >= 2 and _has_comparison_cue(t)):
            return {"workflow": SUMMARY, "mentioned": files, "fields": []}
    if _has_comparison_cue(t):
        return {"workflow": COMPARISON, "mentioned": files, "fields": []}
    fields = _detect_fields(t)
    if fields and _EXTRACTION_VERBS.search(t):
        return {"workflow": EXTRACTION, "mentioned": files,
                "fields": fields}
    return {"workflow": NORMAL, "mentioned": files, "fields": []}


def _has_comparison_cue(t: str) -> bool:
    return any(re.search(p, t) for p in _COMPARISON_CUES)


def _detect_fields(t: str):
    """Canonical legal fields named in the request (order kept, capped)."""
    found = []
    for canonical, variants in FIELD_VOCAB.items():
        if any(v in t for v in variants):
            found.append(canonical)
    return found[:MAX_EXTRACTION_SUBQUERIES]


def known_file_names(vector_store, limit: int = 50, session_id=None):
    """Indexed filenames via the provider abstraction (no model load)."""
    try:
        return [s.get("file_name") for s in vector_store.list_sources(
            limit, session_id=session_id) if s.get("file_name")]
    except TypeError:
        # Duck-typed stores without scope support: persistent view.
        if session_id is not None:
            raise ValueError(
                "Session-scoped listing needs a scope-aware vector store.")
        return [s.get("file_name") for s in vector_store.list_sources(limit)
                if s.get("file_name")]
    except Exception as e:
        logger.warning("Source listing failed: %s", e)
        return []


def _match_known(name: str, known):
    for k in known or []:
        if k and k.lower() == (name or "").lower():
            return k
    return None


def resolve_comparison_targets(mentioned, known):
    """Map to exactly 2 indexed documents or a safe clarification.

    Returns {"targets": [fileA, fileB]|None, "clarification": str|None}.
    Never invents identities: unknown names and ambiguous requests
    yield a deterministic clarification listing real indexed files.
    """
    known = list(known or [])
    matched = []
    for m in mentioned or []:
        hit = _match_known(m, known)
        if hit and hit not in matched:
            matched.append(hit)
    unknown = [m for m in (mentioned or [])
               if not _match_known(m, known)]
    if unknown:
        have = ", ".join(known) if known else "no indexed documents"
        return {"targets": None,
                "clarification": (
                    f"{CLARIFICATION_MARKER} which documents to compare: "
                    f"I don't have {', '.join(unknown)} indexed. "
                    f"Indexed documents: {have}. "
                    f"Please name two, e.g. "
                    f"“Compare the lock-in clauses in {known[0] if known else 'a.pdf'} "
                    f"and {known[1] if len(known) > 1 else 'b.pdf'}.”")}
    if len(matched) >= 2:
        return {"targets": matched[:2], "clarification": None}
    if len(matched) == 1 and len(known) == 2:
        other = known[0] if known[1] == matched[0] else known[1]
        return {"targets": [matched[0], other], "clarification": None}
    if not matched and len(known) == 2:
        # Bare "same/differ" + legal topic against the only two docs.
        return {"targets": list(known), "clarification": None}
    have = ", ".join(known) if known else "no indexed documents"
    return {"targets": None,
            "clarification": (
                f"{CLARIFICATION_MARKER} which documents to compare: "
                f"please name two indexed documents ({have}). "
                f"For example: “Compare the notice periods in "
                f"{known[0] if known else 'a.pdf'} and "
                f"{known[1] if len(known) > 1 else 'b.pdf'}.”")}


def should_auto_resolve_bare_comparison(text: str, known) -> bool:
    """Bare same/differ + legal topic + exactly 2 docs: safe to resolve."""
    if mentioned_files(text) or _PLURAL_DOC_RE.search(_normalize(text)):
        return True
    if len(known or []) != 2:
        return False
    terms = set(_content_terms(_normalize(text)))
    return bool(terms & _COMPARISON_TOPIC_CUES)


def resolve_summary_target(mentioned, known):
    """Exactly one indexed document or a safe clarification (no LLM)."""
    known = list(known or [])
    if len(mentioned or []) == 1:
        hit = _match_known(mentioned[0], known)
        if hit:
            return {"target": hit, "clarification": None}
        have = ", ".join(known) if known else "no indexed documents"
        return {"target": None,
                "clarification": (
                    f"{CLARIFICATION_MARKER} which document to summarize: "
                    f"I don't have {mentioned[0]} indexed. "
                    f"Indexed documents: {have}.")}
    have = ", ".join(known) if known else "no indexed documents"
    return {"target": None,
            "clarification": (
                f"{CLARIFICATION_MARKER} which document to summarize: "
                f"please name one indexed document ({have}).")}


# ---------- evidence collection (existing pipeline only) ---------- #

def _ms(a, b):
    try:
        return max(0, int((float(b) - float(a)) * 1000))
    except (TypeError, ValueError):
        return 0


def retrieve_for_document(topic_query, target_file, config=None,
                          vector_store=None, session_id=None):
    """Evidence scoped to one document: thresholded retrieval filtered to
    the file, topped up with same-file chunks (existing abstraction).

    session_id None means persistent-only; a valid id additionally admits
    that session's rows. Returns (evidence: list[structured], retrieval_ms).
    """
    from backend import _to_structured_source, retrieve_documents

    _t0 = time.monotonic()
    hits = retrieve_documents(topic_query, config=config,
                              vector_store=vector_store,
                              session_id=session_id)
    scoped = [s for s in hits
              if (s.get("file_name") or "").lower()
              == (target_file or "").lower()]
    if vector_store is not None:
        try:
            seen = {s.get("chunk_id") for s in scoped}
            if session_id is None:
                pairs = vector_store.chunks_for_source(target_file)
            else:
                pairs = vector_store.chunks_for_source(
                    target_file, session_id=session_id)
            for doc, _score in pairs:
                struct = _to_structured_source(doc, None)
                if struct.get("chunk_id") not in seen:
                    seen.add(struct.get("chunk_id"))
                    scoped.append(struct)
        except Exception as e:
            logger.warning("Source top-up failed for %s: %s", target_file, e)
    return scoped, _ms(_t0, time.monotonic())


def summary_chunks(target_file, vector_store, limit=MAX_SUMMARY_CHUNKS,
                   session_id=None):
    """Bounded deterministic chunk selection for summaries (no embeddings)."""
    from backend import _to_structured_source

    try:
        if session_id is None:
            pairs = vector_store.chunks_for_source(target_file, limit=limit)
        else:
            pairs = vector_store.chunks_for_source(
                target_file, limit=limit, session_id=session_id)
    except Exception as e:
        logger.warning("Summary chunk fetch failed for %s: %s",
                       target_file, e)
        return []
    out = []
    for doc, _score in (pairs or [])[:limit]:
        struct = _to_structured_source(doc, None)
        content = struct.get("content", "") or ""
        if len(content) > SUMMARY_CHUNK_CHARS:
            struct["content"] = content[:SUMMARY_CHUNK_CHARS]
        out.append(struct)
    return out


def _evidence_block(target_file, evidence):
    lines = [f"--- Source document: {target_file} ---"]
    for s in evidence or []:
        page = s.get("page")
        tag = f"{target_file}" + (f" p. {page}" if page is not None else "")
        lines.append(f"[{tag}] {s.get('content', '')}")
    return "\n".join(lines)


# ---------- structured extraction (deterministic, no LLM) ---------- #

def _variants(term: str):
    yield term
    if term.endswith("s") and len(term) > 4:
        yield term[:-1]
    else:
        yield term + "s"


def _sentence_matches(sentence: str, canonical: str) -> bool:
    low = (sentence or "").lower()
    for variant in _variants(canonical.rstrip("s")):
        if variant and variant in low:
            return True
    for variant in FIELD_VOCAB.get(canonical, ()):
        if variant and len(variant) >= 3 and variant in low:
            return True
    return False


def harvest_extraction_rows(evidence, fields):
    """Verbatim evidence sentences per field (grounded by construction).

    Returns (rows, missing_fields). Rows: {"type", "value", "file_name",
    "page"}. Duplicates removed, stable order.
    """
    from answer_support import split_sentences

    rows, seen, covered = [], set(), set()
    for s in evidence or []:
        for sent in split_sentences(s.get("content", "")):
            snippet = re.sub(r"\s+", " ", sent).strip()
            if len(snippet) < 12:
                continue
            for field in fields or []:
                if _sentence_matches(snippet, field):
                    key = (field, snippet.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    covered.add(field)
                    rows.append({
                        "type": field,
                        "value": snippet[:300],
                        "file_name": s.get("file_name"),
                        "page": s.get("page"),
                    })
    missing = [f for f in (fields or []) if f not in covered]
    return rows[:MAX_EXTRACTION_ROWS], missing


def build_extraction_table(rows, missing_fields=None) -> str:
    """Deterministic markdown table from harvested rows (pure)."""
    lines = ["| # | Type | Value | Document | Page |",
             "|---|------|-------|----------|------|"]
    for i, r in enumerate(rows or [], 1):
        val = (r.get("value", "") or "").replace("|", "/").replace("\n", " ")
        doc = r.get("file_name") or "unknown"
        page = r.get("page")
        lines.append(f"| {i} | {r.get('type', '')} | {val} | {doc} | "
                     f"{page if page is not None else '—'} |")
    out = "\n".join(lines)
    if missing_fields:
        out += ("\n\nNot stated in the retrieved material: "
                + ", ".join(missing_fields) + ".")
    return out


# ---------- workflow runners ---------- #

def _empty_timings():
    return {"retrieval_ms": 0, "support_ms": 0, "preparation_ms": 0,
            "generation_ms": None}


def query_workflow(query_text, config=None, vector_store=None,
                   llm_provider=None, conversation_context=None,
                   known_files=None, on_phase=None, session_id=None,
                   extra_forms=None):
    """Non-streaming workflow runner (tests + compat).

    Returns (answer: str, sources: list, info: dict). NORMAL workflow
    delegates to backend.query_documents so ordinary Q&A is unchanged.
    session_id scopes retrieval (None = persistent-only).
    extra_forms augments normal retrieval only (Phase 5D planned path).
    """
    from backend import query_documents

    t0 = time.monotonic()
    detected = detect_workflow(query_text)
    if detected["workflow"] == NORMAL:
        answer, sources = query_documents(
            query_text, config=config, vector_store=vector_store,
            llm_provider=llm_provider,
            conversation_context=conversation_context, on_phase=on_phase,
            session_id=session_id, extra_forms=extra_forms)
        return answer, sources, {"workflow": NORMAL, "label": None,
                                 "needs_retrieval": True}
    if detected["workflow"] == COMPARISON:
        return _run_comparison(query_text, detected, config, vector_store,
                               llm_provider, known_files, t0, on_phase,
                               stream=False, session_id=session_id)
    if detected["workflow"] == EXTRACTION:
        return _run_extraction(query_text, detected, config, vector_store,
                               known_files, t0, on_phase,
                               session_id=session_id)
    return _run_summary(query_text, detected, config, vector_store,
                        llm_provider, known_files, t0, on_phase,
                        stream=False, session_id=session_id)


def stream_workflow_answer(query_text, config=None, vector_store=None,
                           llm_provider=None, conversation_context=None,
                           known_files=None, on_phase=None, session_id=None,
                           extra_forms=None):
    """Streaming workflow entry (UI). Same (info, stream) shape as
    backend.stream_answer plus workflow/label/fallback keys.

    NORMAL delegates to backend.stream_answer untouched.
    session_id scopes retrieval (None = persistent-only).
    extra_forms augments normal retrieval only (Phase 5D planned path).
    """
    from backend import stream_answer

    detected = detect_workflow(query_text)
    if detected["workflow"] == NORMAL:
        info, stream = stream_answer(
            query_text, config=config, vector_store=vector_store,
            llm_provider=llm_provider,
            conversation_context=conversation_context, on_phase=on_phase,
            session_id=session_id, extra_forms=extra_forms)
        info["workflow"] = NORMAL
        info["label"] = None
        info["fallback_template"] = None
        info["fallback_question"] = query_text
        return info, stream
    if detected["workflow"] == COMPARISON:
        return _run_comparison(query_text, detected, config, vector_store,
                               llm_provider, known_files, time.monotonic(),
                               on_phase, stream=True, session_id=session_id)
    if detected["workflow"] == EXTRACTION:
        return _run_extraction(query_text, detected, config, vector_store,
                               known_files, time.monotonic(), on_phase,
                               stream=True, session_id=session_id)
    return _run_summary(query_text, detected, config, vector_store,
                        llm_provider, known_files, time.monotonic(),
                        on_phase, stream=True, session_id=session_id)


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


def _run_comparison(query_text, detected, config, vector_store,
                    llm_provider, known_files, t0, on_phase, stream,
                    session_id=None):
    from answer_support import UNSUPPORTED, assess_support
    from backend import LOW_RELEVANCE_MESSAGE
    from llm_provider import get_llm_provider
    from output_safety import ThinkStreamSanitizer, strip_think_blocks

    cfg = config
    if cfg is None:
        from config import get_config
        cfg = get_config()
    vs = _resolve_store(cfg, vector_store)
    if known_files is None:
        known_files = known_file_names(vs, session_id=session_id)
    if on_phase is not None:
        on_phase("retrieving")
    _s0 = time.monotonic()
    mentioned = detected.get("mentioned") or []
    if not mentioned and not _PLURAL_DOC_RE.search(_normalize(query_text)) \
            and not should_auto_resolve_bare_comparison(query_text, known_files):
        resolved = {"targets": None, "clarification": None}
        have = ", ".join(known_files) if known_files else "no indexed documents"
        resolved["clarification"] = (
            f"{CLARIFICATION_MARKER} which documents to compare: please "
            f"name two indexed documents ({have}).")
    else:
        resolved = resolve_comparison_targets(mentioned, known_files)
    support_ms = _ms(_s0, time.monotonic())
    if resolved.get("clarification"):
        timings = {"retrieval_ms": 0, "support_ms": support_ms,
                   "preparation_ms": _ms(t0, time.monotonic()),
                   "generation_ms": 0 if not stream else None}
        info = {"answer": resolved["clarification"], "retrieved": [],
                "needs_retrieval": True, "support_level": None,
                "failed": False, "timings": timings,
                "workflow": COMPARISON, "label": PARTIAL_COMPARISON_LABEL,
                "fallback_template": None,
                "fallback_question": query_text}
        return _wrap(info, stream)
    targets = resolved["targets"]
    file_a, file_b = targets[0], targets[1]
    ev_a, ms_a = retrieve_for_document(query_text, file_a, config=cfg,
                                       vector_store=vs, session_id=session_id)
    ev_b, ms_b = retrieve_for_document(query_text, file_b, config=cfg,
                                       vector_store=vs, session_id=session_id)
    retrieval_ms = ms_a + ms_b
    sources = list(ev_a) + list(ev_b)
    if not sources:
        timings = {"retrieval_ms": retrieval_ms, "support_ms": support_ms,
                   "preparation_ms": _ms(t0, time.monotonic()),
                   "generation_ms": 0 if not stream else None}
        info = {"answer": LOW_RELEVANCE_MESSAGE, "retrieved": [],
                "needs_retrieval": True, "support_level": None,
                "failed": False, "timings": timings,
                "workflow": COMPARISON, "label": "Not enough context",
                "fallback_template": None, "fallback_question": query_text}
        return _wrap(info, stream)
    # Partial evidence: a side with only file-browse top-up (score None)
    # and no query-matching (thresholded) chunk must not be presented as
    # fully covered — but its real content still joins the synthesis so a
    # difference is never claimed merely from a retrieval miss.
    def _scored(evidence):
        return [s for s in evidence
                if isinstance(s.get("score"), (int, float))]

    sup_a = assess_support(query_text, ev_a) if ev_a else {"level": UNSUPPORTED}
    sup_b = assess_support(query_text, ev_b) if ev_b else {"level": UNSUPPORTED}
    partial = (not _scored(ev_a)) or (not _scored(ev_b)) or \
        (sup_a["level"] == UNSUPPORTED) or (sup_b["level"] == UNSUPPORTED)
    label = PARTIAL_COMPARISON_LABEL if partial else COMPARISON_LABEL
    context = (_evidence_block(file_a, ev_a) + "\n\n"
               + _evidence_block(file_b, ev_b))
    provider = llm_provider or get_llm_provider(cfg)
    timings = {"retrieval_ms": retrieval_ms, "support_ms": support_ms,
               "preparation_ms": _ms(t0, time.monotonic()),
               "generation_ms": None}
    info = {"answer": None, "retrieved": sources, "needs_retrieval": True,
            "support_level": "comparison", "failed": False,
            "timings": timings, "workflow": COMPARISON, "label": label,
            "fallback_template": COMPARISON_PROMPT_TEMPLATE,
            "fallback_question": query_text,
            "workflow_context": context, "workflow_provider": provider}
    if not stream:
        from backend import EMPTY_RESPONSE_MESSAGE
        from output_safety import is_empty_response

        try:
            raw = provider.generate(context, query_text,
                                    COMPARISON_PROMPT_TEMPLATE)
        except Exception as e:
            logger.warning("Comparison generation failed: %s", e)
            raise ConnectionError(
                "LLM endpoint is unreachable. Make sure LM Studio Server "
                "is running.")
        answer = strip_think_blocks(raw)
        if is_empty_response(raw):
            answer = EMPTY_RESPONSE_MESSAGE
        if partial:
            answer = ("Note: this comparison is incomplete — one document "
                      "has no matching evidence below.\n\n" + answer)
        return answer, sources, info
    return _stream_generated(info, context, query_text,
                             COMPARISON_PROMPT_TEMPLATE, provider,
                             prefix=("Note: this comparison is incomplete — "
                                     "one document has no matching evidence "
                                     "below.\n\n" if partial else ""))


def _run_extraction(query_text, detected, config, vector_store,
                    known_files, t0, on_phase, stream=False,
                    session_id=None):
    from backend import LOW_RELEVANCE_MESSAGE

    cfg = config
    if cfg is None:
        from config import get_config
        cfg = get_config()
    vs = _resolve_store(cfg, vector_store)
    if on_phase is not None:
        on_phase("retrieving")
    fields = detected.get("fields") or []
    mentioned = [m for m in (detected.get("mentioned") or [])
                 if _match_known(m, known_files)] if known_files else []
    scoped = mentioned[:1]
    subqueries = [query_text]
    for field in fields[:MAX_EXTRACTION_SUBQUERIES - 1]:
        q = field + (" " + scoped[0] if scoped else "")
        if q not in subqueries:
            subqueries.append(q)
    best, retrieval_ms = {}, 0
    for q in subqueries[:MAX_EXTRACTION_SUBQUERIES]:
        hits, ms = retrieve_for_document(q, scoped[0], config=cfg,
                                         vector_store=vs,
                                         session_id=session_id) if scoped else \
            _retrieve_all(q, cfg, vs, session_id=session_id)
        retrieval_ms += ms
        for s in hits:
            cid = s.get("chunk_id") or id(s)
            if cid not in best:
                best[cid] = s
    evidence = list(best.values())
    if not evidence:
        timings = {"retrieval_ms": retrieval_ms, "support_ms": 0,
                   "preparation_ms": _ms(t0, time.monotonic()),
                   "generation_ms": 0}
        info = {"answer": LOW_RELEVANCE_MESSAGE, "retrieved": [],
                "needs_retrieval": True, "support_level": None,
                "failed": False, "timings": timings,
                "workflow": EXTRACTION, "label": "Not enough context",
                "fallback_template": None, "fallback_question": query_text}
        return _wrap(info, stream)
    rows, missing = harvest_extraction_rows(evidence, fields)
    if not rows:
        timings = {"retrieval_ms": retrieval_ms, "support_ms": 0,
                   "preparation_ms": _ms(t0, time.monotonic()),
                   "generation_ms": 0}
        info = {"answer": LOW_RELEVANCE_MESSAGE, "retrieved": evidence,
                "needs_retrieval": True, "support_level": None,
                "failed": False, "timings": timings,
                "workflow": EXTRACTION, "label": "Not enough context",
                "fallback_template": None, "fallback_question": query_text}
        return _wrap(info, stream)
    table = build_extraction_table(rows, missing)
    label = EXTRACTION_LABEL if not missing else "Partial answer"
    timings = {"retrieval_ms": retrieval_ms, "support_ms": 0,
               "preparation_ms": _ms(t0, time.monotonic()),
               "generation_ms": 0}
    info = {"answer": table, "retrieved": evidence,
            "needs_retrieval": True, "support_level": None,
            "failed": False, "timings": timings,
            "workflow": EXTRACTION, "label": label,
            "fallback_template": None, "fallback_question": query_text}
    return _wrap(info, stream)


def _retrieve_all(query_text, cfg, vs, session_id=None):
    """Unscoped retrieval with timing (extraction base query)."""
    from backend import retrieve_documents

    _t0 = time.monotonic()
    try:
        hits = retrieve_documents(query_text, config=cfg, vector_store=vs,
                                  session_id=session_id)
    except Exception as e:
        logger.warning("Retrieval failed: %s", e)
        return [], _ms(_t0, time.monotonic())
    return hits, _ms(_t0, time.monotonic())


def _run_summary(query_text, detected, config, vector_store,
                 llm_provider, known_files, t0, on_phase, stream,
                 session_id=None):
    from llm_provider import get_llm_provider

    cfg = config
    if cfg is None:
        from config import get_config
        cfg = get_config()
    vs = _resolve_store(cfg, vector_store)
    if known_files is None:
        known_files = known_file_names(vs, session_id=session_id)
    if on_phase is not None:
        on_phase("retrieving")
    _s0 = time.monotonic()
    resolved = resolve_summary_target(detected.get("mentioned") or [],
                                      known_files)
    support_ms = _ms(_s0, time.monotonic())
    if resolved.get("clarification"):
        timings = {"retrieval_ms": 0, "support_ms": support_ms,
                   "preparation_ms": _ms(t0, time.monotonic()),
                   "generation_ms": 0 if not stream else None}
        info = {"answer": resolved["clarification"], "retrieved": [],
                "needs_retrieval": True, "support_level": None,
                "failed": False, "timings": timings,
                "workflow": SUMMARY, "label": "Partial answer",
                "fallback_template": None, "fallback_question": query_text}
        return _wrap(info, stream)
    target = resolved["target"]
    chunks = summary_chunks(target, vs, session_id=session_id)
    if not chunks:
        timings = {"retrieval_ms": 0, "support_ms": support_ms,
                   "preparation_ms": _ms(t0, time.monotonic()),
                   "generation_ms": 0 if not stream else None}
        info = {"answer": (
                    f"I don't have enough indexed material in {target} "
                    f"to summarize it yet. Build the knowledge base first."),
                "retrieved": [], "needs_retrieval": True,
                "support_level": None, "failed": False, "timings": timings,
                "workflow": SUMMARY, "label": "Not enough context",
                "fallback_template": None, "fallback_question": query_text}
        return _wrap(info, stream)
    provider = llm_provider or get_llm_provider(cfg)
    timings = {"retrieval_ms": 0, "support_ms": support_ms,
               "preparation_ms": _ms(t0, time.monotonic()),
               "generation_ms": None}
    if len(chunks) <= SUMMARY_GROUP_SIZE:
        context = _evidence_block(target, chunks)
        template = SUMMARY_PROMPT_TEMPLATE
    else:
        # Bounded staged summarization: sequential part-summaries over
        # capped groups, then one final call (no fanout, no extra models).
        parts = []
        for i in range(0, len(chunks), SUMMARY_GROUP_SIZE):
            group = chunks[i:i + SUMMARY_GROUP_SIZE]
            grp_ctx = _evidence_block(target, group)
            try:
                raw = provider.generate(
                    grp_ctx,
                    f"Summarize this section of {target}.",
                    SUMMARY_PART_PROMPT_TEMPLATE)
                from output_safety import strip_think_blocks
                parts.append(strip_think_blocks(raw))
            except Exception as e:
                logger.warning("Summary part failed: %s", e)
                raise ConnectionError(
                    "LLM endpoint is unreachable. Make sure LM Studio "
                    "Server is running.")
        context = (f"--- Part-summaries of {target} ---\n"
                   + "\n\n".join(parts))
        template = SUMMARY_PROMPT_TEMPLATE
    info = {"answer": None, "retrieved": chunks, "needs_retrieval": True,
            "support_level": "summary", "failed": False,
            "timings": timings, "workflow": SUMMARY, "label": SUMMARY_LABEL,
            "fallback_template": template, "fallback_question": query_text,
            "workflow_context": context, "workflow_provider": provider}
    if not stream:
        from backend import EMPTY_RESPONSE_MESSAGE
        from output_safety import is_empty_response, strip_think_blocks

        try:
            raw = provider.generate(context, query_text, template)
        except Exception as e:
            logger.warning("Summary generation failed: %s", e)
            raise ConnectionError(
                "LLM endpoint is unreachable. Make sure LM Studio Server "
                "is running.")
        answer = strip_think_blocks(raw)
        if is_empty_response(raw):
            answer = EMPTY_RESPONSE_MESSAGE
        return answer, chunks, info
    return _stream_generated(info, context, query_text, template, provider)


def _one_shot(text):
    yield text


def _wrap(info, stream):
    """Return (info, stream) or (answer, sources, info) by mode."""
    if stream:
        return info, _one_shot(info["answer"])
    return info["answer"], info["retrieved"], info


def _stream_generated(info, context, question, template, provider, prefix=""):
    """Sanitized streaming generation with clean non-stream fallback."""
    from backend import _one_shot
    from output_safety import ThinkStreamSanitizer

    sanitizer = ThinkStreamSanitizer()

    def _gen():
        try:
            if prefix:
                yield prefix
            for raw in provider.generate_stream(context, question,
                                                template):
                safe = sanitizer.feed(raw)
                if safe:
                    yield safe
        except Exception:
            logger.warning("Workflow streaming failed; caller falls back.")
            raise
        tail = sanitizer.flush()
        if tail:
            yield tail

    # Fallback path reuses the SAME evidence/template (never re-retrieves).
    info["fallback_template"] = template
    info["fallback_question"] = question
    if prefix:
        info["fallback_prefix"] = prefix
    return info, _gen()
