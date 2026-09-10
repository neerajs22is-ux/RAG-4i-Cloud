"""Refactored backend: ingestion + retrieval + generation via providers.

Preserved behaviour:
- PyPDFLoader, chunk 1000/200, all-MiniLM-L6-v2, top-k=5, threshold 0.3,
  prompt template unchanged.
- No destructive rebuild (existing Chroma data is kept).
"""

import logging
import os
import time

from config import get_config

logger = logging.getLogger(__name__)

# --- PROMPT (UNCHANGED from original RAG-4i) --- #
PROMPT_TEMPLATE = """
You are an expert legal assistant for a Chartered Accountant firm. 
Answer the question based ONLY on the following context. 
If the answer is not in the context, strictly say "I cannot find this information in the provided documents."

Context:
{context}

Question:
{question}
"""

# Partial/unsupported-context prompt: same grounding contract, plus honest
# scoping (state what IS established, mark the rest unknown, suggest only
# search directions — never claim unseen sections exist).
PARTIAL_PROMPT_TEMPLATE = """
You are an expert legal assistant for a Chartered Accountant firm.
Answer the question using ONLY the following context.

Rules:
- State clearly what the context DOES establish, quoting it closely.
- If the exact question is not fully answered, say plainly what cannot
  be established from this context. Do not guess or fill gaps.
- You may suggest which kinds of provisions would be relevant to check
  (for example termination, notice, or payment provisions), but present
  them ONLY as search directions, never as claims that such sections exist.
- Distinguish facts ("the text states...") from reasonable readings
  ("this suggests..."). When in doubt, use factual wording.
- Never invent contract terms, dates, parties, or mechanics.

Context:
{context}

Question:
{question}
"""

LOW_RELEVANCE_MESSAGE = (
    "I could not find enough relevant information in the documents to answer that."
)
EMPTY_RESPONSE_MESSAGE = (
    "The model returned an empty response. Please try again."
)

# Support levels (see answer_support.py); OVERVIEW shares the scoping prompt.
PARTIAL_SUPPORT = "partial"
OVERVIEW_SUPPORT = "overview"


# ---------- Ingestion ---------- #

def _looks_like_windows_path(path: str) -> bool:
    """True for 'C:\\...' / 'C:/...' style paths on a non-Windows host."""
    import re
    return os.name != "nt" and bool(re.match(r"^[A-Za-z]:[\\/]", path or ""))


def ingest_with_report(folder_path, config=None,
                       storage=None, vector_store=None, on_progress=None):
    """Build/extend the index, returning (success, message, details).

    details always contains: source (local|s3), found, pages, chars,
    chunks, embeddings, vectors_stored, succeeded, failed, failed_files
    (list of {file, reason}). Zero/empty outcomes are explicit, never masked.
    (list of {file, reason}). Zero/empty outcomes are explicit, never masked.
    on_progress(current, total, filename, outcome) mirrors the loader's
    per-document callback (see document_loader).
    """
    from chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_documents
    from document_loader import load_documents_from_folder
    from document_storage import LocalDocumentStorage, get_document_storage
    from embeddings import get_embedding_provider
    from vector_store import get_vector_store

    cfg = config or get_config()
    use_s3 = (getattr(cfg, "document_storage", "local") or "local").lower() == "s3"
    source = "s3" if use_s3 else "local"
    details = {"source": source, "found": 0, "pages": 0, "chars": 0,
               "chunks": 0, "embeddings": 0, "vectors_stored": 0,
               "succeeded": 0, "failed": 0, "failed_files": []}

    if use_s3:
        # S3-backed ingestion: same loader/chunker/embedder, S3 as source.
        # folder_path is intentionally unused here (cloud has no local path).
        if not getattr(cfg, "s3_bucket", ""):
            return False, "S3 bucket is not configured (S3_BUCKET).", details
        from document_loader import load_documents_from_s3

        storage = storage or get_document_storage(cfg)
        try:
            documents, report = load_documents_from_s3(
                storage, on_progress=on_progress)
        except Exception as e:
            logger.warning("S3 ingestion failed: %s", e)
            return False, f"Failed to read from S3: {e}", details
    else:
        if not folder_path:
            return False, "Folder path is required (local mode).", details
        if _looks_like_windows_path(folder_path):
            return False, (
                "That looks like a Windows path, but the application is "
                "running on a non-Windows host (cloud mode). Use S3-backed "
                "ingestion instead of a browser-computer path."), details
        if not os.path.exists(folder_path):
            return False, "Folder path does not exist.", details
        if not os.path.isdir(folder_path):
            return False, "Folder path is not a directory.", details

        if storage is None:
            storage = LocalDocumentStorage(folder_path)

        # 1-2. Locate + load PDFs (per-file errors recorded, not swallowed).
        try:
            documents, report = load_documents_from_folder(
                folder_path, storage, on_progress=on_progress)
        except Exception as e:
            logger.warning("Ingestion failed: %s", e)
            return False, f"Failed to scan folder: {e}", details

    details.update({k: report.get(k, details[k])
                    for k in ("found", "succeeded", "failed")})
    # pages/chars may be absent in older report dicts (e.g. test doubles);
    # derive from loaded documents then (None = unknown, 0 = known-empty).
    details["pages"] = report.get("pages")
    if details["pages"] is None:
        details["pages"] = len(documents)
    details["chars"] = report.get("chars")
    if details["chars"] is None:
        details["chars"] = sum(len(getattr(d, "page_content", "") or "")
                               for d in documents)
    for name in report.get("failed_files", []):
        details["failed_files"].append({
            "file": name,
            "reason": report.get("failed_errors", {}).get(name, "load failed"),
        })

    if report["found"] == 0:
        if use_s3:
            return False, "No PDF files found in the S3 bucket/prefix.", details
        return False, "No PDF files found in that folder.", details

    if not documents:
        failed = ", ".join(report["failed_files"]) if report["failed_files"] else "unknown"
        return False, (
            f"Found {report['found']} document(s) but none could be processed. "
            f"Failed: {report['failed']} ({failed})."
        ), details

    if details["chars"] == 0:
        return False, (
            f"Processed {report['succeeded']} document(s) "
            f"({details['pages']} pages) but extracted zero usable text. "
            "The PDFs may be scanned images (OCR not enabled) or empty."
        ), details

    # 3. Chunk (1000/200 preserved).
    try:
        chunks = chunk_documents(
            documents,
            chunk_size=getattr(cfg, "chunk_size", CHUNK_SIZE),
            chunk_overlap=getattr(cfg, "chunk_overlap", CHUNK_OVERLAP),
        )
    except Exception as e:
        logger.warning("Chunking failed: %s", e)
        return False, f"Failed to chunk documents: {e}", details

    if not chunks:
        return False, "No text could be extracted from the PDFs.", details
    details["chunks"] = len(chunks)

    # 4-5. Embed + store (appends; never deletes existing DB).
    try:
        if vector_store is None:
            embedding_provider = get_embedding_provider(cfg)
            vector_store = get_vector_store(cfg, embedding_provider)
        details["embeddings"] = len(chunks)
        added = vector_store.build_index(chunks)
        details["vectors_stored"] = added
    except Exception as e:
        logger.warning("Vector store build failed: %s", e)
        return False, (
            "Failed to build the index (vector DB unavailable). "
            "Check that dependencies are installed and chroma_db is writable."
        ), details

    base = (
        f"Indexed {report['succeeded']}/{report['found']} document(s) "
        f"({added} chunks). "
        f"Succeeded: {report['succeeded']}, Failed: {report['failed']}."
    )
    if report["failed"]:
        failed_names = ", ".join(report["failed_files"])
        return True, base + f" Failed files: {failed_names}.", details
    return True, base + " Success! Knowledge Base updated (existing data kept).", details


def create_vector_db_from_folder(folder_path, config=None,
                                 storage=None, vector_store=None):
    """Build/extend the index from a local folder (non-destructive).

    Returns (success: bool, message: str). Message always includes
    found/succeeded/failed counts and failed filenames when any fail.
    """
    ok, msg, _details = ingest_with_report(folder_path, config=config,
                                           storage=storage,
                                           vector_store=vector_store)
    return ok, msg


# ---------- Retrieval ---------- #

def _to_structured_source(doc, score):
    meta = getattr(doc, "metadata", {}) or {}
    source_path = meta.get("source_path") or meta.get("source")
    file_name = meta.get("file_name")
    if not file_name and source_path:
        file_name = os.path.basename(source_path)
    page = meta.get("page", None)
    if not isinstance(page, int):
        page = None
    try:
        score_val = float(score)
    except (TypeError, ValueError):
        score_val = None
    return {
        "source": source_path,
        "source_path": source_path,
        "file_name": file_name,
        "page": page,
        "score": score_val,
        "document_id": meta.get("document_id"),
        "chunk_id": meta.get("chunk_id"),
        "content": getattr(doc, "page_content", ""),
    }


def normalize_query(query_text: str) -> str:
    """Canonical form: trim, collapse whitespace, drop trailing ?/!/..."""
    import re
    t = re.sub(r"\s+", " ", (query_text or "").strip())
    return t.rstrip("?!.")


_CORE_DROP = {
    "is", "are", "was", "the", "a", "an", "do", "does", "tell", "about",
    "please", "to", "of", "on", "for", "me", "you", "your", "i", "my",
    "it", "that", "this",
}


def query_core(query_text: str) -> str:
    """Light de-framed core: drop function words, keep question words and
    content words (measured: preserves meaning, never invents terms)."""
    import re
    words = re.findall(r"[A-Za-z'-]+", normalize_query(query_text))
    kept = [w for w in words if w.lower() not in _CORE_DROP]
    return " ".join(kept).strip()


def build_query_forms(query_text: str):
    """Deterministic retrieval forms: [normalized, content core].

    No LLM rewriting. Forms are deduplicated; the original wording always
    participates so behavior can only gain recall, never lose it.

    NOTE: a curated legal-synonym third form was measured against MiniLM
    and consistently scored lowest of all forms (it never became the
    max-pooled winner), so it was deliberately NOT shipped.
    """
    forms = []
    for form in (normalize_query(query_text), query_core(query_text)):
        if form and form not in forms:
            forms.append(form)
    return forms or [query_text]


def retrieve_documents(query_text, config=None, vector_store=None, k=None,
                       threshold=None, session_id=None, extra_forms=None):
    """Search the vector store; return structured sources (filtered by threshold).

    Tries each deterministic query form and keeps each chunk's best score
    (max-pooling), then applies the unchanged top-k/threshold rule.
    session_id None means persistent-only; a valid opaque id additionally
    admits that session's rows (never another session's). Malformed IDs
    raise ValueError. The store is called without the kwarg when unbound
    so duck-typed stores keep working.
    extra_forms (Phase 5D): validated planner queries APPENDED after the
    deterministic forms (deduplicated). Same mechanism, scoring, top-k,
    and threshold. None/empty means no augmentation (4.12 behavior).
    """
    from document_scope import validate_session_id
    from embeddings import get_embedding_provider
    from vector_store import get_vector_store

    if session_id is not None:
        validate_session_id(session_id)
    cfg = config or get_config()
    top_k = k if k is not None else getattr(cfg, "retrieval_k", 5)
    thresh = threshold if threshold is not None else getattr(cfg, "relevance_threshold", 0.3)

    if vector_store is None:
        embedding_provider = get_embedding_provider(cfg)
        vector_store = get_vector_store(cfg, embedding_provider)

    # Raises if DB missing/unavailable -> caller maps to user message.
    best = {}
    order = []
    forms = build_query_forms(query_text)
    for extra in extra_forms or []:
        cleaned = (extra or "").strip()
        if cleaned and cleaned not in forms:
            forms.append(cleaned)
    for form in forms:
        if session_id is None:
            hits = vector_store.search(form, k=top_k)
        else:
            hits = vector_store.search(form, k=top_k, session_id=session_id)
        for doc, score in hits:
            try:
                s = float(score)
            except (TypeError, ValueError):
                continue
            cid = (getattr(doc, "metadata", {}) or {}).get("chunk_id") \
                or id(doc)
            if cid not in best or s > best[cid][1]:
                best[cid] = (doc, s)
            if cid not in order:
                order.append(cid)
    results = sorted((best[cid] for cid in order),
                     key=lambda pair: pair[1], reverse=True)[:top_k]
    structured = [_to_structured_source(doc, score) for doc, score in results]
    # Preserve original behaviour: relevance threshold 0.3.
    filtered = [s for s in structured if s["score"] is not None and s["score"] >= thresh]
    return filtered


# ---------- Generation ---------- #

def generate_answer(question, retrieved_sources, config=None, llm_provider=None,
                    prompt_template=None, support_level=None):
    """Invoke the LLM over retrieved context; return answer string.

    Raw model output is normalized at this boundary (think-block removal);
    a think-only/empty response is reported explicitly, never shown raw.
    support_level selects the prompt: PARTIAL/OVERVIEW evidence uses the
    scoping prompt; DIRECT keeps the original prompt unchanged.
    """
    from llm_provider import get_llm_provider
    from output_safety import is_empty_response, strip_think_blocks

    cfg = config or get_config()
    template = _select_template(support_level) \
        if prompt_template is None else prompt_template
    if not retrieved_sources:
        return LOW_RELEVANCE_MESSAGE
    context_text = "\n\n---\n\n".join([s.get("content", "") for s in retrieved_sources])
    provider = llm_provider or get_llm_provider(cfg)
    try:
        raw = provider.generate(context_text, question, template)
    except Exception as e:
        logger.warning("LLM generation failed: %s", e)
        hint = getattr(provider, "offline_message", None)
        if callable(hint):
            # Provider-specific guidance (lmstudio text unchanged).
            raise ConnectionError(hint(cfg, e))
        raise ConnectionError(
            "LLM endpoint is unreachable. Make sure LM Studio Server is running "
            f"at {getattr(cfg, 'llm_base_url', 'http://localhost:1234/v1')}."
        )
    answer = strip_think_blocks(raw)
    if is_empty_response(raw):
        logger.info("LLM returned only hidden reasoning or empty text.")
        return EMPTY_RESPONSE_MESSAGE
    return answer


def query_documents(query_text, config=None, vector_store=None, llm_provider=None,
                    conversation_context=None, on_phase=None,
                    session_id=None, extra_forms=None):
    """Combined retrieve + generate (kept for UI compat).

    The intent observer routes first: conversation/capability/out-of-scope
    are answered directly without retrieval; document intents (including
    followups, whose query is anchored to the prior question) use the full
    RAG pipeline unchanged. Returns (answer: str, sources: list[dict]).

    on_phase(name) is an optional progress hook ("retrieving" /
    "generating") for honest staged loading states. It never changes
    behavior and defaults to off.

    session_id scopes retrieval (None = persistent-only). Malformed IDs
    raise ValueError.
    """
    from query_router import (CAPABILITY, CONVERSATION, DOCUMENT_FOLLOWUP,
                              OUT_OF_SCOPE, expand_followup_query,
                              observe_query, reply_for_route)

    obs = observe_query(query_text, conversation_context)
    if obs.intent in (CONVERSATION, CAPABILITY, OUT_OF_SCOPE):
        return reply_for_route(query_text, obs.intent), []
    cfg = config or get_config()
    _t0 = time.monotonic()
    prepared = _prepare_generation(query_text, cfg, vector_store,
                                   llm_provider, conversation_context,
                                   obs, on_phase, session_id,
                                   extra_forms=extra_forms)
    _prep_timings = prepared.get("timings", {})
    if prepared["failed"]:
        logger.info("query answered in %.2fs (sources=%d, route=%s "
                    "retrieval_ms=%s support_ms=%s)",
                    time.monotonic() - _t0, len(prepared["retrieved"]),
                    obs.intent, _prep_timings.get("retrieval_ms"),
                    _prep_timings.get("support_ms"))
        return prepared["answer"], prepared["retrieved"]
    if prepared["answer"] is not None:
        # Decided without the LLM (e.g. unsupported evidence).
        _total_ms = max(0, int((time.monotonic() - _t0) * 1000))
        logger.info("query answered in %.2fs (sources=%d, route=%s "
                    "retrieval_ms=%s support_ms=%s generation_ms=0 total_ms=%d)",
                    time.monotonic() - _t0, len(prepared["retrieved"]),
                    obs.intent, _prep_timings.get("retrieval_ms"),
                    _prep_timings.get("support_ms"), _total_ms)
        return prepared["answer"], prepared["retrieved"]
    if on_phase is not None:
        on_phase("generating")
    _g0 = time.monotonic()
    try:
        answer = generate_answer(
            prepared.get("effective", query_text), prepared["retrieved"],
            config=cfg, llm_provider=prepared["provider"],
            support_level=prepared["support_level"],
        )
    except ConnectionError as e:
        return str(e), prepared["retrieved"]
    except Exception as e:
        logger.warning("Query failed: %s", e)
        return (
            "An error occurred while generating the answer. "
            "Make sure LM Studio Server is running.",
            prepared["retrieved"],
        )
    _generation_ms = max(0, int((time.monotonic() - _g0) * 1000))
    _total_ms = max(0, int((time.monotonic() - _t0) * 1000))
    logger.info("query answered in %.2fs (sources=%d, route=%s "
                "retrieval_ms=%s support_ms=%s generation_ms=%d total_ms=%d)",
                time.monotonic() - _t0, len(prepared["retrieved"]),
                obs.intent, _prep_timings.get("retrieval_ms"),
                _prep_timings.get("support_ms"), _generation_ms, _total_ms)
    return answer, prepared["retrieved"]


def _prepare_generation(query_text, cfg, vector_store, llm_provider,
                        conversation_context, obs, on_phase=None,
                        session_id=None, extra_forms=None):
    """Shared preparation for streaming and non-streaming generation.

    Runs routing (already done by caller), retrieval (exactly once),
    support assessment, and prompt selection. Returns a dict with:
      failed (bool), answer (str|None: set when decided without the LLM),
      retrieved, support_level, provider.
    UNSUPPORTED never reaches generation: answer is set, no LLM needed.
    PARTIAL with missing terms yields one deterministic clarification
    (never a loop: confirmations resolve to the original question first).

    Timings (real monotonic-clock measurements, no behaviour change):
      timings = {"retrieval_ms": int, "support_ms": int,
                 "preparation_ms": int} on every return path.

    session_id scopes retrieval (None = persistent-only). Malformed IDs
    raise ValueError here, before any retrieval work.
    """
    from answer_support import (DIRECT, PARTIAL, UNSUPPORTED,
                                assess_support, build_clarification,
                                detect_broad_scope,
                                resolve_effective_question,
                                unsupported_reply)
    from document_scope import validate_session_id
    from embeddings import get_embedding_provider
    from llm_provider import get_llm_provider
    from query_router import DOCUMENT_FOLLOWUP, expand_followup_query
    from vector_store import get_vector_store

    if session_id is not None:
        validate_session_id(session_id)

    def _ms(a, b):
        try:
            return max(0, int((float(b) - float(a)) * 1000))
        except (TypeError, ValueError):
            return 0

    _prep_t0 = time.monotonic()
    if vector_store is None:
        vector_store = get_vector_store(
            cfg, get_embedding_provider(cfg))
    provider = llm_provider or get_llm_provider(cfg)
    effective, already_clarified = resolve_effective_question(
        query_text, conversation_context)
    retrieval_query = effective
    if already_clarified or obs.intent == DOCUMENT_FOLLOWUP:
        # Confirmed questions need their topic context too: expand the
        # original question the same way a follow-up would be expanded.
        retrieval_query = expand_followup_query(effective, conversation_context)
    broad, target_file = detect_broad_scope(effective)
    if on_phase is not None:
        on_phase("retrieving")
    _r0 = time.monotonic()
    try:
        retrieved = retrieve_documents(
            retrieval_query, config=cfg, vector_store=vector_store,
            session_id=session_id, extra_forms=extra_forms,
        )
        if broad and target_file:
            # Source-aware top-up: same-file chunks as admissible file
            # evidence (threshold still guards query-similarity results).
            # Degrades to retrieval-only if the provider top-up fails.
            similar = len(retrieved)
            try:
                if session_id is None:
                    top_up = vector_store.chunks_for_source(target_file)
                else:
                    top_up = vector_store.chunks_for_source(
                        target_file, session_id=session_id)
            except Exception as e:
                logger.warning("Source top-up failed for %s: %s",
                               target_file, e)
                top_up = []
            seen = {s.get("chunk_id") for s in retrieved}
            for doc, _score in top_up:
                struct = _to_structured_source(doc, None)
                if struct.get("chunk_id") not in seen:
                    seen.add(struct.get("chunk_id"))
                    retrieved.append(struct)
            logger.info("broad scope file=%s similar=%d total=%d",
                        target_file, similar, len(retrieved))
    except Exception as e:
        logger.warning("Retrieval failed: %s", e)
        _retrieval_ms = _ms(_r0, time.monotonic())
        return {"failed": True, "answer": (
            "The knowledge base is unavailable. Please build the index first "
            "and check that the vector database is accessible."),
            "retrieved": [], "support_level": None, "provider": provider,
            "effective": effective,
            "timings": {"retrieval_ms": _retrieval_ms, "support_ms": 0,
                        "preparation_ms": _ms(_prep_t0, time.monotonic())}}
    _retrieval_ms = _ms(_r0, time.monotonic())
    _s0 = time.monotonic()
    if not retrieved:
        return {"failed": False, "answer": LOW_RELEVANCE_MESSAGE,
                "retrieved": [], "support_level": None, "provider": provider,
                "effective": effective,
                "timings": {"retrieval_ms": _retrieval_ms, "support_ms": 0,
                            "preparation_ms": _ms(_prep_t0, time.monotonic())}}
    support = assess_support(retrieval_query, retrieved)
    if support["level"] == UNSUPPORTED:
        _support_ms = _ms(_s0, time.monotonic())
        return {"failed": False, "answer": unsupported_reply(effective),
                "retrieved": retrieved, "support_level": None,
                "provider": provider, "effective": effective,
                "timings": {"retrieval_ms": _retrieval_ms,
                            "support_ms": _support_ms,
                            "preparation_ms": _ms(_prep_t0, time.monotonic())}}
    if (support["level"] == PARTIAL and support.get("missing")
            and not already_clarified):
        _support_ms = _ms(_s0, time.monotonic())
        return {"failed": False,
                "answer": build_clarification(
                    effective, support, retrieved),
                "retrieved": retrieved, "support_level": None,
                "provider": provider, "effective": effective,
                "timings": {"retrieval_ms": _retrieval_ms,
                            "support_ms": _support_ms,
                            "preparation_ms": _ms(_prep_t0, time.monotonic())}}
    level = OVERVIEW_SUPPORT if broad else (
        None if support["level"] == DIRECT else PARTIAL_SUPPORT)
    _support_ms = _ms(_s0, time.monotonic())
    return {"failed": False, "answer": None, "retrieved": retrieved,
            "support_level": level, "provider": provider,
            "effective": effective,
            "timings": {"retrieval_ms": _retrieval_ms,
                        "support_ms": _support_ms,
                        "preparation_ms": _ms(_prep_t0, time.monotonic())}}


def preview_answer(query_text, config=None, vector_store=None,
                   conversation_context=None, session_id=None):
    """Decide whether a question would reach generation (no LLM call).

    Runs the exact preparation stream_answer uses (routing, broad-scope
    top-up, support assessment) and reports whether a generated answer
    would follow. Used to validate suggested questions before showing
    them. Returns {"will_generate": bool, "reason": str}.
    session_id scopes retrieval (None = persistent-only).
    """
    from query_router import (CAPABILITY, CONVERSATION,
                              OUT_OF_SCOPE, observe_query)

    cfg = config or get_config()
    obs = observe_query(query_text, conversation_context)
    if obs.intent in (CONVERSATION, CAPABILITY, OUT_OF_SCOPE):
        return {"will_generate": False,
                "reason": "routed reply, no generation"}
    prepared = _prepare_generation(query_text, cfg, vector_store, None,
                                   conversation_context, obs,
                                   session_id=session_id)
    if prepared["failed"]:
        return {"will_generate": False, "reason": "retrieval unavailable"}
    if prepared["answer"] is not None:
        return {"will_generate": False,
                "reason": "decided without generation"}
    return {"will_generate": True, "reason": "generation"}


def stream_answer(query_text, config=None, vector_store=None,
                  llm_provider=None, conversation_context=None,
                  on_phase=None, session_id=None, extra_forms=None):
    """Streaming answer with synchronous preparation metadata.

    Runs routing/retrieval/assessment exactly once (same evidence and
    prompts as query_documents()) and returns (info, stream) where info
    holds retrieved/needs_retrieval/support_level/answer. If info["answer"]
    is set, the answer was decided without the LLM and stream yields it
    whole. Otherwise stream yields sanitized generation chunks; the caller
    assembles the final text and applies the empty-response guard.
    On generation failure the stream raises so the caller can fall back
    cleanly (reusing info["retrieved"], never re-retrieving).
    session_id scopes retrieval (None = persistent-only).
    """
    from output_safety import ThinkStreamSanitizer

    cfg = config or get_config()
    from query_router import (CAPABILITY, CONVERSATION, DOCUMENT_FOLLOWUP,
                              OUT_OF_SCOPE, expand_followup_query,
                              observe_query, reply_for_route)
    obs = observe_query(query_text, conversation_context)
    needs_retrieval = obs.needs_retrieval
    if obs.intent in (CONVERSATION, CAPABILITY, OUT_OF_SCOPE):
        answer = reply_for_route(query_text, obs.intent)
        return ({"answer": answer, "retrieved": [], "needs_retrieval": False,
                 "support_level": None, "failed": False,
                 "timings": {"retrieval_ms": 0, "support_ms": 0,
                             "preparation_ms": 0, "generation_ms": None}},
                _one_shot(answer))

    prepared = _prepare_generation(query_text, cfg, vector_store,
                                   llm_provider, conversation_context,
                                   obs, on_phase, session_id,
                                   extra_forms=extra_forms)
    info = {"answer": prepared["answer"],
            "retrieved": prepared["retrieved"],
            "needs_retrieval": True,
            "support_level": prepared["support_level"],
            "failed": prepared["failed"],
            "timings": dict(prepared.get("timings", {}))}
    # Generation timing is measured by the caller (stream consumption);
    # placeholder keeps the shape stable for telemetry.
    info["timings"].setdefault("generation_ms", None)
    if prepared["answer"] is not None or prepared["failed"]:
        return info, _one_shot(prepared["answer"])

    sanitizer = ThinkStreamSanitizer()
    provider = prepared["provider"]
    context_text = "\n\n---\n\n".join(
        s.get("content", "") for s in prepared["retrieved"])
    template = _select_template(prepared["support_level"])

    def _gen():
        try:
            for raw in provider.generate_stream(
                    context_text, prepared.get("effective", query_text),
                    template):
                safe = sanitizer.feed(raw)
                if safe:
                    yield safe
        except Exception:
            logger.warning("Streaming generation failed; caller falls back.")
            raise
        tail = sanitizer.flush()
        if tail:
            yield tail

    return info, _gen()


def _one_shot(text):
    yield text


def generate_answer_stream(query_text, config=None, vector_store=None,
                           llm_provider=None, conversation_context=None,
                           on_phase=None):
    """Yield sanitized answer text incrementally (streaming path).

    Preparation (routing/retrieval/assessment) runs exactly once with the
    same evidence and prompts as query_documents(); only generation chunks
    are streamed. Yields sanitized text pieces; the caller assembles the
    final answer and applies the usual empty-response guard. Raises on
    generation failure so the caller can fall back cleanly.
    """
    info, stream = stream_answer(
        query_text, config=config, vector_store=vector_store,
        llm_provider=llm_provider, conversation_context=conversation_context,
        on_phase=on_phase)
    for piece in stream:
        yield piece


def _select_template(support_level):
    if support_level in (PARTIAL_SUPPORT, OVERVIEW_SUPPORT):
        return PARTIAL_PROMPT_TEMPLATE
    return PROMPT_TEMPLATE


def classify_query(query_text) -> str:
    """Routing label for a message (for UI display; routing is deterministic)."""
    from query_router import route_query

    return route_query(query_text)


def delete_indexed_document(document_id: str, config=None, vector_store=None,
                            storage=None, storage_path=None) -> dict:
    """Delete a document's vectors (and optionally its stored file).

    Additive-safe inverse of ingestion: removes chunks matching document_id
    from the active vector store; if storage+storage_path are given, also
    deletes the stored file. Never touches other documents.
    Returns {"vectors_removed": int, "file_removed": bool}.
    """
    from vector_store import get_vector_store

    cfg = config or get_config()
    vs = vector_store
    if vs is None:
        from embeddings import get_embedding_provider

        vs = get_vector_store(cfg, get_embedding_provider(cfg))
    try:
        removed = vs.delete_by_document(document_id) or 0
    except Exception as e:
        logger.warning("Vector delete failed for %s: %s", document_id, e)
        removed = 0
    file_removed = False
    if storage is not None and storage_path:
        try:
            storage.delete_document(storage_path)
            file_removed = True
        except Exception as e:
            logger.warning("File delete failed for %s: %s", storage_path, e)
    logger.info("delete document_id=%s vectors=%d file=%s",
                document_id, removed, file_removed)
    return {"vectors_removed": removed, "file_removed": file_removed}


# ---------- Status helpers (for UI, all from actual checks) ---------- #

def get_knowledge_base_status(config=None, vector_store=None) -> dict:
    """Return real KB state: ready, chunk_count, path."""
    from vector_store import get_vector_store

    cfg = config or get_config()
    try:
        vs = vector_store
        if vs is None:
            # Status must not load the embedding model when DB is missing;
            # get_status() returns early without model init in that case.
            vs = get_vector_store(cfg)
        status = vs.get_status()
        # Also surface configured path for UI counts.
        status.setdefault("persist_directory", cfg.chroma_path)
        return status
    except Exception as e:
        return {
            "ready": False,
            "exists": False,
            "persist_directory": getattr(cfg, "chroma_path", "chroma_db"),
            "chunk_count": None,
            "document_count": None,
            "error": str(e),
        }


def check_llm_status(config=None, llm_provider=None) -> dict:
    """Return real LLM reachability (no hardcoded 'Ready')."""
    from llm_provider import get_llm_provider

    cfg = config or get_config()
    provider = llm_provider or get_llm_provider(cfg)
    try:
        reachable = provider.is_reachable()
    except Exception:
        reachable = False
    return {
        "reachable": reachable,
        "base_url": getattr(cfg, "llm_base_url", ""),
        "model": getattr(cfg, "llm_model", ""),
        "provider": getattr(cfg, "llm_provider", ""),
        "describe": getattr(provider, "describe", "LLM"),
    }
