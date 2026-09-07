"""Refactored backend: ingestion + retrieval + generation via providers.

Preserved behaviour:
- PyPDFLoader, chunk 1000/200, all-MiniLM-L6-v2, top-k=5, threshold 0.3,
  prompt template unchanged.
- No destructive rebuild (existing Chroma data is kept).
"""

import os

from config import get_config

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

# Support levels (see answer_support.py); OVERVIEW shares the scoping prompt.
PARTIAL_SUPPORT = "partial"
OVERVIEW_SUPPORT = "overview"


# ---------- Ingestion ---------- #

def _looks_like_windows_path(path: str) -> bool:
    """True for 'C:\\...' / 'C:/...' style paths on a non-Windows host."""
    import re
    return os.name != "nt" and bool(re.match(r"^[A-Za-z]:[\\/]", path or ""))


def ingest_with_report(folder_path, config=None,
                       storage=None, vector_store=None):
    """Build/extend the index, returning (success, message, details).

    details always contains: source (local|s3), found, pages, chars,
    chunks, embeddings, vectors_stored, succeeded, failed, failed_files
    (list of {file, reason}). Zero/empty outcomes are explicit, never masked.
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
            documents, report = load_documents_from_s3(storage)
        except Exception as e:
            print(f"S3 ingestion failed: {e}")
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
            documents, report = load_documents_from_folder(folder_path, storage)
        except Exception as e:
            print(f"Ingestion failed: {e}")
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
        print(f"Chunking failed: {e}")
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
        print(f"Vector store build failed: {e}")
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
    """
    forms = []
    for form in (normalize_query(query_text), query_core(query_text)):
        if form and form not in forms:
            forms.append(form)
    return forms or [query_text]


def retrieve_documents(query_text, config=None, vector_store=None, k=None,
                       threshold=None):
    """Search the vector store; return structured sources (filtered by threshold).

    Tries each deterministic query form and keeps each chunk's best score
    (max-pooling), then applies the unchanged top-k/threshold rule.
    """
    from embeddings import get_embedding_provider
    from vector_store import get_vector_store

    cfg = config or get_config()
    top_k = k if k is not None else getattr(cfg, "retrieval_k", 5)
    thresh = threshold if threshold is not None else getattr(cfg, "relevance_threshold", 0.3)

    if vector_store is None:
        embedding_provider = get_embedding_provider(cfg)
        vector_store = get_vector_store(cfg, embedding_provider)

    # Raises if DB missing/unavailable -> caller maps to user message.
    best = {}
    order = []
    for form in build_query_forms(query_text):
        for doc, score in vector_store.search(form, k=top_k):
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
    if prompt_template is not None:
        template = prompt_template
    elif support_level in (PARTIAL_SUPPORT, OVERVIEW_SUPPORT):
        template = PARTIAL_PROMPT_TEMPLATE
    else:
        template = PROMPT_TEMPLATE
    if not retrieved_sources:
        return LOW_RELEVANCE_MESSAGE
    context_text = "\n\n---\n\n".join([s.get("content", "") for s in retrieved_sources])
    provider = llm_provider or get_llm_provider(cfg)
    try:
        raw = provider.generate(context_text, question, template)
    except Exception as e:
        print(f"LLM generation failed: {e}")
        raise ConnectionError(
            "LLM endpoint is unreachable. Make sure LM Studio Server is running "
            f"at {getattr(cfg, 'llm_base_url', 'http://localhost:1234/v1')}."
        )
    answer = strip_think_blocks(raw)
    if is_empty_response(raw):
        print("LLM returned only hidden reasoning or empty text.")
        return ("The model returned an empty response. "
                "Please try again.")
    return answer


def query_documents(query_text, config=None, vector_store=None, llm_provider=None,
                    conversation_context=None, on_phase=None):
    """Combined retrieve + generate (kept for UI compat).

    The intent observer routes first: conversation/capability/out-of-scope
    are answered directly without retrieval; document intents (including
    followups, whose query is anchored to the prior question) use the full
    RAG pipeline unchanged. Returns (answer: str, sources: list[dict]).

    on_phase(name) is an optional progress hook ("retrieving" /
    "generating") for honest staged loading states. It never changes
    behavior and defaults to off.
    """
    from query_router import (CAPABILITY, CONVERSATION, DOCUMENT_FOLLOWUP,
                              OUT_OF_SCOPE, expand_followup_query,
                              observe_query, reply_for_route)

    obs = observe_query(query_text, conversation_context)
    if obs.intent in (CONVERSATION, CAPABILITY, OUT_OF_SCOPE):
        return reply_for_route(query_text, obs.intent), []
    cfg = config or get_config()
    retrieval_query = query_text
    if obs.intent == DOCUMENT_FOLLOWUP:
        retrieval_query = expand_followup_query(query_text, conversation_context)
    from answer_support import (DIRECT, PARTIAL, UNSUPPORTED,
                                assess_support, detect_broad_scope,
                                unsupported_reply)
    from embeddings import get_embedding_provider
    from vector_store import get_vector_store

    if vector_store is None:
        vector_store = get_vector_store(
            cfg, get_embedding_provider(cfg))
    broad, target_file = detect_broad_scope(query_text)
    if on_phase is not None:
        on_phase("retrieving")
    try:
        retrieved = retrieve_documents(
            retrieval_query, config=cfg, vector_store=vector_store
        )
        if broad and target_file:
            # Source-aware top-up: same-file chunks as admissible file
            # evidence (threshold still guards query-similarity results).
            seen = {s.get("chunk_id") for s in retrieved}
            for doc, _score in vector_store.chunks_for_source(target_file):
                struct = _to_structured_source(doc, None)
                if struct.get("chunk_id") not in seen:
                    seen.add(struct.get("chunk_id"))
                    retrieved.append(struct)
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return (
            "The knowledge base is unavailable. Please build the index first "
            "and check that the vector database is accessible.",
            [],
        )
    if not retrieved:
        return LOW_RELEVANCE_MESSAGE, []
    support = assess_support(retrieval_query, retrieved)
    if support["level"] == UNSUPPORTED:
        return unsupported_reply(query_text), retrieved
    level = OVERVIEW_SUPPORT if broad else (
        None if support["level"] == DIRECT else PARTIAL_SUPPORT)
    if on_phase is not None:
        on_phase("generating")
    try:
        answer = generate_answer(
            query_text, retrieved, config=cfg, llm_provider=llm_provider,
            support_level=level,
        )
    except ConnectionError as e:
        return str(e), retrieved
    except Exception as e:
        print(f"Query failed: {e}")
        return (
            "An error occurred while generating the answer. "
            "Make sure LM Studio Server is running.",
            retrieved,
        )
    return answer, retrieved


def classify_query(query_text) -> str:
    """Routing label for a message (for UI display; routing is deterministic)."""
    from query_router import route_query

    return route_query(query_text)


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
