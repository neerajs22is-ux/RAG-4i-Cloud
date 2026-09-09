"""Unified readiness model for the pilot (Phase 4.11).

Single UI-facing representation built from REAL existing checks only:
  - model_state: result of model_warmup.warmup() ({"state": "ready"|...})
  - kb_status: result of backend.get_knowledge_base_status()

No new backend checks, no duplicated logic. Pure functions so AppTest
and unit tests can cover every state without Streamlit or models.

States:
  READY            model ready + KB ready
  MODEL_NOT_READY  KB ready, model not
  KB_NOT_READY     model ready, KB not (not built / empty)
  BOTH_NOT_READY   neither ready
  DEGRADED         KB check itself reported an error (vector DB
                   unavailable) — infrastructure, not just "empty".
"""

READY = "READY"
MODEL_NOT_READY = "MODEL_NOT_READY"
KB_NOT_READY = "KB_NOT_READY"
BOTH_NOT_READY = "BOTH_NOT_READY"
DEGRADED = "DEGRADED"

_ALL_STATES = (READY, MODEL_NOT_READY, KB_NOT_READY, BOTH_NOT_READY, DEGRADED)


def is_model_ready(model_state) -> bool:
    """True only for the real warmup READY shape."""
    return isinstance(model_state, dict) and model_state.get("state") == "ready"


def is_kb_ready(kb_status) -> bool:
    return bool((kb_status or {}).get("ready"))


def summarize_readiness(model_state, kb_status) -> dict:
    """One readiness dict from existing state. Never probes anything.

    Returns {"state": ..., "model_ready": bool, "kb_ready": bool,
             "kb_error": str|None, "kb_exists": bool, ...}.
    """
    kb = kb_status or {}
    model_ok = is_model_ready(model_state)
    kb_ok = bool(kb.get("ready"))
    kb_error = kb.get("error")
    kb_exists = bool(kb.get("exists"))
    if kb_error:
        state = DEGRADED
    elif model_ok and kb_ok:
        state = READY
    elif (not model_ok) and (not kb_ok):
        state = BOTH_NOT_READY
    elif not model_ok:
        state = MODEL_NOT_READY
    else:
        state = KB_NOT_READY
    return {
        "state": state,
        "model_ready": model_ok,
        "kb_ready": kb_ok,
        "kb_error": kb_error,
        "kb_exists": kb_exists,
        "chunk_count": kb.get("chunk_count"),
        "document_count": kb.get("document_count"),
    }


def checklist_items(readiness: dict, kb_status=None, is_cloud: bool = False) -> list:
    """State-aware first-run checklist rows.

    Each row: {"label": str, "done": bool, "hint": str|None}.
    Uses only the readiness dict (no extra checks).
    """
    kb = kb_status or {}
    model_ok = bool(readiness.get("model_ready"))
    kb_ok = bool(readiness.get("kb_ready"))
    docs = kb.get("document_count")
    if kb_ok and isinstance(docs, int):
        kb_label = f"Knowledge base ready ({docs} document{'s' if docs != 1 else ''} indexed)"
    elif kb_ok:
        kb_label = "Knowledge base ready"
    else:
        kb_label = "Knowledge base not built"
    where = "S3 bucket" if is_cloud else "sidebar"
    return [
        {"label": "Model ready" if model_ok else "Model not ready",
         "done": model_ok,
         "hint": None if model_ok else "Wait for warmup or use Retry connection."},
        {"label": kb_label,
         "done": kb_ok,
         "hint": None if kb_ok else f"Build the knowledge base in the {where} →"},
    ]


def composer_state(readiness: dict) -> dict:
    """Disabled state for the chat composer. Returns {"disabled": bool, "reason": str|None}."""
    state = readiness.get("state")
    if state == READY:
        return {"disabled": False, "reason": None}
    if state == MODEL_NOT_READY:
        return {"disabled": True,
                "reason": "Chat is disabled until the assistant model is ready. "
                          "Use “Retry connection” above if this persists."}
    if state == DEGRADED:
        return {"disabled": True,
                "reason": "Chat is unavailable: the vector database reported an error. "
                          "Check the app logs and that the vector store is accessible, then retry."}
    # KB_NOT_READY / BOTH_NOT_READY
    return {"disabled": True,
            "reason": "Chat is disabled until the knowledge base is built. "
                      "Use “Build/Update Database” in the Knowledge base panel first."}


def empty_state_steps(is_cloud: bool, bucket: str = "", prefix: str = "") -> list:
    """Three guided steps for the empty/not-built KB state.

    Adapts wording to the actual local/S3 mode and available controls.
    Does not invent controls: local uses the folder-path + Build button,
    S3 uses the S3 Build button.
    """
    if is_cloud:
        src = f"s3://{bucket}/{prefix}" if bucket else "the configured S3 bucket/prefix"
        step1 = (f"1. Add PDFs to {src}, then press "
                 f"“Build/Update Database from S3” in the sidebar Knowledge base panel.")
    else:
        step1 = ("1. Add PDFs to a folder on the app host, enter that folder path "
                 "in the sidebar Knowledge base panel, then press “Build/Update Database”.")
    return [
        step1,
        "2. Wait for “Knowledge Base: Ready” in the sidebar status.",
        "3. Ask a question below — answers cite filename, page, and relevance.",
    ]


def recovery_message(readiness: dict, kb_status=None, model_state=None,
                     is_cloud: bool = False) -> dict:
    """Actionable recovery copy for known conditions.

    Returns {"title": str, "what": str, "next": str, "retry": bool}.
    Coding errors (unexpected exceptions) must NOT be routed here —
    callers keep logging those and showing friendly_error().
    """
    state = readiness.get("state")
    kb = kb_status or {}
    if state == READY:
        return {"title": "Ready",
                "what": "Model and knowledge base are both ready.",
                "next": "Ask a question below.",
                "retry": False}
    if state == DEGRADED:
        detail = (kb.get("error") or "").strip()
        what = "The vector database check reported an error."
        if detail:
            # Never leak paths/secrets: keep only the first short fragment.
            what += f" Details: {detail[:160]}"
        if is_cloud:
            nxt = ("Check that the vector store is reachable (Chroma path or "
                   "RDS tunnel), then rebuild or retry. See app logs for details.")
        else:
            nxt = ("Check that chroma_db is writable and dependencies are installed, "
                   "then rebuild the index. See app logs for details.")
        return {"title": "Knowledge base unavailable",
                "what": what, "next": nxt, "retry": True}
    if state == MODEL_NOT_READY:
        detail = ""
        if isinstance(model_state, dict) and model_state.get("detail"):
            detail = str(model_state["detail"])
        what = detail or "The assistant model is not ready."
        nxt = ("Make sure the embedding model can load and LM Studio Server is running, "
               "then use “Retry connection”.")
        return {"title": "Model not ready", "what": what, "next": nxt, "retry": True}
    if state in (KB_NOT_READY, BOTH_NOT_READY):
        if not readiness.get("model_ready") and not readiness.get("kb_ready"):
            title = "Model and knowledge base not ready"
            what = "The assistant model is not reachable and no index is built yet."
            nxt = "Build the knowledge base in the sidebar, then use “Retry connection”."
            return {"title": title, "what": what, "next": nxt, "retry": True}
        exists = bool(kb.get("exists"))
        if exists:
            what = "The knowledge base exists but has no indexed chunks."
        else:
            what = "The knowledge base has not been built yet."
        nxt = ("Use “Build/Update Database”"
               + (" from S3" if is_cloud else "")
               + " in the sidebar Knowledge base panel, then ask again.")
        return {"title": "Knowledge base not ready",
                "what": what, "next": nxt, "retry": False}
    return {"title": "Not ready",
            "what": "The assistant is not ready yet.",
            "next": "Wait a moment and retry.",
            "retry": True}
