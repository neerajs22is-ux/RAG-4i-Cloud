"""Browser-local session restore: pure serialization layer (Phase 4.11).

Scope is ONLY accidental-refresh recovery of the CURRENT conversation:
  - role/content + display metadata (label, sources, strength, guard_note,
    notice, latency/timings, seq)
  - timestamp of the snapshot
  - no credentials, no document corpus, no vectors, no config
  - no server-side storage (no disk/S3/DB writes here)

Browser integration (localStorage read/write) lives in app.py as a thin
one-way JS saver + explicit Restore/Start-fresh choice. This module stays
pure + fully tested: serialize() / deserialize() round-trip without
Streamlit, and corrupt input degrades to empty (never raises to the user).

Privacy: messages may quote legal text; snapshots stay browser-local
(localStorage) and ephemeral. Telemetry never receives message content.
"""

import copy
from datetime import datetime, timezone

SCHEMA_VERSION = 1
_MAX_MESSAGES = 200
_ALLOWED_ROLES = ("user", "assistant")
_MAX_SESSION_DOCS = 10


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_source(src: dict) -> dict:
    """Keep only the structured source fields the UI renders."""
    s = src or {}
    score = s.get("score")
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    page = s.get("page")
    page = page if isinstance(page, int) else None
    return {
        "source": s.get("source"),
        "source_path": s.get("source_path"),
        "file_name": s.get("file_name"),
        "page": page,
        "score": score,
        "document_id": s.get("document_id"),
        "chunk_id": s.get("chunk_id"),
        "content": s.get("content", "") or "",
    }


def _clean_message(msg: dict) -> dict | None:
    if not isinstance(msg, dict):
        return None
    role = msg.get("role")
    if role not in _ALLOWED_ROLES:
        return None
    content = msg.get("content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    # Cap content length defensively (localStorage quota ~5MB).
    if len(content) > 20000:
        content = content[:20000]
    sources = []
    for s in (msg.get("sources") or [])[:10]:
        if isinstance(s, dict):
            sources.append(_clean_source(s))
    out = {
        "role": role,
        "content": content,
        "sources": sources,
    }
    for key in ("label", "notice", "strength", "guard_note",
                "latency_ms", "total_ms", "seq"):
        if msg.get(key) is not None:
            out[key] = msg[key]
    # Timings: small ints only.
    timings = msg.get("timings")
    if isinstance(timings, dict):
        clean_timings = {}
        for k in ("retrieval_ms", "support_ms", "preparation_ms",
                  "generation_ms", "total_ms"):
            v = timings.get(k)
            if isinstance(v, (int, float)):
                clean_timings[k] = max(0, int(v))
        if clean_timings:
            out["timings"] = clean_timings
    # seq: keep ints only.
    if "seq" in out and not isinstance(out["seq"], int):
        try:
            out["seq"] = int(out["seq"])
        except (TypeError, ValueError):
            out.pop("seq", None)
    return out


def _clean_session_doc(entry: dict | None) -> dict | None:
    """Keep upload-registry metadata only (never contents, no secrets)."""
    if not isinstance(entry, dict):
        return None
    try:
        size = max(0, int(entry.get("size_bytes") or 0))
    except (TypeError, ValueError):
        size = 0
    try:
        chunks = max(0, int(entry.get("chunks_indexed") or 0))
    except (TypeError, ValueError):
        chunks = 0
    out = {
        "display_name": str(entry.get("display_name") or "")[:120],
        "safe_name": str(entry.get("safe_name") or "")[:140],
        "size_bytes": size,
        "content_sha256": str(entry.get("content_sha256") or "")[:64],
        "document_id": str(entry.get("document_id") or "")[:64],
        "status": entry.get("status") if entry.get("status") in (
            "ready", "failed", "already-indexed") else "failed",
        "chunks_indexed": chunks,
        "error": str(entry.get("error") or "")[:300],
    }
    return out


def _clean_session_block(session) -> dict | None:
    if not isinstance(session, dict):
        return None
    from document_scope import is_valid_session_id
    sid = session.get("id")
    if not is_valid_session_id(sid):
        return None
    docs = []
    for entry in (session.get("documents") or [])[:_MAX_SESSION_DOCS]:
        cleaned = _clean_session_doc(entry)
        if cleaned is not None:
            docs.append(cleaned)
    return {"id": sid, "documents": docs}


def serialize_conversation(messages, saved_at: str | None = None,
                           session=None) -> dict:
    """Snapshot the current conversation (pure, no I/O).

    session is an optional {"id": session_id, "documents": registry}
    block rebound on restore so session uploads stay usable after an
    accidental refresh. Old snapshots without it still load.
    """
    clean = []
    for m in (messages or [])[-_MAX_MESSAGES:]:
        c = _clean_message(m)
        if c is not None:
            clean.append(c)
    out = {
        "schema": SCHEMA_VERSION,
        "saved_at": saved_at or _utc_now_iso(),
        "messages": clean,
    }
    block = _clean_session_block(session)
    if block is not None:
        out["session"] = block
    return out


def deserialize_conversation(data) -> dict:
    """Validate a snapshot. Returns {"messages": [...], "saved_at": str|None, "ok": bool}.

    Never raises on corrupt input: returns ok=False + empty messages.
    Drops unknown/sketchy fields; keeps only renderable message data.
    """
    if not isinstance(data, dict):
        return {"messages": [], "saved_at": None, "ok": False}
    if data.get("schema") != SCHEMA_VERSION:
        # Forward-tolerance: accept missing schema only if messages look sane?
        # No — strict: unknown schema degrades to empty (never misrender).
        if data.get("schema") is not None:
            return {"messages": [], "saved_at": None, "ok": False}
    raw = data.get("messages")
    if not isinstance(raw, list):
        return {"messages": [], "saved_at": None, "ok": False}
    clean = []
    for m in raw[:_MAX_MESSAGES]:
        c = _clean_message(m)
        if c is not None:
            clean.append(c)
    saved_at = data.get("saved_at")
    if not isinstance(saved_at, str):
        saved_at = None
    out = {"messages": clean, "saved_at": saved_at, "ok": True}
    block = _clean_session_block(data.get("session"))
    if block is not None:
        out["session"] = block
    return out


def should_offer_restore(current_messages, snapshot) -> bool:
    """Offer restore only when there is something to restore into an empty chat."""
    if current_messages:
        return False
    if not isinstance(snapshot, dict) or not snapshot.get("ok"):
        return False
    return bool(snapshot.get("messages"))


def restore_messages_into_state(state, snapshot) -> int:
    """Copy validated snapshot messages into a session-state-like dict.

    Returns number of messages restored. Re-seeds msg_seq so later IDs
    stay stable and unique. Never touches providers/config/vectors.
    """
    msgs = list((snapshot or {}).get("messages") or [])
    state["messages"] = copy.deepcopy(msgs)
    max_seq = 0
    for m in msgs:
        try:
            s = int(m.get("seq") or 0)
        except (TypeError, ValueError):
            s = 0
        max_seq = max(max_seq, s)
    state["msg_seq"] = max(int(state.get("msg_seq") or 0), max_seq)
    # Restored turns retire stale follow-up buttons; they rebuild on next answer.
    # The source page viewer is UI-only state pointing at old seqs: drop it.
    state.pop("last_followups", None)
    state.pop("pending_prompt", None)
    state.pop("last_failed", None)
    state.pop("open_source", None)
    return len(msgs)


def restore_session_into_state(state, snapshot) -> bool:
    """Rebind a validated snapshot session block (same-browser lineage).

    Sets session_id + session_docs so uploads stay usable after an
    accidental refresh. Returns True when adopted, False otherwise
    (callers then mint a fresh session). Never touches vectors/config.
    """
    block = _clean_session_block((snapshot or {}).get("session"))
    if block is None:
        return False
    state["session_id"] = block["id"]
    state["session_docs"] = block["documents"]
    return True


def clear_restore_state(state) -> None:
    """New Chat / Start fresh: drop conversation-only restore crumbs."""
    state.pop("_restore_dismissed", None)
    state.pop("_restore_snapshot", None)
    # NOTE: feedback clearing is owned by the New-Chat audit; kept here
    # only for restore crumbs so responsibilities stay separate.
