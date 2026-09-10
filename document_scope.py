"""Document scope primitives (Phase 5B).

Every indexed chunk carries explicit identity/session/document scope:

    identity -> session -> document -> chunk

Rules (see docs/PHASE_5_SESSION_SCOPE.md):
  - persistent corpus: scope="persistent", no session_id.
  - session corpus: scope="session" plus a REQUIRED opaque 128-bit id.
  - Session IDs are random (secrets.token_hex(16)): never derived from
    paths, filenames, user identity, or document content, and never a
    raw filesystem path.
  - chunk_id derivation is unchanged (content-hash) and MUST NOT include
    scope: re-indexing the same content yields the same IDs (idempotent
    upserts, no duplicates).
  - All rejections are loud ValueErrors: malformed IDs, session writes
    without a session, contradictions, unknown scope values. Nothing
    degrades silently to unscoped access.

Pure module: no Streamlit, no providers, no I/O. Fully unit-testable.
"""

import re
import secrets

PERSISTENT = "persistent"
SESSION = "session"

_VALID_SCOPES = frozenset({PERSISTENT, SESSION})

# Opaque 128-bit session id: 32 lowercase hex chars (secrets.token_hex(16)).
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def new_session_id() -> str:
    """Create a fresh opaque 128-bit session identifier."""
    return secrets.token_hex(16)


def is_valid_session_id(session_id) -> bool:
    """True only for 32-char lowercase hex strings (nothing derived)."""
    return isinstance(session_id, str) and bool(
        _SESSION_ID_RE.match(session_id))


def validate_session_id(session_id) -> str:
    """Return the id or raise ValueError (never coerce, never default)."""
    if not is_valid_session_id(session_id):
        raise ValueError(
            "Invalid session_id: expected an opaque 32-char hex id "
            "(see new_session_id()); never derive one from paths, "
            "filenames, user identity, or document content.")
    return session_id


def resolve_chunk_scope(metadata, session_id=None):
    """Determine (scope, session_id|None) for one chunk write.

    session_id is the BATCH binding (e.g. the uploading session) or None
    for persistent writes. Pre-existing chunk metadata must not
    contradict it. Returns (PERSISTENT, None) or (SESSION, sid).
    Raises ValueError on: malformed batch session_id, unknown scope
    value, session scope without any session id, or contradictory
    batch/metadata combinations.
    """
    meta = metadata or {}
    meta_scope = meta.get("scope")
    meta_sid = meta.get("session_id") or None
    if session_id is None:
        if meta_scope is None:
            return PERSISTENT, None
        if meta_scope == PERSISTENT:
            if meta_sid is not None:
                raise ValueError(
                    "Contradictory scope metadata: scope='persistent' "
                    "with a session_id present.")
            return PERSISTENT, None
        if meta_scope == SESSION:
            if meta_sid is None:
                raise ValueError(
                    "Refusing session-scoped write without a session_id.")
            return SESSION, validate_session_id(meta_sid)
        raise ValueError(
            f"Unknown scope value: {meta_scope!r} "
            f"(expected {PERSISTENT!r} or {SESSION!r}).")
    sid = validate_session_id(session_id)
    if meta_scope is None:
        return SESSION, sid
    if meta_scope == SESSION:
        if meta_sid is None:
            raise ValueError(
                "Refusing session-scoped write without a session_id.")
        if meta_sid != sid:
            raise ValueError(
                "Refusing cross-session write: chunk metadata names a "
                "different session_id than the batch binding.")
        return SESSION, sid
    if meta_scope == PERSISTENT:
        raise ValueError(
            "Contradictory scope metadata: persistent chunk in a "
            "session-scoped write batch.")
    raise ValueError(
        f"Unknown scope value: {meta_scope!r} "
        f"(expected {PERSISTENT!r} or {SESSION!r}).")
