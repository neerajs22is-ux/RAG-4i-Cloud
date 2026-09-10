"""Session document uploads (Phase 5C).

Browser PDF -> validate -> session-scoped storage -> existing loader ->
existing 1000/200 chunking -> existing MiniLM embeddings -> existing
vector-store indexing with scope="session".

Rules:
  - PDF only (.pdf + %PDF- magic), <=10MB/file, <=5 files/session,
    <=50MB/session. Limits enforced BEFORE expensive processing.
  - Browser filenames are display labels only: object keys and identities
    derive from sanitized names + content hash, never raw paths.
  - document_id is deterministic per (session, content, name): same bytes
    re-uploaded in the same session upsert idempotently; other sessions
    and the persistent corpus can never collide.
  - Ordering is deliberate: validate -> store object -> load/chunk/index
    -> verify vectors -> mark ready. Build failures best-effort delete
    the staged object; partial states are reported, never marked ready.
  - No LLM in ingestion. No contents in results/telemetry (metadata only).
  - Pure helpers wherever possible (fully unit-testable without
    Streamlit, S3, or models).
"""

import hashlib
import logging
import os
import re

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_FILES_PER_SESSION = 5
MAX_SESSION_BYTES = 50 * 1024 * 1024
VERIFY_LIMIT = 1000

READY = "ready"
FAILED = "failed"
ALREADY_INDEXED = "already-indexed"


def sanitize_filename(name: str) -> str:
    """Safe basename (no paths, bounded charset). Extension preserved for
    display; content type is enforced by magic bytes, never the name."""
    base = (name or "").replace("\\", "/").split("/")[-1].strip()
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if len(base) > 100:
        stem, dot, ext = base.rpartition(".")
        base = (stem[:90] + dot + ext[:10]) if dot else base[:100]
    return base or "document"


def session_prefix(session_id: str) -> str:
    """S3 prefix for one session (bucket root, outside the corpus prefix)."""
    from document_scope import validate_session_id
    return f"sessions/{validate_session_id(session_id)}/uploads/"


def session_object_key(session_id: str, safe_name: str) -> str:
    return session_prefix(session_id) + safe_name


def content_sha256_of(data: bytes) -> str:
    return hashlib.sha256(bytes(data or b"")).hexdigest()


def session_document_id(session_id: str, content_sha256: str,
                        safe_name: str) -> str:
    """Deterministic, content-bound, session-namespaced document identity.

    Never based on browser paths. Same (session, bytes, name) always maps
    to the same id (idempotent retry); any change forks deterministically
    instead of colliding (no accidental merge with other sessions or the
    persistent corpus, whose ids are path-derived).
    """
    from document_scope import validate_session_id
    key = ":".join(["session-doc", validate_session_id(session_id),
                    content_sha256 or "", safe_name or ""])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _pdf_diagnosis(data: bytes) -> str:
    """Header-level PDF check: 'ok' | 'not-pdf' | 'encrypted' | 'corrupt'."""
    if not data or data[:5] != b"%PDF-":
        return "not-pdf"
    try:
        from pypdf import PdfReader
        from io import BytesIO
        reader = PdfReader(BytesIO(bytes(data)))
        if getattr(reader, "is_encrypted", False):
            return "encrypted"
        return "ok"
    except Exception:
        return "corrupt"


def validate_upload(data: bytes, display_name: str):
    """Cheap checks first (no loading/embedding). Returns (ok, reason).

    Content type is enforced by PDF magic bytes, never the browser
    filename (display_name is a label only).
    """
    size = len(data or b"")
    if size == 0:
        return False, "Empty file: nothing to index."
    if size > MAX_FILE_BYTES:
        return False, (
            f"File too large ({size // (1024 * 1024)} MB): "
            f"limit is {MAX_FILE_BYTES // (1024 * 1024)} MB per file.")
    diagnosis = _pdf_diagnosis(bytes(data))
    if diagnosis == "not-pdf":
        return False, "Not a PDF file (missing PDF header)."
    if diagnosis == "encrypted":
        return False, "Encrypted/password-protected PDFs are unsupported."
    if diagnosis != "ok":
        return False, "Unreadable PDF (file appears corrupt)."
    return True, ""


def get_session_store(config=None, session_id=None, client=None):
    """Session-scoped DocumentStorage reusing the configured backend.

    S3: same bucket, prefix sessions/<sid>/uploads/ (outside the
    persistent corpus prefix, so session objects never pollute corpus
    listings). Local: a session subdirectory under SESSION_STORAGE_DIR.
    """
    from document_scope import validate_session_id
    from document_storage import LocalDocumentStorage, get_document_storage

    validate_session_id(session_id)
    cfg = config
    if cfg is None:
        from config import get_config
        cfg = get_config()
    mode = (getattr(cfg, "document_storage", "local") or "local").lower()
    if mode == "s3":
        from s3_document_storage import S3DocumentStorage
        bucket = getattr(cfg, "s3_bucket", "") or ""
        if not bucket:
            raise ValueError("S3 bucket is not configured (S3_BUCKET).")
        return S3DocumentStorage(
            bucket=bucket, prefix=session_prefix(session_id),
            region=getattr(cfg, "s3_region", "ap-south-2") or "ap-south-2",
            client=client)
    root = os.path.abspath(getattr(cfg, "session_storage_dir",
                                   "session_uploads") or "session_uploads")
    return LocalDocumentStorage(os.path.join(root, session_id, "uploads"))


def _record_size(record: dict) -> int:
    try:
        return max(0, int(record.get("size_bytes") or 0))
    except (TypeError, ValueError):
        return 0


def reconcile_registry(registry, vector_doc_ids=None):
    """Split registry into (active_records, unknown_vector_count).

    Active = ready, or failed-but-object-retained. Unknown = vector rows
    for this session absent from the registry (e.g. state was reset):
    counted conservatively at MAX_FILE_BYTES each for quota purposes.
    """
    active = [r for r in (registry or [])
              if isinstance(r, dict) and (
                  r.get("status") == READY or (
                      r.get("status") == FAILED
                      and r.get("object_stored", True)))]
    unknown = 0
    if vector_doc_ids is not None:
        known = {r.get("document_id") for r in active
                 if r.get("document_id")}
        unknown = len(set(vector_doc_ids) - known)
    return active, unknown


def check_session_quota(registry, new_size_bytes, vector_doc_ids=None):
    """Enforce count + byte caps BEFORE expensive processing."""
    active, unknown = reconcile_registry(registry, vector_doc_ids)
    if len(active) + unknown + 1 > MAX_FILES_PER_SESSION:
        return False, (
            f"Session file limit reached ({MAX_FILES_PER_SESSION} files). "
            f"Start a New Chat for a fresh session.")
    total = (sum(_record_size(r) for r in active)
             + unknown * MAX_FILE_BYTES + max(0, int(new_size_bytes or 0)))
    if total > MAX_SESSION_BYTES:
        return False, (
            f"Session storage limit reached "
            f"({MAX_SESSION_BYTES // (1024 * 1024)} MB total). "
            f"Start a New Chat for a fresh session.")
    return True, ""


def find_duplicate(registry, content_sha256):
    """Ready record with identical bytes, if any (retry shortcut)."""
    for r in (registry or []):
        if isinstance(r, dict) and r.get("status") == READY \
                and r.get("content_sha256") == content_sha256:
            return r
    return None


def upsert_registry_record(registry, record):
    """Replace-or-append by document_id (deterministic, no duplicates)."""
    registry = registry if isinstance(registry, list) else []
    for i, r in enumerate(registry):
        if isinstance(r, dict) and r.get("document_id") == record.get(
                "document_id"):
            registry[i] = record
            return registry
    registry.append(record)
    return registry


def _load_staged_pdf(storage, key_or_path, *, display_name, document_id,
                     source_id):
    """Load one staged object through the shared PyPDFLoader primitive."""
    import shutil
    import tempfile

    from document_loader import load_pdf_file

    data = storage.get_document(key_or_path)
    workdir = tempfile.mkdtemp(prefix="sessup_")
    try:
        local_path = os.path.join(workdir, "upload.pdf")
        with open(local_path, "wb") as f:
            f.write(data)
        docs = load_pdf_file(local_path)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    for doc in docs:
        page = doc.metadata.get("page", None)
        if not isinstance(page, int):
            page = None
        doc.metadata["source_path"] = source_id
        doc.metadata["source"] = source_id
        doc.metadata["file_name"] = display_name
        doc.metadata["page"] = page
        doc.metadata["document_id"] = document_id
    return docs


def _failed_result(display_name, size, safe_name, sha, document_id, reason,
                   registry, object_stored=False):
    record = {"display_name": display_name, "safe_name": safe_name,
              "size_bytes": size, "content_sha256": sha,
              "document_id": document_id, "status": FAILED,
              "chunks_indexed": 0, "error": reason,
              "object_stored": object_stored}
    if isinstance(registry, list) and document_id:
        upsert_registry_record(registry, record)
    return {"status": FAILED, "display_name": display_name,
            "safe_name": safe_name, "size_bytes": size,
            "content_sha256": sha, "document_id": document_id,
            "chunks_indexed": 0, "error": reason}


def ingest_session_upload(data, display_name, session_id, *, config=None,
                          storage=None, vector_store=None, registry=None,
                          on_progress=None):
    """Validate -> store object -> load/chunk/embed/index -> verify.

    Returns a metadata-only result dict (never document contents).
    Partial states are explicit (failed, never ready); retrying identical
    bytes resolves to the same IDs (idempotent, no duplicate vectors).
    """
    from chunking import chunk_documents
    from config import get_config
    from document_scope import validate_session_id
    from embeddings import get_embedding_provider
    from vector_store import get_vector_store

    validate_session_id(session_id)
    size = len(data or b"")
    display = sanitize_filename(display_name)
    ok, reason = validate_upload(data, display)
    if not ok:
        logger.warning("Session upload rejected (%s): %s", display, reason)
        return _failed_result(display, size, None, None, None, reason,
                              registry)
    cfg = config or get_config()
    store = storage or get_session_store(cfg, session_id)
    sha = content_sha256_of(data)
    dup = find_duplicate(registry or [], sha)
    if dup is not None:
        logger.info("Session upload already indexed (%s).", display)
        result = dict(dup)
        result["status"] = ALREADY_INDEXED
        return result
    vs = vector_store
    if vs is None:
        vs = get_vector_store(cfg, get_embedding_provider(cfg))
    try:
        vector_ids = {s.get("document_id") for s in vs.list_sources(
            limit=1000, session_id=session_id) if s.get("document_id")}
    except TypeError:
        raise ValueError(
            "Session uploads need a scope-aware vector store.")
    except Exception as e:
        logger.warning("Session vector listing failed: %s", e)
        vector_ids = None
    ok, reason = check_session_quota(registry or [], size, vector_ids)
    if not ok:
        logger.warning("Session upload quota rejected (%s): %s", display, reason)
        return _failed_result(display, size, None, sha, None, reason,
                              registry)
    safe = re.sub(r"\.pdf$", "", display, flags=re.IGNORECASE)[:80]
    safe = (safe or "document") + "-" + sha[:12] + ".pdf"
    document_id = session_document_id(session_id, sha, safe)
    if on_progress is not None:
        on_progress(1, 4, display, "validated")
    try:
        key_or_path = store.save_document(safe, bytes(data))
    except Exception as e:
        logger.warning("Session object store failed (%s): %s", display, e)
        return _failed_result(display, size, safe, sha, document_id,
                              f"Could not store the upload: {e}", registry)
    if on_progress is not None:
        on_progress(2, 4, display, "stored")
    try:
        source_id = _source_id_for(store, session_id, safe)
        docs = _load_staged_pdf(store, key_or_path, display_name=display,
                                document_id=document_id, source_id=source_id)
        chars = sum(len(getattr(d, "page_content", "") or "") for d in docs)
        if not docs or chars == 0:
            raise ValueError(
                "No extractable text (scanned image or empty PDF?).")
        chunks = chunk_documents(docs)
        if not chunks:
            raise ValueError("No extractable text.")
        if on_progress is not None:
            on_progress(3, 4, display, "chunked")
        stored = vs.build_index(chunks, session_id=session_id)
        verify = [d for d, _s in vs.chunks_for_source(
            display, limit=VERIFY_LIMIT, session_id=session_id)
            if (d.metadata or {}).get("document_id") == document_id]
        if not verify:
            return _failed_result(
                display, size, safe, sha, document_id,
                "Stored, but no vectors were indexed for this document.",
                registry, object_stored=True)
        if on_progress is not None:
            on_progress(4, 4, display, "indexed")
    except Exception as e:
        logger.warning("Session ingest failed (%s): %s", display, e)
        try:
            store.delete_document(key_or_path)
            object_stored = False
        except Exception as de:
            logger.warning("Session object cleanup failed (%s): %s",
                           display, de)
            object_stored = True
        return _failed_result(display, size, safe, sha, document_id,
                              f"Could not index the upload: {e}", registry,
                              object_stored=object_stored)
    record = {"display_name": display, "safe_name": safe,
              "size_bytes": size, "content_sha256": sha,
              "document_id": document_id, "status": READY,
              "chunks_indexed": stored, "error": "",
              "object_stored": True}
    if isinstance(registry, list):
        upsert_registry_record(registry, record)
    logger.info("session upload ready files=1 bytes=%d indexed=%d",
                size, stored)
    result = dict(record)
    return result


def _source_id_for(store, session_id, safe_name):
    """Stable source identifier (never a browser path, never a tmp path)."""
    bucket = getattr(store, "bucket", "")
    if bucket:
        prefix = getattr(store, "prefix", "") or ""
        key = f"{prefix}{safe_name}" if prefix else safe_name
        return f"s3://{bucket}/{key}"
    return f"session://{session_id}/uploads/{safe_name}"


def ensure_session_id(state) -> str:
    """Return a valid current session id, minting + resetting on damage."""
    from document_scope import is_valid_session_id, new_session_id

    sid = state.get("session_id") if isinstance(state, dict) else None
    docs = state.get("session_docs") if isinstance(state, dict) else None
    if not isinstance(docs, list):
        docs = []
        if isinstance(state, dict):
            state["session_docs"] = docs
    if not is_valid_session_id(sid):
        if docs:
            logger.info("Dropping %d registry entries with invalid "
                        "session binding.", len(docs))
            docs.clear()
        sid = new_session_id()
        if isinstance(state, dict):
            state["session_id"] = sid
    return sid


def rotate_session_id(state) -> str:
    """New-Chat detach: fresh binding, cleared registry, vectors untouched
    (no physical deletion — retention is a later concern)."""
    from document_scope import new_session_id

    sid = new_session_id()
    if isinstance(state, dict):
        state["session_id"] = sid
        state["session_docs"] = []
    return sid
