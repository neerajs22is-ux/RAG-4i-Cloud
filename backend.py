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

LOW_RELEVANCE_MESSAGE = (
    "I could not find enough relevant information in the documents to answer that."
)


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


def retrieve_documents(query_text, config=None, vector_store=None, k=None,
                       threshold=None):
    """Search the vector store; return structured sources (filtered by threshold)."""
    from embeddings import get_embedding_provider
    from vector_store import get_vector_store

    cfg = config or get_config()
    top_k = k if k is not None else getattr(cfg, "retrieval_k", 5)
    thresh = threshold if threshold is not None else getattr(cfg, "relevance_threshold", 0.3)

    if vector_store is None:
        embedding_provider = get_embedding_provider(cfg)
        vector_store = get_vector_store(cfg, embedding_provider)

    # Raises if DB missing/unavailable -> caller maps to user message.
    results = vector_store.search(query_text, k=top_k)
    structured = [_to_structured_source(doc, score) for doc, score in results]
    # Preserve original behaviour: relevance threshold 0.3.
    filtered = [s for s in structured if s["score"] is not None and s["score"] >= thresh]
    return filtered


# ---------- Generation ---------- #

def generate_answer(question, retrieved_sources, config=None, llm_provider=None,
                    prompt_template=None):
    """Invoke the LLM over retrieved context; return answer string."""
    from llm_provider import get_llm_provider

    cfg = config or get_config()
    template = prompt_template or PROMPT_TEMPLATE
    if not retrieved_sources:
        return LOW_RELEVANCE_MESSAGE
    context_text = "\n\n---\n\n".join([s.get("content", "") for s in retrieved_sources])
    provider = llm_provider or get_llm_provider(cfg)
    try:
        return provider.generate(context_text, question, template)
    except Exception as e:
        print(f"LLM generation failed: {e}")
        raise ConnectionError(
            "LLM endpoint is unreachable. Make sure LM Studio Server is running "
            f"at {getattr(cfg, 'llm_base_url', 'http://localhost:1234/v1')}."
        )


def query_documents(query_text, config=None, vector_store=None, llm_provider=None):
    """Combined retrieve + generate (kept for UI compat).

    Returns (answer: str, sources: list[dict]).
    """
    cfg = config or get_config()
    try:
        retrieved = retrieve_documents(
            query_text, config=cfg, vector_store=vector_store
        )
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return (
            "The knowledge base is unavailable. Please build the index first "
            "and check that the vector database is accessible.",
            [],
        )
    if not retrieved:
        return LOW_RELEVANCE_MESSAGE, []
    try:
        answer = generate_answer(
            query_text, retrieved, config=cfg, llm_provider=llm_provider
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
