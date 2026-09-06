"""PDF locate/load step of ingestion (PyPDFLoader, unchanged behaviour)."""

import hashlib
import os
from typing import Dict, List, Tuple


def stable_document_id(source_path: str) -> str:
    """Stable ID for the same source document (absolute-path hash)."""
    abs_path = os.path.abspath(source_path)
    return hashlib.sha1(abs_path.encode("utf-8")).hexdigest()


def standardize_document_metadata(source_path: str, page, extra=None) -> Dict:
    """Build standard metadata. Page is preserved as-is or None."""
    abs_path = os.path.abspath(source_path)
    metadata = {
        "source_path": abs_path,
        # Back-compat: original code used `source`.
        "source": abs_path,
        "file_name": os.path.basename(abs_path),
        "page": page if isinstance(page, int) else None,
        "document_id": stable_document_id(abs_path),
    }
    if extra:
        metadata.update(extra)
    return metadata


def load_pdf_file(pdf_path: str):
    """Load a single PDF via PyPDFLoader with standardized metadata.

    Raises on failure so callers can record per-file errors.
    """
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    abs_path = os.path.abspath(pdf_path)
    doc_id = stable_document_id(abs_path)
    file_name = os.path.basename(abs_path)
    for doc in docs:
        page = doc.metadata.get("page", None)
        if not isinstance(page, int):
            page = None
        doc.metadata["source_path"] = abs_path
        doc.metadata["source"] = abs_path
        doc.metadata["file_name"] = file_name
        doc.metadata["page"] = page
        doc.metadata["document_id"] = doc_id
    return docs


def load_documents_from_folder(folder_path: str, storage=None):
    """Load all PDFs under folder_path.

    Returns (documents, report) where report has:
        found, succeeded, failed, failed_files
    A single PDF failure never aborts the whole run.
    """
    from document_storage import LocalDocumentStorage

    if storage is None:
        storage = LocalDocumentStorage(folder_path)

    pdf_files = storage.list_documents("**/*.pdf")
    report = {
        "found": len(pdf_files),
        "succeeded": 0,
        "failed": 0,
        "failed_files": [],
    }
    documents: List = []
    for pdf_file in pdf_files:
        try:
            docs = load_pdf_file(pdf_file)
            documents.extend(docs)
            report["succeeded"] += 1
        except Exception as e:
            report["failed"] += 1
            report["failed_files"].append(os.path.basename(pdf_file))
            print(f"Error loading {pdf_file}: {e}")
            continue
    return documents, report
