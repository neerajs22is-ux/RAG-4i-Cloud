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


def _s3_source_id(bucket: str, key: str) -> str:
    """Stable S3 source identifier (never a local temp path)."""
    return f"s3://{bucket}/{key}"


def load_documents_from_s3(storage, tmp_dir=None):
    """Load all PDFs from an S3DocumentStorage via temp downloads.

    Downloads each object, loads with the existing PyPDFLoader path, then
    rewrites metadata to the stable S3 identifier (page preserved as-is).
    Returns (documents, report) with the same shape as folder loading.
    Temp files are removed best-effort afterwards.
    """
    import hashlib
    import shutil
    import tempfile

    keys = storage.list_documents("**/*.pdf")
    report = {
        "found": len(keys),
        "succeeded": 0,
        "failed": 0,
        "failed_files": [],
    }
    documents: List = []
    workdir = tmp_dir or tempfile.mkdtemp(prefix="s3ingest_")
    try:
        for key in keys:
            try:
                data = storage.get_document(key)
                file_name = key.replace("\\", "/").split("/")[-1]
                local_path = os.path.join(workdir, file_name)
                with open(local_path, "wb") as f:
                    f.write(data)
                docs = load_pdf_file(local_path)
                source_id = _s3_source_id(storage.bucket, key)
                doc_id = hashlib.sha1(source_id.encode("utf-8")).hexdigest()
                for doc in docs:
                    page = doc.metadata.get("page", None)
                    if not isinstance(page, int):
                        page = None
                    doc.metadata["source_path"] = source_id
                    doc.metadata["source"] = source_id
                    doc.metadata["file_name"] = file_name
                    doc.metadata["page"] = page
                    doc.metadata["document_id"] = doc_id
                documents.extend(docs)
                report["succeeded"] += 1
            except Exception as e:
                report["failed"] += 1
                report["failed_files"].append(
                    key.replace("\\", "/").split("/")[-1])
                print(f"Error loading {key}: {e}")
                continue
    finally:
        if tmp_dir is None:
            shutil.rmtree(workdir, ignore_errors=True)
    return documents, report
