"""Chunking step (RecursiveCharacterTextSplitter, 1000/200 unchanged)."""

import hashlib
from typing import List

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_text_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )


def chunk_documents(documents: List, chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP):
    """Split documents and attach stable chunk_id metadata.

    chunk_id = sha1(document_id + start_index + index) so it is unique
    per chunk while document_id stays stable per source file.
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        doc_id = meta.get("document_id", "unknown")
        start = meta.get("start_index", i)
        raw = f"{doc_id}:{start}:{i}".encode("utf-8")
        chunk.metadata["chunk_id"] = hashlib.sha1(raw).hexdigest()[:16]
        # Ensure standard keys always exist.
        chunk.metadata.setdefault("source_path", chunk.metadata.get("source"))
        chunk.metadata.setdefault("source", chunk.metadata.get("source_path"))
        if "file_name" not in chunk.metadata:
            import os
            src = chunk.metadata.get("source_path") or chunk.metadata.get("source") or ""
            chunk.metadata["file_name"] = os.path.basename(src) if src else "unknown"
        chunk.metadata.setdefault("page", None)
        chunk.metadata.setdefault("document_id", doc_id)
    return chunks
