"""Copy/additive migration: PDFs -> loader -> chunker -> embedder -> PostgreSQL.

Never touches the Chroma database (no delete/overwrite/rebuild of chroma_db).

Usage:
    python migrate_to_postgres.py <pdf-folder> [--table chunks]

Retained behavior: same loader, chunk_size=1000, chunk_overlap=200,
same all-MiniLM-L6-v2 embeddings as Chroma.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config  # noqa: E402


def migrate(folder, table="chunks"):
    from chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_documents
    from document_loader import load_documents_from_folder
    from document_storage import LocalDocumentStorage
    from embeddings import get_embedding_provider
    from postgres_vector_store import get_postgres_store

    cfg = get_config()
    storage = LocalDocumentStorage(folder)
    documents, report = load_documents_from_folder(folder, storage)
    print(f"Found={report['found']} ok={report['succeeded']} "
          f"failed={report['failed']} {report['failed_files']}")
    if not documents:
        return False, "Nothing to migrate (no loadable PDFs)."
    chunks = chunk_documents(
        documents,
        chunk_size=getattr(cfg, "chunk_size", CHUNK_SIZE),
        chunk_overlap=getattr(cfg, "chunk_overlap", CHUNK_OVERLAP),
    )
    store = get_postgres_store(cfg, get_embedding_provider(cfg), table=table)
    store.ensure_schema()
    inserted = store.build_index(chunks)
    st = store.get_status()
    msg = (f"Migrated {report['succeeded']}/{report['found']} docs "
           f"({inserted} new chunks). "
           f"Postgres now: {st.get('chunk_count')} chunks, "
           f"{st.get('document_count')} docs. Chroma untouched.")
    if report["failed"]:
        msg += f" Failed files: {', '.join(report['failed_files'])}."
    return True, msg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder")
    ap.add_argument("--table", default="chunks")
    args = ap.parse_args()
    ok, msg = migrate(args.folder, args.table)
    print(msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
