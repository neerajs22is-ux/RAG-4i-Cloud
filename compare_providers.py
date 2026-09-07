"""Compare Chroma vs PostgreSQL on the same query (same k/threshold).

Usage:
    python compare_providers.py "What is the lock-in period?" [--k 5]

Prints per-provider: content snippet, source, file_name, page, score,
then whether the top relevant chunk matched. Scores need not be equal.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config  # noqa: E402


def show(name, rows):
    print(f"--- {name} ({len(rows)} hits) ---")
    for r in rows[:5]:
        print(f"  file={r.get('file_name')} page={r.get('page')} "
              f"score={r.get('score') and round(r['score'], 3)} "
              f"src={r.get('source')}")
        print(f"    {(r.get('content') or '')[:160]}")


def main():
    ap = argparse.ArgumentParser(description="Chroma vs Postgres comparison")
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--table", default="chunks")
    args = ap.parse_args()

    from backend import retrieve_documents
    from embeddings import get_embedding_provider
    from vector_store import ChromaVectorStore, get_vector_store
    from postgres_vector_store import get_postgres_store

    cfg = get_config()
    k = args.k or cfg.retrieval_k
    ep = get_embedding_provider(cfg)
    chroma = ChromaVectorStore(persist_directory=cfg.chroma_path,
                               embedding_provider=ep)
    pg = get_postgres_store(cfg, ep, table=args.table)
    try:
        c_rows = retrieve_documents(args.query, config=cfg, vector_store=chroma,
                                    k=k)
    except Exception as e:
        print(f"Chroma error: {e}")
        c_rows = []
    try:
        p_rows = retrieve_documents(args.query, config=cfg, vector_store=pg,
                                    k=k)
    except Exception as e:
        print(f"PostgreSQL error: {e}")
        p_rows = []
    show("Chroma", c_rows)
    show("PostgreSQL", p_rows)
    match = bool(c_rows and p_rows and (
        c_rows[0].get("chunk_id") == p_rows[0].get("chunk_id")
        or (c_rows[0].get("file_name") == p_rows[0].get("file_name")
            and c_rows[0].get("content", "")[:80]
            == p_rows[0].get("content", "")[:80])))
    print(f"MATCH_TOP_CHUNK={match} "
          f"(content match, scores may differ)")


if __name__ == "__main__":
    main()
