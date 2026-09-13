"""Production-routed cloud benchmark execution (Step 12 harness fix).

Step 11's runner called backend.retrieve_documents directly, bypassing
workflows.py task routing (comparison/summary/session). This module is
the ONLY benchmark change: it invokes workflows.query_workflow -- the
same non-streaming production entry used by tests/compat -- with the
cloud vector store and the arm LLM provider.

Nothing here changes retrieval, reranking, groundedness, planner,
prompts, scoring, gold, or scenarios. Session scenarios index their
inline session documents under a fresh opaque id and bind it, exactly
like production session uploads (persistent corpus untouched).
"""

import time

from document_scope import new_session_id
from workflows import query_workflow


class _Chunk:
    def __init__(self, text, meta):
        self.page_content = text
        self.metadata = dict(meta or {})


def _session_chunks(documents, prefix):
    objs = []
    for di, entry in enumerate(documents or []):
        fname = entry.get("file") or f"session-doc-{di}.pdf"
        for ci, chunk in enumerate(entry.get("chunks") or []):
            text = (chunk or {}).get("text") or ""
            if not text:
                continue
            cid = f"{prefix}-{di}-{ci}"
            objs.append(_Chunk(text, {
                "chunk_id": cid, "document_id": f"{prefix}-{di}",
                "source_path": fname, "source": fname,
                "file_name": fname, "page": (chunk or {}).get("page", 0),
            }))
    return objs


def run_cloud_scenario(scenario, *, config, vector_store, llm_provider):
    """Run one scenario through production routing.

    Returns dict with answer/sources/info/workflow/session_id. Raises
    on routing failure (callers record, never retry).
    """
    if not isinstance(scenario, dict) or not scenario.get("query"):
        raise ValueError("scenario needs a query")
    t0 = time.monotonic()
    sid = None
    sess = scenario.get("session") or {}
    sess_docs = sess.get("documents") or []
    if sess_docs:
        sid = new_session_id()
        objs = _session_chunks(sess_docs, prefix="bench-session")
        if not objs:
            raise ValueError("session scenario has no indexable chunks")
        vector_store.build_index(objs, session_id=sid)
        if sess.get("bind") == "other":
            sid = new_session_id()
    answer, sources, info = query_workflow(
        scenario["query"], config=config, vector_store=vector_store,
        llm_provider=llm_provider, session_id=sid)
    wall_ms = max(0, int((time.monotonic() - t0) * 1000))
    return {"answer": answer, "sources": sources or [], "info": info or {},
            "workflow": (info or {}).get("workflow"),
            "session_id": sid, "routing_ms": wall_ms}
