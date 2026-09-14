"""Bounded multi-subquery retrieval and evidence merge (Phase 4, Step 4.1).

For planner mode DECOMPOSE only: each subquery runs independently
through the EXISTING frozen retrieve_documents (same k/threshold/scope),
then results merge deterministically with subquery provenance.
DIRECT/REWRITE never touch this module.

Harvested patterns (adapted, not imported):
- Controllable-RAG-Agent: same question fanned across retrievers with
  citation-tagged concatenation  ->  per-subquery retrieval with
  evidence_tag labels ([Q1], [Q1+Q2]).
- LlamaIndex SubQuestionQueryEngine: independent subquery execution,
  failed subqueries filtered (never fatal), sources kept per pair  ->
  per-subquery try/except isolation + provenance-preserving merge.
- RAGFlow multi-doc handling: inspected, nothing cheaply adoptable
  without its framework; our merge covers the need deterministically.

Merge policy (round-robin by per-subquery rank): take rank 1 of every
subquery, then rank 2, and so on, skipping already-taken chunk_ids,
until FANOUT_MERGE_CAP. This prefers coverage ACROSS subqueries over
globally-highest scores, deterministically. Dedupe key is chunk_id.
"""

FANOUT_MERGE_CAP = 9
MAX_SUBQUERIES = 3


def fan_out_retrieval(plan, retrieve_fn):
    """Run each subquery independently. Never raises.

    plan: planner dict (mode DECOMPOSE, queries list).
    retrieve_fn(query) -> list of structured sources (frozen path).
    Returns (per_subquery, record) where per_subquery items hold
    {index, query, sources, error}. Failures are isolated per
    subquery; other subqueries still return evidence.
    """
    record = {"mode": "decompose", "subqueries": [], "merged_count": 0,
              "cap": FANOUT_MERGE_CAP}
    per_subquery = []
    queries = []
    try:
        if isinstance(plan, dict):
            queries = list(plan.get("queries") or [])
    except Exception:
        queries = []
    for index, query in enumerate(queries[:MAX_SUBQUERIES]):
        item = {"index": index, "query": query, "sources": [],
                "error": None}
        try:
            hits = retrieve_fn(query) or []
            item["sources"] = list(hits)
        except Exception as e:
            item["error"] = "%s: %s" % (type(e).__name__, str(e)[:160])
        per_subquery.append(item)
        record["subqueries"].append(
            {"index": index, "query": query,
             "n_sources": len(item["sources"]), "error": item["error"]})
    return per_subquery, record


def merge_fanout(per_subquery, cap=FANOUT_MERGE_CAP):
    """Deterministic coverage-preferring merge. Never raises.

    Each merged chunk is a copy of its structured source plus:
      subqueries: [0-based subquery indices supporting it]
      subquery_ranks: {index: 1-based rank within that subquery}
      evidence_tag: "Q1" or "Q1+Q3" for generation context labels
      retrieval_round: preserved when already set, else 1
    """
    try:
        merged = []
        by_id = {}
        subs = list(per_subquery or [])
        depth = 0
        while len(merged) < cap:
            progressed = False
            for item in subs:
                sources = item.get("sources") or []
                if depth < len(sources):
                    progressed = True
                    src = sources[depth]
                    if not isinstance(src, dict):
                        continue
                    cid = src.get("chunk_id") or id(src)
                    if cid in by_id:
                        entry = by_id[cid]
                        if item["index"] not in entry["subqueries"]:
                            entry["subqueries"].append(item["index"])
                            entry["subquery_ranks"][item["index"]] = \
                                depth + 1
                            entry["evidence_tag"] = _tag(
                                entry["subqueries"])
                        continue
                    row = dict(src)
                    row["subqueries"] = [item["index"]]
                    row["subquery_ranks"] = {item["index"]: depth + 1}
                    row["evidence_tag"] = _tag([item["index"]])
                    if "retrieval_round" not in row:
                        row["retrieval_round"] = 1
                    by_id[cid] = row
                    merged.append(row)
                    if len(merged) >= cap:
                        break
            if not progressed:
                break
            depth += 1
            if depth > 1000:
                break
        return merged
    except Exception:
        return []


def _tag(indices):
    return "+".join("Q%d" % (i + 1) for i in sorted(indices))


def render_tagged_context(sources):
    """Evidence text with subquery tags (generation context).

    Sources without a tag render exactly as before (bare content), so
    DIRECT/REWRITE context is byte-identical to today.
    """
    blocks = []
    for s in (sources or []):
        text = (s or {}).get("content", "")
        tag = (s or {}).get("evidence_tag")
        blocks.append(("[%s] " % tag + text) if tag else text)
    return "\n\n---\n\n".join(blocks)
