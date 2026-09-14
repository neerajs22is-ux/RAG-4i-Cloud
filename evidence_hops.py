"""Bounded two-hop dependent retrieval (Phase 4, Step 4.2).

Hop 1 retrieves normally; a Hop-2 query is derived ONLY from structured
information already present (original question + Hop-1 evidence +
extracted constraints). No LLM, no free-form question generation.

Harvested patterns (adapted, not imported):
- Controllable-RAG-Agent: plan steps refined so each carries all needed
  info + past-steps anti-repeat  ->  Hop-2 is self-contained
  (bridge entity + unresolved need) and can never equal a prior query.
- Semantica ContextManagerWithProvenance.add_context(context, source=)
  ->  hop-tagged evidence with a link to the triggering finding.
- LlamaIndex: no dependent-subquery pattern exists (its sub-questions
  are upfront-generated and independent); our dependent hop fills that
  gap deterministically instead.

Hard bounds: max 2 hops, max 1 follow-up query, no hop 3 (structural:
this module has no loop and never calls itself), no replanning.
Conflicts never trigger Hop 2 (never invent a resolution); sufficient
evidence never triggers Hop 2; failures return Hop-1 evidence intact.
"""

import logging
import re

logger = logging.getLogger(__name__)

MAX_HOPS = 2
MAX_HOP_QUERY_CHARS = 200

_CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
_FILE_RE = re.compile(r"\b[\w][\w\-]*\.pdf\b", re.IGNORECASE)
_QUOTED_RE = re.compile(r"\"([^\"]{2,80})\"|'([^']{2,80})'")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are",
    "was", "were", "it", "its", "this", "that", "for", "with", "as",
    "by", "at", "be", "from", "which", "what", "when", "where", "how",
    "does", "do", "did", "can", "could", "should", "would", "will",
    "there", "here", "about",
})


def _norm(text):
    return _WS_RE.sub(" ", str(text or "").lower()).strip()


def _tokens(text):
    return [t for t in _WORD_RE.findall(str(text or "").lower())
            if t not in _STOP and len(t) > 2]


def _bridge_terms(evidence_contents, question):
    """Ordered candidate bridges: files, quoted spans, entities, numbers.

    Only terms actually present in Hop-1 evidence; ranked files >
    quoted > capitalized phrases > numbers (most identifying first).
    """
    blob = " ".join(str(c or "") for c in (evidence_contents or []))
    q_toks = set(_tokens(question))
    files, quoted, phrases, numbers = [], [], [], []
    for m in _FILE_RE.findall(blob):
        if m not in files:
            files.append(m)
    for m in _QUOTED_RE.findall(blob):
        span = (m[0] or m[1]).strip()
        if span and span not in quoted:
            quoted.append(span)
    for m in _CAP_PHRASE_RE.findall(blob):
        if m not in phrases and len(m) > 3:
            phrases.append(m)
    try:
        from groundedness import extract_numbers
        for n in extract_numbers(blob):
            if n not in numbers:
                numbers.append(n)
    except Exception:
        pass
    cands = files + quoted + phrases + numbers
    # Prefer bridges sharing vocabulary with the question (connected),
    # keep the rest as fallback (entity discovery needs them).
    connected = [c for c in cands
                 if set(_tokens(c)) & q_toks or c in files]
    rest = [c for c in cands if c not in connected]
    return connected + rest


def _constraints_of(question):
    try:
        from query_plan import extract_constraints
        return extract_constraints(question)
    except Exception:
        return []


def derive_hop2(original_question, hop1_sources, constraints=None,
                prior_queries=()):
    """Derive ONE dependent follow-up query, or None.

    Returns {"query", "bridge", "unresolved"} or None when: Hop-1 is
    sufficient, conflicting, empty, erroneous; no unresolved atoms;
    no bridge combines into a valid query; constraints would be lost;
    or the query repeats a prior one.
    """
    try:
        from evidence_verification import verify_evidence
        verdict = verify_evidence(original_question, hop1_sources)
    except Exception as e:
        logger.warning("Hop-2 derivation failed open: %s", e)
        return None
    if not isinstance(verdict, dict):
        return None
    if verdict.get("reason") in ("all-supported", "no-checkable-claims",
                                 "no-evidence", "verifier-error"):
        return None
    if verdict.get("conflicting_claims"):
        return None  # never invent a resolution
    unresolved = [u for u in verdict.get("unsupported_claims", []) or []]
    if not unresolved:
        return None
    # Pad with already-supported terms so the follow-up stays anchored
    # to the original need (never invented: all terms come from the
    # verdict over the real question).
    need = list(unresolved[:2])
    for text in verdict.get("supported_claims", []) or []:
        if len(need) >= 2:
            break
        if text not in need:
            need.append(text)
    contents = [(s or {}).get("content", "") for s in (hop1_sources or [])
                if isinstance(s, dict)]
    bridges = _bridge_terms(contents, original_question)
    if constraints is None:
        constraints = _constraints_of(original_question)
    prior = {_norm(original_question)}
    prior.update(_norm(q) for q in (prior_queries or []))
    for bridge in bridges:
        query = "%s %s" % (bridge, " ".join(need))
        if len(query) > MAX_HOP_QUERY_CHARS:
            continue
        if _norm(query) in prior:
            continue
        missing = [c for c in (constraints or [])
                   if str(c).lower() not in query.lower()]
        if missing:
            continue
        return {"query": query, "bridge": bridge,
                "unresolved": unresolved[:2]}
    return None


def run_two_hop(original_question, hop1_sources, retrieve_fn,
                hop1_query=None, constraints=None):
    """Execute at most ONE dependent hop. Never raises, never loops.

    Returns (merged_sources, record) with record {hops_used, max_hops,
    hop1, hop2:{triggered, query, finding, sources, error}}. Merged
    rows carry hop=[1]/[2]/[1,2] plus H1/H2 evidence tags.
    """
    record = {"hops_used": 1, "max_hops": MAX_HOPS,
              "hop1": {"n_sources": len(hop1_sources or [])},
              "hop2": {"triggered": False, "query": None, "finding": None,
                       "sources": [], "error": None}}
    merged = []
    try:
        from evidence_fanout import merge_fanout
        hop1_query = hop1_query or original_question
        derived = derive_hop2(original_question, hop1_sources,
                              constraints,
                              prior_queries=[hop1_query])
        if derived is None:
            merged = merge_fanout(
                [{"index": 0, "query": hop1_query,
                  "sources": list(hop1_sources or []), "error": None}],
                tag_prefix="H")
            _annotate_hops(merged, {0})
            return merged, record
        record["hop2"]["triggered"] = True
        record["hop2"]["query"] = derived["query"]
        record["hop2"]["finding"] = {
            "bridge": derived["bridge"],
            "unresolved": derived["unresolved"]}
        try:
            hop2_sources = list(retrieve_fn(derived["query"]) or [])
        except Exception as e:
            record["hop2"]["error"] = "%s: %s" % (
                type(e).__name__, str(e)[:160])
            hop2_sources = []
        record["hop2"]["sources"] = [
            (s or {}).get("chunk_id") for s in hop2_sources
            if isinstance(s, dict)]
        merged = merge_fanout(
            [{"index": 0, "query": hop1_query,
              "sources": list(hop1_sources or []), "error": None},
             {"index": 1, "query": derived["query"],
              "sources": hop2_sources, "error": None}],
            tag_prefix="H")
        _annotate_hops(merged, {0, 1}, hop2_ids={
            (s or {}).get("chunk_id") for s in hop2_sources
            if isinstance(s, dict)})
        record["hops_used"] = 2
        return merged, record
    except Exception as e:
        logger.warning("Two-hop failed open: %s", e)
        return list(hop1_sources or []), record


def _annotate_hops(merged, _all, hop2_ids=None):
    hop2_ids = hop2_ids or set()
    for row in merged:
        idx = set(row.get("subqueries", []))
        hops = set()
        if 0 in idx:
            hops.add(1)
        if 1 in idx or row.get("chunk_id") in hop2_ids:
            hops.add(2)
        if not hops:
            hops.add(1)
        row["hops"] = sorted(hops)
