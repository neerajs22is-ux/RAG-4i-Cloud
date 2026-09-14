"""Bounded deterministic query planning (Phase 2, Step 2.1).

Classifies a user query BEFORE the frozen retrieval path into:

- DIRECT: single intent; the query is preserved verbatim (fast path).
- REWRITE: same information need, but noisy wording; exactly one
  cleaned retrieval query (filler/noise removal only, never invention).
- DECOMPOSE: several independently answerable needs; up to 3 queries.

Not to be confused with Phase-5D query_planner.py (LLM reasoning
plans, frozen, disabled by default). This module is deterministic,
dependency-free, and always safe: any failure, malformed output, or
bound violation silently falls back to DIRECT on the original query.

Hard bounds: max 3 queries, no recursion, no fan-out here (DECOMPOSE
prepares queries; retrieval internals are untouched), no answers.
"""

import re

DIRECT = "direct"
REWRITE = "rewrite"
DECOMPOSE = "decompose"
MODES = frozenset({DIRECT, REWRITE, DECOMPOSE})

MAX_QUERIES = 3
MAX_QUERY_CHARS = 200

_FILLER_LEAD = re.compile(
    r"^(?:please|kindly|hi|hello|hey|dear|sir|madam|can you|could you|"
    r"would you|will you|tell me|please tell me|i want to know|"
    r"i need to know|i would like to know|give me|show me|find me|"
    r"question\s*:)\b[\s,:]*",
    re.IGNORECASE)
_FILLER_TRAIL = re.compile(
    r"[\s.?!]*(?:please|thanks|thank you|thx|asap|urgent)[\s.?!]*$",
    re.IGNORECASE)
_NOISE_PUNCT = re.compile(r"([?!.,;:]){2,}")
_WS = re.compile(r"\s+")

_FILE_RE = re.compile(r"\b[\w][\w\-]*\.pdf\b", re.IGNORECASE)
_QUOTED_RE = re.compile(r"\"([^\"]{2,80})\"|'([^']{2,80})'")
_NUMBER_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d+(?:\.\d+)?\s*(?:%|percent|months?|years?|days?|weeks?|"
    r"hours?|minutes?|dollars?|usd|inr|rs\.?|kg|km)\b",
    re.IGNORECASE)
_QUALIFIER_RE = re.compile(
    r"\b(not|no|never|none|only|except|unless|cannot|can\'t|must|"
    r"required|always|without|than|more than|less than|under|over|"
    r"above|below|between|vs\.?|versus|table|list|bullets?|summary|"
    r"summarize|overview)\b",
    re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _words(text):
    return [w for w in _WORD_RE.findall(str(text or ""))
            if len(w.strip("'")) >= 3]


def extract_constraints(query_text):
    """Literal spans that planned queries must preserve verbatim."""
    q = str(query_text or "")
    out = []
    for m in _FILE_RE.findall(q):
        out.append(m)
    for m in _QUOTED_RE.findall(q):
        span = (m[0] or m[1]).strip()
        if span:
            out.append(span)
    for m in _NUMBER_RE.findall(q):
        out.append(re.sub(r"\s+", " ", m).strip())
    for m in _QUALIFIER_RE.findall(q):
        out.append(m.lower())
    seen = set()
    keep = []
    for t in out:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            keep.append(t)
    return keep


def _clean_query(query_text):
    """Deterministic noise removal. Returns cleaned string (may equal)."""
    t = str(query_text or "").strip()
    prev = None
    while prev != t:
        prev = t
        t = _FILLER_LEAD.sub("", t).strip()
        t = _FILLER_TRAIL.sub("", t).strip()
    t = _NOISE_PUNCT.sub(r"\1", t)
    t = _WS.sub(" ", t).strip()
    t = t.strip(" ,;:")
    return t


def _split_clauses(query_text):
    parts = re.split(r"[;\n]+|(?<=[?!])\s+", str(query_text or ""))
    return [p.strip(" ,;:") for p in parts if p and p.strip(" ,;:")]


def _validate(mode, original, queries, constraints, rationale):
    """Return a canonical plan, or a DIRECT fallback (never raises)."""
    try:
        if mode not in MODES:
            raise ValueError("bad mode")
        qs = [str(q or "").strip() for q in (queries or [])]
        qs = [q for q in qs if q]
        if not qs or len(qs) > MAX_QUERIES:
            raise ValueError("bad count")
        if any(len(q) > MAX_QUERY_CHARS for q in qs):
            raise ValueError("too long")
        if mode == DIRECT:
            qs = [str(original or "").strip()]
        elif mode == REWRITE:
            if len(qs) != 1 or qs[0] == str(original or "").strip():
                raise ValueError("bad rewrite")
        else:  # DECOMPOSE
            if len(qs) < 2:
                raise ValueError("bad decompose")
        low = " ".join(qs).lower()
        for c in constraints or []:
            if str(c).lower() not in low:
                raise ValueError("constraint dropped: %s" % c)
        return {"mode": mode, "original_query": str(original or ""),
                "queries": qs, "rationale": str(rationale or "")[:200],
                "constraints": list(constraints or [])}
    except Exception as e:
        return {"mode": DIRECT,
                "original_query": str(original or ""),
                "queries": [str(original or "").strip()],
                "rationale": "fallback to DIRECT: %s" % e,
                "constraints": list(constraints or [])}


def plan_query(query_text):
    """Classify + plan. Never raises; worst case is DIRECT passthrough."""
    try:
        original = str(query_text or "")
        if not original.strip():
            return _validate(DIRECT, original, [original], [],
                             "empty query preserved")
        constraints = extract_constraints(original)
        clauses = _split_clauses(original)
        substantial = [c for c in clauses if len(_words(c)) >= 3]
        if len(substantial) >= 2:
            if len(substantial) > MAX_QUERIES:
                return _validate(
                    DIRECT, original, [original], constraints,
                    "too many intents for bounded decompose; kept DIRECT")
            return _validate(
                DECOMPOSE, original, substantial, constraints,
                "%d independent clauses" % len(substantial))
        cleaned = _clean_query(original)
        if cleaned and cleaned != original.strip() \
                and len(_words(cleaned)) >= 2:
            return _validate(
                REWRITE, original, [cleaned], constraints,
                "noise/filler removed; constraints preserved")
        return _validate(DIRECT, original, [original], constraints,
                         "single intent; query preserved verbatim")
    except Exception:
        return {"mode": DIRECT,
                "original_query": str(query_text or ""),
                "queries": [str(query_text or "").strip()],
                "rationale": "fallback to DIRECT: planner error",
                "constraints": []}


def effective_retrieval_query(query_text):
    """Forms-input for the frozen pipeline. Never raises.

    REWRITE flows its single cleaned query through existing retrieval.
    DIRECT is byte-identical to today. DECOMPOSE fan-out is deferred
    (retrieval internals untouched), so it also yields the original.
    """
    try:
        plan = plan_query(query_text)
    except Exception:
        return query_text
    if plan.get("mode") == REWRITE:
        qs = plan.get("queries") or []
        if len(qs) == 1 and qs[0]:
            return qs[0]
    return query_text
