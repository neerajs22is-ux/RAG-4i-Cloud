"""Bounded evidence verification (Phase 3, Step 3.1).

Sits AFTER frozen retrieval and BEFORE generation: given the user
question and the retrieved chunks, it determines whether the evidence
sufficiently supports the requested facts. Deterministic and
model-free; reuses answer_support.content_terms and
groundedness.extract_numbers instead of inventing new NLP.

Verdict classes:
- SUPPORTED: every checkable atom found in evidence, no conflicts.
- UNSUPPORTED: checkable atoms exist but evidence is disjoint.
- INSUFFICIENT: evidence is related yet leaves key atoms uncovered.
- CONFLICTING: relevant chunks disagree on a number/date or a
  negated phrase (never resolved by guessing).

Conservative by design: normalized matching (case, possessives,
digit grouping) recognizes semantic equivalence, while numbers,
dates, units, negations, qualifiers, and entity names stay distinct.
The generated answer is NEVER treated as evidence.

BOUNDED CORRECTION (Step 3.2): one verdict-gated corrective search.
warrants_correction decides deterministically; corrective_query builds
a focused query from unresolved numbers/quoted spans/constraints and
disputed values (never invented); merge_evidence dedupes with
provenance (retrieval_round 1/2, cap 10); the verifier runs exactly
once more, then the process STOPS. Controllable-RAG-Agent's
retrieve-vs-answer task decision and NVIDIA's bounded verification
gate are the harvested patterns, adapted to deterministic form.

Failure is fail-open (mirrors the groundedness gate): any internal
error yields sufficient=True with reason "verdict-error" so the
existing answer path keeps working. Bounds: one corrective round,
one re-verification, no loops, no model calls; DECOMPOSE plans are
verified per subquery against the SAME evidence (fan-out remains
deferred by design).
"""

import logging
import re

from answer_support import content_terms
from groundedness import extract_numbers

logger = logging.getLogger(__name__)

VERIFICATION_VERSION = 1

NEGATION_CUES = frozenset({"not", "no", "never", "cannot", "can't"})
MIN_RELEVANT_SHARED = 1
# Plain terms use normalized substring/inflection matching (word order,
# possessives, plurals, and number formats are equivalent). Open
# synonyms ("lease" vs "agreement") are OUT OF SCOPE for the
# deterministic core: they fail as insufficient, never as falsely
# supported. Numbers, quoted spans, and planner constraints are always
# strict.
MAX_ATOMS = 20
MAX_CLAIM_CHARS = 200

_NUM_SEP_RE = re.compile(r"(?<=\d)[,\s](?=\d)")
_POSSESSIVE_RE = re.compile(r"['\u2019]s\b")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_QUOTED_RE = re.compile(r"\"([^\"]{2,80})\"|'([^']{2,80})'")
_STOP = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are",
    "was", "were", "it", "its", "this", "that", "for", "with", "as",
    "by", "at", "be", "from", "which", "what", "when", "where", "how",
    "does", "do", "did", "can", "could", "should", "would", "will",
    "there", "here", "about",
})


def _norm(text):
    t = str(text or "").lower()
    t = _POSSESSIVE_RE.sub("", t)
    t = _NUM_SEP_RE.sub("", t)
    return _WS_RE.sub(" ", t).strip()


def _tokens(text):
    return [t for t in _WORD_RE.findall(str(text or "").lower())
            if t not in _STOP and len(t) > 2]


def _inflections(term):
    yield term
    if term.endswith("s") and len(term) > 4:
        yield term[:-1]
    else:
        yield term + "s"


def _split_value_unit(number_text):
    m = re.match(r"^([\d.,-]+)\s*(.*)$",
                 _norm(number_text).strip())
    if not m:
        return _norm(number_text), ""
    return m.group(1).replace(",", ""), m.group(2).strip()


def _atoms_of(question, extra_constraints=None):
    atoms = []
    seen = set()

    def _add(kind, text):
        norm = _norm(text)
        if norm and norm not in seen and len(atoms) < MAX_ATOMS:
            seen.add(norm)
            atoms.append({"kind": kind, "text": str(text)[:80],
                          "norm": norm})

    for num in extract_numbers(question):
        _add("number", num)
    for m in _QUOTED_RE.findall(str(question or "")):
        span = (m[0] or m[1]).strip()
        if span:
            _add("quoted", span)
    for term in content_terms(question):
        _add("term", term)
    for c in (extra_constraints or []):
        _add("constraint", c)
    return atoms


def _chunk_norms(sources):
    out = []
    for s in (sources or []):
        if not isinstance(s, dict):
            continue
        content = s.get("content") or ""
        if not content:
            continue
        out.append({
            "norm": _norm(content),
            "toks": set(_tokens(content)),
            "ref": {"chunk_id": s.get("chunk_id"),
                    "file_name": s.get("file_name"),
                    "page": s.get("page"),
                    "retrieval_round": s.get("retrieval_round", 1)},
        })
    return out


def _term_supported(term_norm, chunk_norm):
    for variant in _inflections(term_norm):
        if variant and variant in chunk_norm:
            return True
    return False


def _atom_supported(atom, chunks):
    return any((_norm(atom["norm"]) in c["norm"]
                if atom["kind"] in ("number", "quoted", "constraint")
                else _term_supported(atom["norm"], c["norm"]))
               for c in chunks)


def _relevant_chunks(q_toks, chunks):
    return [c for c in chunks
            if len(set(q_toks) & c["toks"]) >= MIN_RELEVANT_SHARED]


def _number_conflicts(question, chunks):
    """Distinct values, same unit, across relevant chunks."""
    relevant = _relevant_chunks(_tokens(question), chunks)
    by_unit = {}
    for c in relevant:
        seen_here = set()
        for num in extract_numbers(c["norm"]):
            value, unit = _split_value_unit(num)
            if not value or (unit, value) in seen_here:
                continue
            seen_here.add((unit, value))
            by_unit.setdefault(unit, {}).setdefault(value, set()).add(
                c["ref"].get("chunk_id"))
    conflicts = []
    for unit, values in sorted(by_unit.items()):
        if len(values) > 1:
            vals = sorted(values)
            conflicts.append(
                "conflicting %s: %s" % (
                    unit or "value",
                    " vs ".join(vals)[:MAX_CLAIM_CHARS]))
    return conflicts


def _bigrams(toks):
    seq = sorted(toks)
    return set(zip(seq, seq[1:]))


def _negation_conflicts(question, chunks):
    """One relevant chunk negates a phrase another affirms."""
    relevant = _relevant_chunks(_tokens(question), chunks)
    neg = [c for c in relevant
           if any(re.search(r"\b" + re.escape(q) + r"\b", c["norm"])
                  for q in NEGATION_CUES)]
    pos = [c for c in relevant if c not in neg]
    conflicts = []
    for n in neg:
        for p in pos:
            shared = _bigrams(n["toks"]) & _bigrams(p["toks"])
            if shared:
                conflicts.append(
                    "negation conflict on shared phrase "
                    "(see %s vs %s)" % (
                        n["ref"].get("chunk_id"),
                        p["ref"].get("chunk_id")))
                break
        if conflicts:
            break
    return conflicts


def _verify_one(question, chunks, extra_constraints=None):
    atoms = _atoms_of(question, extra_constraints)
    refs = [c["ref"] for c in chunks]
    base = {"supported_claims": [], "unsupported_claims": [],
            "conflicting_claims": [], "evidence_refs": refs}
    if not chunks:
        base.update(sufficient=False, confidence="low",
                    reason="no-evidence", verification_required=False)
        return base
    if not atoms:
        base.update(sufficient=True, confidence="high",
                    reason="no-checkable-claims",
                    verification_required=False)
        return base
    for atom in atoms:
        (base["supported_claims"]
         if _atom_supported(atom, chunks)
         else base["unsupported_claims"]).append(atom["text"])
    conflicts = _number_conflicts(question, chunks)
    conflicts += _negation_conflicts(question, chunks)
    base["conflicting_claims"] = conflicts[:4]
    if conflicts:
        base.update(sufficient=False, confidence="low",
                    reason="conflicting-evidence",
                    verification_required=True)
    elif base["unsupported_claims"]:
        q_toks = set(_tokens(question))
        disjoint = all(not (q_toks & c["toks"]) for c in chunks)
        base.update(
            sufficient=False,
            confidence="low" if disjoint else "medium",
            reason="unsupported-claims" if disjoint
            else "insufficient-evidence",
            verification_required=True)
    else:
        base.update(sufficient=True, confidence="high",
                    reason="all-supported",
                    verification_required=True)
    return base


def verify_evidence(question, sources, plan=None, constraints=None):
    """Verify retrieved evidence before generation. Never raises.

    sources: structured dicts (content/chunk_id/file_name/page).
    plan: optional planner dict; DECOMPOSE verifies each subquery
      against the same evidence and merges the verdicts.
    Returns the bounded verification result (sufficient, claim lists,
    evidence_refs, confidence, reason, verification_required).
    """
    try:
        chunks = _chunk_norms(sources)
        mode = (plan or {}).get("mode") if isinstance(plan, dict) else None
        subqueries = ((plan or {}).get("queries") or []) \
            if mode == "decompose" else []
        if subqueries:
            merged = {"supported_claims": [], "unsupported_claims": [],
                      "conflicting_claims": [],
                      "evidence_refs": [c["ref"] for c in chunks],
                      "subqueries": []}
            ok = True
            for sub in subqueries[:3]:
                one = _verify_one(sub, chunks, constraints)
                merged["subqueries"].append(
                    {"query": sub, "sufficient": one["sufficient"],
                     "reason": one["reason"]})
                for key in ("supported_claims", "unsupported_claims",
                            "conflicting_claims"):
                    for item in one[key]:
                        if item not in merged[key]:
                            merged[key].append(item)
                ok = ok and one["sufficient"]
            merged["sufficient"] = ok
            merged["confidence"] = "high" if ok else "medium"
            merged["reason"] = ("all-subqueries-supported" if ok
                                else "subquery-evidence-gap")
            merged["verification_required"] = True
            return merged
        result = _verify_one(question, chunks, constraints)
        result["subqueries"] = []
        return result
    except Exception as e:
        logger.warning("Evidence verification failed open: %s", e)
        return {"sufficient": True, "supported_claims": [],
                "unsupported_claims": [], "conflicting_claims": [],
                "evidence_refs": [], "subqueries": [], "confidence": "low",
                "reason": "verifier-error", "verification_required": False}


# ---------- bounded correction (one round, then STOP) ---------- #

MAX_CORRECTION_ROUNDS = 1
MAX_CORRECTIVE_QUERY_CHARS = 200
MERGED_EVIDENCE_CAP = 10
CORRECTABLE_REASONS = frozenset({"insufficient-evidence",
                                 "conflicting-evidence"})


def warrants_correction(question, verdict):
    """Deterministic policy: is ONE corrective search meaningful?

    Insufficient/conflicting evidence warrants it (a focused query may
    surface the missing or clarifying chunk). Disjoint/unsupported
    evidence warrants it ONLY for exact-matchable anchors (numbers or
    quoted spans the FTS path can hit); vague requests never re-search.
    No-evidence and verdict errors never re-search.
    """
    if not isinstance(verdict, dict):
        return False
    reason = verdict.get("reason")
    if reason in CORRECTABLE_REASONS:
        return True
    if reason == "unsupported-claims":
        q = str(question or "")
        return bool(extract_numbers(q) or _QUOTED_RE.search(q))
    return False


def corrective_query(question, verdict, constraints=None):
    """Build ONE focused retrieval query from unresolved items.

    Uses only numbers/quoted spans/constraints missing from evidence,
    disputed values, and already-supported context terms. Never invents
    terms. Returns None when no meaningful query exists.
    """
    if not isinstance(verdict, dict):
        return None
    supported = {_norm(t) for t in verdict.get("supported_claims", [])}
    spans = []
    for num in extract_numbers(question):
        if _norm(num) not in supported:
            spans.append(num)
    for m in _QUOTED_RE.findall(str(question or "")):
        span = (m[0] or m[1]).strip()
        if span and _norm(span) not in supported:
            spans.append(span)
    for c in (constraints or []):
        if _norm(c) not in supported:
            spans.append(str(c))
    for item in verdict.get("conflicting_claims", []) or []:
        m = re.match(r"^conflicting (.*?): (.*)$", str(item))
        if m:
            unit = m.group(1).strip()
            if unit and unit != "value":
                spans.append(unit)
            for val in m.group(2).split(" vs "):
                spans.append(val.strip())
    for cue in sorted(NEGATION_CUES):
        if re.search(r"\b" + re.escape(cue) + r"\b",
                     str(question or "").lower()) \
                and cue not in " ".join(spans).lower():
            spans.append(cue)
    for text in verdict.get("supported_claims", [])[:2]:
        if text not in spans:
            spans.append(text)
    query = re.sub(r"\s+", " ",
                   " ".join(s for s in spans if s)).strip()
    query = query[:MAX_CORRECTIVE_QUERY_CHARS].strip()
    if not query or _norm(query) == _norm(question):
        return None
    return query


def merge_evidence(original, corrective, cap=MERGED_EVIDENCE_CAP):
    """Union with provenance; dedupe by chunk_id; bounded size.

    Returns (merged, gained_chunk_ids). Inputs are never mutated.
    """
    merged = []
    seen = set()
    for d, rnd in ((original or [], 1), (corrective or [], 2)):
        for s in d:
            if not isinstance(s, dict):
                continue
            cid = s.get("chunk_id") or id(s)
            if cid in seen:
                continue
            seen.add(cid)
            row = dict(s)
            row["retrieval_round"] = rnd
            merged.append(row)
            if len(merged) >= cap:
                break
        if len(merged) >= cap:
            break
    gained = [s["chunk_id"] for s in merged
              if s.get("retrieval_round") == 2]
    return merged, gained


def run_bounded_correction(question, initial_sources, retrieve_fn,
                           plan=None, constraints=None):
    """Verify, at most ONE correction round, re-verify, STOP.

    retrieve_fn(query) -> list of structured sources (the frozen
    pipeline). Returns (final_sources, verification). The verification
    carries a "correction" record with triggered/rounds/verifications
    counts so bounds are auditable. Never raises, never loops.
    """
    def _record(verdict, **kw):
        rec = {"triggered": False, "rounds": 0, "verifications": 1,
               "query": None, "gained": [],
               "reason": verdict.get("reason")}
        rec.update(kw)
        verdict = dict(verdict)
        verdict["correction"] = rec
        return verdict

    try:
        v1 = verify_evidence(question, initial_sources, plan, constraints)
    except Exception as e:
        logger.warning("Evidence verification failed open: %s", e)
        return initial_sources, _record(
            {"sufficient": True, "reason": "verifier-error",
             "verification_required": False})
    if not warrants_correction(question, v1):
        return initial_sources, _record(v1)
    cq = corrective_query(question, v1, constraints)
    if not cq:
        v1c = _record(v1, reason="no-focused-query")
        return initial_sources, v1c
    try:
        new_hits = retrieve_fn(cq) or []
    except Exception as e:
        logger.warning("Corrective retrieval failed open: %s", e)
        v1c = _record(v1, reason="retrieval-error")
        return initial_sources, v1c
    merged, gained = merge_evidence(initial_sources, new_hits)
    try:
        v2 = verify_evidence(question, merged, plan, constraints)
    except Exception as e:
        logger.warning("Evidence re-verification failed open: %s", e)
        v1c = _record(v1, reason="reverify-error")
        return initial_sources, v1c
    v2c = _record(v2, triggered=True, rounds=1, verifications=2,
                  query=cq, gained=gained, reason=v2.get("reason"))
    return merged, v2c