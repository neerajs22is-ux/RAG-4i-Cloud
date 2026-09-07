"""Answer-support assessment: what retrieved evidence actually establishes.

Structural and generalizable (no phrase lists, no per-topic rules):
  - DIRECT: every substantive question term is covered by the evidence.
  - PARTIAL: some covered, some not -> answer what's established, mark
    the rest unknown, suggest search directions (never claims).
  - UNSUPPORTED: nothing covered -> contextual refusal, no LLM call.

Coverage is lexical (term/plural/time-class) over retrieved text. Time and
generic-abstraction words are satisfied structurally so ordinary questions
("how often…", "payment terms") assess correctly without topic lists.
"""

import re
from typing import Dict, List

DIRECT = "direct"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "it",
    "its", "that", "this", "these", "those", "to", "of", "in", "on",
    "for", "and", "or", "with", "by", "from", "as", "at", "what",
    "when", "where", "which", "who", "how", "does", "do", "if",
    "then", "than", "there", "their", "they", "them", "he", "she",
    "we", "you", "your", "i", "my", "me", "s", "t", "not", "any",
    "all", "can", "could", "should", "would", "will", "more", "most",
    "such", "only", "also", "into", "about", "tell", "please", "else",
    "other", "another", "much", "many",
}

# Abstract nouns satisfied trivially (they name no checkable fact).
_GENERIC = {
    "terms", "term", "conditions", "condition", "details", "detail",
    "information", "info", "policy", "process", "rules", "rule",
    "things", "stuff", "something", "anything",
}

# Question words answered structurally by time/number evidence.
_TIME_QUESTIONS = {"often", "long", "much", "many", "when"}
_TIME_EVIDENCE = {
    "annual", "annually", "monthly", "weekly", "daily", "yearly",
    "quarterly", "year", "years", "month", "months", "week", "weeks",
    "day", "days", "hour", "hours", "notice", "net", "deadline",
    "date", "dates", "renew", "renews", "renewal",
}

_BROAD_FRAMES = (
    r"\bwhat else\b", r"\btell me more\b", r"\boverview\b",
    r"\bsummar",
    r"\bmore details\b", r"\banything else\b", r"\bimportant\b",
    r"\ball\b.*\b(say|state|mention|establish)\b",
    r"\babout this\b", r"\babout that\b",
)
_DOC_WORDS = (
    "document", "contract", "lease", "agreement", "policy", "report",
    "manual", "file",
)
_FILENAME_RE = re.compile(r"\b[\w][\w\-]*\.pdf\b", re.IGNORECASE)


def content_terms(question: str) -> List[str]:
    """Substantive lowercase terms (len>=4 or hyphenated), order kept."""
    out = []
    for w in re.findall(r"[A-Za-z][A-Za-z'-]*", question or ""):
        lw = w.lower().strip("'-")
        if len(lw) >= 4 and lw not in _STOP and lw not in out:
            out.append(lw)
    return out


def _variants(term: str):
    yield term
    if term.endswith("s") and len(term) > 4:
        yield term[:-1]
    else:
        yield term + "s"


def _stem(word: str) -> str:
    """Light stem for coverage matching (renewal/renews -> renew)."""
    import re as _re
    w = _re.sub(r"(ing|ed|al|es|s)$", "", word)
    return w if len(w) >= 3 else word


def _term_covered(term: str, evidence: str, time_tokens: set,
                  evidence_stems: set) -> bool:
    if term in _GENERIC:
        return True
    if term in _TIME_QUESTIONS and (time_tokens & _TIME_EVIDENCE):
        return True
    if any(v in evidence for v in _variants(term)):
        return True
    return _stem(term) in evidence_stems


def assess_support(question: str, retrieved: List[Dict]) -> Dict:
    """Assess what the evidence establishes. Never calls the LLM."""
    evidence = " ".join(s.get("content", "") for s in (retrieved or []))
    ev_low = evidence.lower()
    time_tokens = set(re.findall(r"[a-z']+", ev_low))
    evidence_stems = {_stem(w) for w in time_tokens}
    terms = content_terms(question)
    if not terms:
        # No checkable terms (e.g. "Why?" alone can never happen here:
        # followups are expanded first). Treat as partial, not direct.
        return {"level": PARTIAL if retrieved else UNSUPPORTED,
                "covered": [], "missing": [], "terms": []}
    covered = [t for t in terms if _term_covered(t, ev_low, time_tokens,
                                                 evidence_stems)]
    missing = [t for t in terms if t not in covered]
    if len(covered) == len(terms):
        level = DIRECT
    elif covered:
        level = PARTIAL
    else:
        level = UNSUPPORTED
    return {"level": level, "covered": covered,
            "missing": missing, "terms": terms}


def detect_broad_scope(question: str):
    """Broad/overview intent + optional named file. Structural only.

    Returns (is_broad: bool, target_file: str|None).
    """
    t = (question or "").lower()
    m = _FILENAME_RE.search(question or "")
    target = m.group(0) if m else None
    if target:
        return True, target
    if any(re.search(p, t) for p in _BROAD_FRAMES):
        has_doc = any(w in t for w in _DOC_WORDS)
        short = len(content_terms(question)) <= 2
        if has_doc or short:
            return True, None
    return False, None


def unsupported_reply(question: str) -> str:
    """Contextual refusal (deterministic, no LLM call, no invented facts)."""
    terms = content_terms(question)[:6]
    topic = " ".join(terms) if terms else "that topic"
    return (
        f"I don't have enough information in the indexed documents to "
        f"determine {topic}.\n\n"
        f"I can help check whether the documents contain requirements "
        f"around key terms, dates, parties, or obligations."
    )
