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
# Framing verbs: how the question is asked, not what is asked.
_GENERIC_FRAMING = {
    "cover", "covers", "say", "says", "tell", "tells", "contain",
    "contains", "include", "includes", "list", "lists", "describe",
    "describes", "show", "shows", "give", "gives",
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
    if term in _GENERIC or term in _GENERIC_FRAMING:
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


# Answers starting with one of these are system wordings (refusals,
# clarifications, errors), never model claims: the guard skips them.
_SYSTEM_PREFIXES = (
    "I could not find", "I don't have enough", "The model returned",
    "The knowledge base", "An error occurred",
    "Quick check before I answer:",
)


def split_sentences(text: str):
    """Naive sentence split (good enough for a honesty heuristic)."""
    import re as _re
    return [s.strip() for s in _re.split(r"(?<=[.!?])\s+", text or "")
            if s.strip()]


def citation_guard(answer: str, retrieved: List[Dict]) -> Dict:
    """Flag answer sentences with no lexical support in the evidence.

    Pure post-generation honesty check: never mutates prompt, retrieval,
    or text. Returns {"flagged": [...], "count": int, "total": int}.
    System wordings (refusals/clarifications/errors) are never flagged.
    """
    if not retrieved or not (answer or "").strip():
        return {"flagged": [], "count": 0, "total": 0}
    if answer.strip().startswith(_SYSTEM_PREFIXES):
        return {"flagged": [], "count": 0, "total": 0}
    evidence = " ".join(s.get("content", "") for s in retrieved).lower()
    ev_stems = {_stem(w) for w in re.findall(r"[a-z']+", evidence)}
    flagged, total = [], 0
    for sentence in split_sentences(answer):
        terms = [t for t in content_terms(sentence)]
        if not terms:
            continue
        total += 1
        if not any(t in evidence or _stem(t) in ev_stems for t in terms):
            flagged.append(sentence)
    return {"flagged": flagged, "count": len(flagged), "total": total}


CLARIFICATION_MARKER = "Quick check before I answer:"

_CONFIRMATIONS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "proceed",
    "go ahead", "answer anyway", "please do", "yes please", "do it",
    "answer it", "go on",
}


def build_clarification(question: str, support: Dict,
                        retrieved: List[Dict]) -> str:
    """One deterministic clarifying question for PARTIAL evidence.

    Names only covered/missing terms and the retrieved filename; never
    claims unseen sections exist. At most one per user question (callers
    must check prior_was_clarification first).
    """
    covered = [t for t in support.get("covered", [])
               if t not in _TIME_QUESTIONS and t not in _GENERIC
               and t not in _GENERIC_FRAMING] or ["related material"]
    missing = support.get("missing", []) or ["that detail"]
    files = [s.get("file_name") for s in (retrieved or [])
             if s.get("file_name")]
    doc = files[0] if files else "the documents"
    return (
        f"{CLARIFICATION_MARKER} I found material on "
        f"{', '.join(covered[:3])} but nothing on "
        f"{', '.join(missing[:3])} — should I answer what {doc} says "
        f"about {', '.join(covered[:3])} specifically? Reply “yes” to "
        f"proceed, or rephrase your question."
    )


def prior_was_clarification(context) -> bool:
    """True if the last assistant message was our clarification."""
    from conversation_memory import coerce_context

    context = coerce_context(context)
    if not context:
        return False
    for msg in reversed(context[-6:]):
        if msg.get("role") == "assistant":
            return (msg.get("content", "").strip()
                    .startswith(CLARIFICATION_MARKER))
    return False


def is_confirmation(text: str, context=None) -> bool:
    """Bare confirmation, valid only right after our clarification."""
    t = (text or "").strip().lower().strip(" .,!?;:'\"-")
    if t not in _CONFIRMATIONS and not any(
            t.startswith(p) for p in ("yes", "sure", "ok", "go ahead")):
        return False
    if context is None:
        return True
    return prior_was_clarification(context)


def clarified_question(context) -> str:
    """Original question behind a confirmation: most recent substantive
    user message that is not itself a confirmation."""
    from conversation_memory import coerce_context

    context = coerce_context(context)
    if not context:
        return ""
    for msg in reversed(list(context)):
        if msg.get("role") != "user":
            continue
        text = msg.get("content", "").strip()
        if not text or is_confirmation(text):
            continue
        if content_terms(text):
            return text
    return ""


def resolve_effective_question(query_text: str, context):
    """Map a bare confirmation to the question it confirms.

    Returns (effective_question, already_clarified). Confirmation is only
    recognized right after our own clarification, and at most one
    clarification is ever issued per question (callers skip clarifying
    again when already_clarified is True).
    """
    if is_confirmation(query_text, context):
        original = clarified_question(context)
        if original:
            return original, True
    return query_text, False
