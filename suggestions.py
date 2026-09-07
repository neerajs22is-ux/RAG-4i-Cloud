"""Suggestion engine: next-question prompts, never answers.

Produces exactly-3 suggestion lists from real context only:
  - initial_suggestions(file_names): corpus-adaptive starters.
  - followup_suggestions(question, retrieved): terms from retrieved text.

Deterministic (no LLM). Suggestions are prompts, not facts: every topic
word comes from retrieved chunk text or real filenames; otherwise generic
document-relevant fallbacks are used. Never invents topics.
"""

import re
from typing import Dict, List

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "it",
    "its", "that", "this", "these", "those", "to", "of", "in", "on",
    "for", "and", "or", "with", "by", "from", "as", "at", "what",
    "when", "where", "which", "who", "how", "does", "do", "is",
    "if", "then", "than", "there", "their", "they", "them", "he",
    "she", "we", "you", "your", "i", "my", "me", "s", "t", "not",
    "no", "any", "all", "can", "could", "should", "would", "will",
    "more", "most", "such", "only", "also", "into", "periods",
}

_GENERIC_INITIAL = [
    "What are the key terms in these documents?",
    "What dates and deadlines are mentioned?",
    "Who are the parties involved?",
]

_GENERIC_FOLLOWUP = [
    "What are the key dates mentioned?",
    "Who are the parties involved?",
    "What obligations are described?",
]

# Cue words licensing the "exceptions" question: only suggest it when the
# evidence itself hints that qualified/exceptional cases may exist.
_EXCEPTION_CUES = {
    "exception", "exceptions", "exempt", "exemption", "exclusion",
    "unless", "however", "otherwise", "provided", "override",
}


def _evidence_cues(texts) -> set:
    words = set()
    for text in texts:
        words.update(re.findall(r"[a-z']+", (text or "").lower()))
    return words & _EXCEPTION_CUES


def _top_terms(texts, exclude=(), limit=3, min_len=5):
    """Most frequent significant words across texts (stable order)."""
    counts = {}
    first_seen = {}
    pos = 0
    banned = set(_STOP) | {w.lower() for w in exclude}
    for text in texts:
        for w in re.findall(r"[A-Za-z][A-Za-z'-]*", text or ""):
            lw = w.lower().strip("'-")
            if len(lw) < min_len or lw in banned:
                continue
            counts[lw] = counts.get(lw, 0) + 1
            if lw not in first_seen:
                first_seen[lw] = pos
            pos += 1
    ranked = sorted(counts, key=lambda w: (-counts[w], first_seen[w]))
    return ranked[:limit]


def initial_suggestions(file_names, limit: int = 3) -> List[str]:
    """Exactly `limit` starters grounded in real filenames (or generic)."""
    out = []
    for name in (file_names or [])[:limit]:
        base = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
        base = base.replace("_", " ").replace("-", " ").strip()
        if base:
            out.append(f"What are the key terms in {name}?")
    for g in _GENERIC_INITIAL:
        if len(out) >= limit:
            break
        if g not in out:
            out.append(g)
    return out[:limit]


def followup_suggestions(question: str, retrieved: List[Dict],
                         limit: int = 3) -> List[str]:
    """Exactly `limit` follow-ups from retrieved context (or safe fallback).

    Topic words come only from retrieved chunk text, the asked question,
    or real filenames. Nothing is invented.
    """
    retrieved = retrieved or []
    texts = [s.get("content", "") for s in retrieved]
    files = [s.get("file_name") for s in retrieved if s.get("file_name")]
    q_terms = set(re.findall(r"[a-z']+", (question or "").lower()))
    terms = [t for t in _top_terms(texts, exclude=q_terms) if t not in q_terms]
    out = []
    if terms:
        out.append(f"Tell me more about {terms[0]}.")
    if files:
        out.append(f"What else does {files[0]} say?")
    # Answerable-first: the "exceptions" question only when evidence cues
    # suggest qualified cases; otherwise an evidence-anchored II form.
    if _evidence_cues(texts):
        out.append("Are there any exceptions to these terms?")
    elif len(terms) > 1:
        out.append(f"What does {files[0] if files else 'the document'} "
                   f"say about {terms[1]}?")
    for g in _GENERIC_FOLLOWUP:
        if len(out) >= limit:
            break
        if g not in out:
            out.append(g)
    # Last resort: never return fewer than `limit`.
    while len(out) < limit:
        out.append("What other details are in these documents?")
    return out[:limit]


def remember_followups(state, prompt_text, sources):
    """Persist last grounded answer for top-level button rendering.

    Pure dict logic (Streamlit session-state compatible) so reruns render
    buttons without re-running the pipeline. Cleared when there are no
    sources so stale buttons never linger.
    """
    if sources:
        state["last_followups"] = {"q": prompt_text, "sources": sources,
                                   "seq": len(state.get("messages", []))}
    else:
        state["last_followups"] = None


def current_followups(state, limit: int = 3):
    """Button specs for the latest grounded answer, or None.

    Keys embed the message count so a new answer retires stale buttons.
    """
    saved = state.get("last_followups")
    if not saved or not saved.get("sources"):
        return None
    follows = followup_suggestions(saved["q"], saved["sources"], limit)
    seq = len(state.get("messages", []))
    return [{"label": sug, "key": f"follow_{seq}_{i}"}
            for i, sug in enumerate(follows)]
