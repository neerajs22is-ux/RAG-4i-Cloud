"""Lightweight conversational routing (deterministic, no LLM call).

Runs BEFORE document retrieval so casual input never hits the RAG pipeline:

    CONVERSATIONAL -> short canned reply, no retrieval
    DOCUMENT_QUERY -> existing RAG pipeline unchanged
    OUT_OF_SCOPE   -> polite redirect, no retrieval

Patterns are deliberately narrow: anything not clearly conversational or
clearly out-of-scope falls through to DOCUMENT_QUERY so grounding behavior
is never weakened by the router.
"""

import re

CONVERSATIONAL = "conversational"
DOCUMENT_QUERY = "document_query"
OUT_OF_SCOPE = "out_of_scope"

_GREETINGS = {
    "hi", "hii", "hiii", "hello", "hey", "heyy", "yo",
    "good morning", "good afternoon", "good evening", "morning",
    "namaste",
}
_THANKS = {
    "thanks", "thank you", "thankyou", "thx", "thanks a lot",
    "thank you so much", "much appreciated",
}
_FAREWELLS = {
    "bye", "bye bye", "goodbye", "good night", "see you", "see ya",
    "take care",
}
_WELLBEING = {
    "how are you", "how r u", "how are you doing", "how is it going",
    "how do you do",
}

# Narrow out-of-scope signals (general knowledge / creative / unrelated tasks).
_OUT_OF_SCOPE_PATTERNS = (
    r"\bcapital of\b",
    r"\bweather\b|\btemperature outside\b|\bforecast\b",
    r"\bpoem\b|\bpoetry\b|\bhaiku\b|\bsonnet\b",
    r"\bjoke\b|\bfunny\b",
    r"\bpresident\b|\bprime minister\b|\belection\b",
    r"\bwho won\b|\bscore\b.*\bmatch\b|\bcricket\b|\bfootball\b",
    r"\bstock price\b|\bshare price\b|\bcrypto\b|\bbitcoin\b",
    r"\btranslate\b|\btranslation\b",
    r"\bwrite\b.*\b(essay|story|letter|email|code|program|script)\b",
    r"\brecipe\b|\bcook\b",
    r"\bmovie\b|\bsong\b|\blyrics\b",
    r"\bmeaning of life\b",
)


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip(" .,!?;:'\"-")


def route_query(text: str) -> str:
    """Classify user input. Defaults to DOCUMENT_QUERY (safe side)."""
    t = _normalize(text)
    if not t:
        return CONVERSATIONAL
    if t in _GREETINGS or t in _THANKS or t in _FAREWELLS or t in _WELLBEING:
        return CONVERSATIONAL
    for pat in _OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, t):
            return OUT_OF_SCOPE
    return DOCUMENT_QUERY


def reply_for_route(text: str, route: str) -> str:
    """Canned reply for non-document routes (no retrieval, no LLM)."""
    t = _normalize(text)
    if route == CONVERSATIONAL:
        if t in _THANKS:
            return "You're welcome!"
        if t in _FAREWELLS:
            return "Goodbye! Feel free to come back if you have questions about your documents."
        if t in _WELLBEING:
            return ("I'm doing well, thanks! How can I help? You can ask me "
                    "about the documents I've been given, including contracts, "
                    "clauses, dates, obligations, and other document details.")
        return ("Hi! How can I help? You can ask me about the documents I've "
                "been given, including contracts, clauses, dates, obligations, "
                "and other document details.")
    if route == OUT_OF_SCOPE:
        return ("I'm mainly here to help with the documents connected to this "
                "assistant. You can ask me about contract terms, clauses, dates, "
                "parties, obligations, and other information in those documents.")
    raise ValueError(f"Unknown route: {route}")
