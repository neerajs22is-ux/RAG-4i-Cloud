"""Intent observer: decides WHEN to retrieve, never WHAT the answer is.

Approach: deterministic classifier over intent FEATURES (linguistic frames
+ dialogue structure), not example-phrase lists. Chosen over a per-message
LLM classifier because it adds zero latency/cost, cannot hallucinate a
route, degrades safely, and is fully testable. The API is shaped so an LLM
classifier could replace these internals later.

Categories:
    CONVERSATION  - greetings, thanks, farewells, wellbeing (closed class).
    CAPABILITY    - questions about the assistant itself (2nd-person +
                    capability frame). Generalizes across phrasings.
    DOCUMENT_QUERY    - substantive question -> RAG (safe default).
    DOCUMENT_FOLLOWUP - short/ambiguous message depending on prior document
                    conversation -> RAG with expanded query.
    OUT_OF_SCOPE  - clearly unrelated (narrow patterns) -> redirect.

Grounding rule: uncertain -> DOCUMENT_QUERY. needs_retrieval is True for
both document intents. Reasons are developer/debug only, never user-facing.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

CONVERSATIONAL = "conversational"  # back-compat alias of CONVERSATION
CONVERSATION = "conversation"
CAPABILITY = "capability"
DOCUMENT_QUERY = "document_query"
DOCUMENT_FOLLOWUP = "document_followup"
OUT_OF_SCOPE = "out_of_scope"

_RETRIEVAL_INTENTS = frozenset({DOCUMENT_QUERY, DOCUMENT_FOLLOWUP})


@dataclass(frozen=True)
class QueryIntent:
    intent: str
    confidence: float
    needs_retrieval: bool
    reason: str = field(default="", compare=False)


# --- closed-class conversation (greetings are finite; lists are correct) ---
_GREETINGS = {
    "hi", "hii", "hello", "hey", "heyy", "yo", "good morning",
    "good afternoon", "good evening", "morning", "namaste",
}
# Single-word social tokens: any short message built ONLY from these (+
# politeness fillers) is conversational regardless of exact combination.
_SOCIAL_TOKENS = {
    "hi", "hii", "hello", "hey", "heyy", "yo", "morning", "evening",
    "afternoon", "namaste", "thanks", "thank", "please", "well", "there",
}
# Structural social frames with open slots (generalize unseen paraphrases):
# "how's <slot>", "what's <slot>", "hope <slot>" where the slot is a
# wellbeing/life token — never a document noun (guarded below).
_SOCIAL_FRAMES = (
    r"\bgood\s+(day|morning|afternoon|evening)\b",
    r"\bhow\s*(?:'s|\bis|\bare)\s*(it\s+going|things|your\s+day|"
    r"you\s+doing|you\s+been|going|going\s+on|life|your\s+day\s+going|day)\b",
    r"\bhow\s+was\s+your\s+day\b",
    r"\bwhat\s*(?:'s|\bis)\s*(up|on|going\s+on|new|happening)\b",
    r"\bhope\s+you(?:'re|\bare)\s*(well|doing\s+well)\b",
    r"\bhope\s+your\s+day\b",
    r"\bhow\s+have\s+you\s+been\b",
)
# Document-domain nouns: if present, the message is substantive even when
# it wears conversational clothing ("How is the lock-in going?").
_DOC_NOUNS = {
    "lease", "contract", "agreement", "deed", "clause", "period",
    "lock-in", "lock", "rent", "party", "parties", "termination",
    "renewal", "expiry", "expire", "deposit", "payment", "notice",
    "obligation", "terms", "conditions", "document", "policy",
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

# --- capability: 2nd-person + capability frame (generalizes phrasing) ---
_ASSISTANT_REFS = r"(you|your|yours|\bu\b)"
_CAPABILITY_VERBS = (
    r"(can|could|would|will|may|might|should|do|does|did|help|helps|assist"
    r"|use|answer|know|handle|support|suggest|ask|capable|abilities|features"
    r"|functions|good\s+for)"
)
# Words consumed by the capability frame; what remains decides factuality.
_FRAME_WORDS = {
    "what", "how", "can", "could", "would", "will", "may", "might",
    "should", "do", "does", "did", "you", "your", "yours", "u", "me",
    "my", "i", "help", "helps", "assist", "use", "answer", "know",
    "handle", "support", "suggest", "ask", "capable", "abilities",
    "features", "functions", "good", "for", "about", "with", "to",
    "the", "a", "an", "am", "is", "are", "be", "it", "this", "that",
    "of", "in", "on", "my",
}
_CAPABILITY_SHORTCUTS = (
    r"\bhow\s+to\s+use\b",
    r"\bwhat\s+to\s+ask\b",
    r"\bsuggest\b.*\bask\b",
    r"\bexamples?\b.*\bquestions?\b",
    r"\bhow\s+(do|should|can)\s+i\s+(use|interact|ask|talk)\b",
)


def _capability_remainder(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z']+", text) if w not in _FRAME_WORDS]


def _looks_capability(text: str) -> Optional[str]:
    """Feature conjunction: addresses the assistant about itself AND leaves
    almost no factual remainder (else it may be a document question)."""
    if not re.search(_ASSISTANT_REFS, text):
        return None
    if not re.search(_CAPABILITY_VERBS, text):
        return None
    if not re.search(r"\?|\b(what|how|can|could|do|should|tell|suggest|show)\b", text):
        return None
    rest = _capability_remainder(text)
    if len(rest) >= 2:
        return None  # substantive remainder -> may be factual, stay safe
    return f"assistant-directed, remainder={rest}"

# --- narrow out-of-scope signals (unchanged policy) ---
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

# --- followup: structural dependence on prior document talk ---
_FOLLOWUP_FRAMES = (
    # NOTE: _normalize strips trailing punctuation, so frames must NOT
    # require a literal "?".
    r"^why\b", r"^how come\b", r"^and\b",
    r"\btell me more\b", r"^more\b",
    r"\bwhat about\b", r"\bhow about\b",
    r"\bdoes that\b", r"\bis that\b", r"\bdo they\b", r"\bare they\b",
    r"\bwhat does that mean\b", r"\bexplain\b.*\b(that|this|it)\b",
)
_DEPENDENCE_MARKERS = (
    "it", "its", "they", "them", "their", "that", "those", "this", "these",
    "there", "he", "him", "his", "she", "her", "them too",
)
_CONTENT_WORD_MIN = 3  # fewer content words => likely dependent


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip(" .,!?;:'\"-")


def _content_words(text: str) -> List[str]:
    stop = {"a", "an", "the", "is", "are", "was", "it", "that", "this",
            "to", "of", "in", "on", "for", "do", "does", "me", "more",
            "about", "too", "there", "s", "t"}
    return [w for w in re.findall(r"[a-z']+", text) if w not in stop]


def _had_document_exchange(context: Optional[List[Dict]]) -> bool:
    """True if recent context contains substantive assistant output to build on."""
    if not context:
        return False
    for msg in reversed(context[-6:]):
        if msg.get("role") == "assistant" and len(msg.get("content", "")) > 40:
            return True
    return False


def _substantive_user_questions(context) -> List[str]:
    """All substantive user questions, oldest first."""
    out = []
    for msg in context or []:
        if msg.get("role") == "user":
            text = msg.get("content", "").strip()
            if len(_content_words(_normalize(text))) >= _CONTENT_WORD_MIN:
                out.append(text)
    return out


def _last_user_document_question(context: Optional[List[Dict]]) -> str:
    subs = _substantive_user_questions(context)
    return subs[-1] if subs else ""


def _looks_social(text: str) -> bool:
    """Short social message with no document nouns.

    Two structural tests (either suffices):
      1. every token is a known social filler (any novel combination works);
      2. matches a social frame (how's/what's/hope slots) with no doc nouns.
    A document noun anywhere vetoes: substantive wins (safe side).
    """
    words = re.findall(r"[a-z']+", text)
    if not words or len(words) > 8:
        return False
    if any(w in _DOC_NOUNS or w.rstrip("s") in _DOC_NOUNS for w in words):
        return False
    if all(w in _SOCIAL_TOKENS for w in words):
        return True
    return any(re.search(pat, text) for pat in _SOCIAL_FRAMES)


def _looks_followup(text: str, context: Optional[List[Dict]]) -> Optional[str]:
    """Structural followup test. Returns reason or None."""
    if not _had_document_exchange(context):
        return None
    for pat in _FOLLOWUP_FRAMES:
        if re.search(pat, text):
            return f"followup frame {pat}"
    words = _content_words(text)
    markers = [w for w in re.findall(r"[a-z']+", text)
               if w in _DEPENDENCE_MARKERS]
    if len(words) < _CONTENT_WORD_MIN and (markers or len(words) <= 1):
        return f"short/dependent ({len(words)} content words)"
    return None


def observe_query(message, conversation_context=None) -> QueryIntent:
    """Classify intent. Uncertain -> DOCUMENT_QUERY (retrieval, safe side).

    conversation_context may be a plain message list or a
    ConversationMemory (bounded window); both are coerced to a list.
    """
    from conversation_memory import coerce_context

    context = coerce_context(conversation_context)
    t = _normalize(message or "")
    if not t:
        return QueryIntent(CONVERSATION, 0.95, False, "empty input")
    if t in _GREETINGS or t in _THANKS or t in _FAREWELLS or t in _WELLBEING:
        return QueryIntent(CONVERSATION, 0.95, False, "closed-class chat")
    if _looks_social(t):
        return QueryIntent(CONVERSATION, 0.85, False, "social frame")
    followup_reason = _looks_followup(t, context)
    if followup_reason:
        return QueryIntent(DOCUMENT_FOLLOWUP, 0.75, True, followup_reason)
    for pat in _CAPABILITY_SHORTCUTS:
        if re.search(pat, t):
            return QueryIntent(CAPABILITY, 0.8, False, f"capability {pat}")
    cap_reason = _looks_capability(t)
    if cap_reason:
        return QueryIntent(CAPABILITY, 0.8, False, cap_reason)
    for pat in _OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, t):
            return QueryIntent(OUT_OF_SCOPE, 0.7, False, f"off-topic {pat}")
    return QueryIntent(DOCUMENT_QUERY, 0.6, True, "default: substantive")


def expand_followup_query(message: str, context: Optional[List[Dict]],
                          max_chars: int = 200) -> str:
    """Anchor a followup for retrieval with topic selection.

    Continuations ("and payment?", "also…") extend the conversation TOPIC
    (oldest substantive question); reframes ("what about…") build on the
    most recent one. The anchor keeps natural sentence form — measured to
    retrieve better than term bags on MiniLM — truncated to max_chars.
    Assistant text is never used.
    """
    from conversation_memory import coerce_context

    subs = _substantive_user_questions(coerce_context(context))
    if not subs:
        return message.strip()
    first = _normalize(message or "")
    if first.startswith(("and ", "also ", "plus ")) and len(subs) > 1:
        prior = subs[0]
    else:
        prior = subs[-1]
    composed = f"{prior} {message.strip()}"
    return composed[:max_chars].strip() or message.strip()


# --- back-compat thin wrappers (route-first API used by backend/UI) ---

def route_query(text: str) -> str:
    """Legacy label: conversational/document_query/out_of_scope."""
    obs = observe_query(text)
    if obs.intent == CONVERSATION:
        return CONVERSATIONAL
    if obs.intent == OUT_OF_SCOPE:
        return OUT_OF_SCOPE
    return DOCUMENT_QUERY


def reply_for_route(text: str, route: str) -> str:
    """Canned reply for non-document routes (no retrieval, no LLM)."""
    t = _normalize(text)
    if route in (CONVERSATIONAL, CONVERSATION):
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
    if route == CAPABILITY:
        return ("I can answer questions about the documents connected to this "
                "assistant — for example contract terms, parties, dates, "
                "obligations, renewal and termination clauses. Just ask, "
                "e.g. “What is the lock-in period in the lease deed?”")
    if route == OUT_OF_SCOPE:
        return ("I'm mainly here to help with the documents connected to this "
                "assistant. You can ask me about contract terms, clauses, dates, "
                "parties, obligations, and other information in those documents.")
    raise ValueError(f"Unknown route: {route}")
