"""Bounded conversation memory for the intent observer.

Keeps a small recent-turn window (default 3 turns = 6 messages) used ONLY to:
  - recognize follow-ups ("why?", "tell me more", "what about renewal?"),
  - anchor a follow-up retrieval query to the last substantive user
    question (prior question + current message).

Guardrails against contamination:
  - Out-of-scope and conversational turns never expand retrieval (only
    DOCUMENT_FOLLOWUP expands, and only with the single most recent
    substantive user question).
  - Short/greeting turns (<3 content words) are ignored as anchors.
  - Documents remain the source of truth: the previous ASSISTANT answer is
    never used as retrieval evidence or answer content.
"""

from collections import deque
from typing import Dict, List, Optional, Union
import re

_DEFAULT_TURNS = 3


def _content_len(text: str) -> int:
    stop = {"a", "an", "the", "is", "are", "was", "it", "that", "this",
            "to", "of", "in", "on", "for", "do", "does", "me", "more",
            "about", "too", "there", "s", "t", "hi", "hello", "hey",
            "thanks", "thank", "you", "please"}
    return len([w for w in re.findall(r"[a-z']+", (text or "").lower())
                if w not in stop])


class ConversationMemory:
    """Fixed-size recent-turn window with focus helpers."""

    def __init__(self, max_turns: int = _DEFAULT_TURNS):
        self._messages = deque(maxlen=max(1, max_turns) * 2)

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content or ""})

    def reset(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def as_context(self) -> List[Dict]:
        """Plain list form accepted everywhere a context list is."""
        return list(self._messages)

    def had_document_exchange(self) -> bool:
        for msg in reversed(self._messages):
            if msg.get("role") == "assistant" and len(msg.get("content", "")) > 40:
                return True
        return False

    def last_user_document_question(self) -> str:
        """Most recent substantive user message (greetings skipped)."""
        for msg in reversed(self._messages):
            if msg.get("role") == "user" and _content_len(msg.get("content", "")) >= 3:
                return msg.get("content", "").strip()
        return ""


def coerce_context(context: Optional[Union[List[Dict], ConversationMemory]]
                   ) -> List[Dict]:
    """Accept a ConversationMemory or a plain list; always return a list."""
    if context is None:
        return []
    if hasattr(context, "as_context"):
        return context.as_context()
    return list(context)
