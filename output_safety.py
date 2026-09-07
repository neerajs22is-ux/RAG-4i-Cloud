"""Output safety: normalize raw LLM text at the application boundary.

Currently handles Qwen `<think>...</think>` reasoning blocks. Separated
from retrieval and routing logic on purpose.

Rules:
  - Remove every complete <think>...</think> block (case-insensitive).
  - Collapse leftover blank lines; preserve the final answer verbatim.
  - Ordinary text (including the word "think") is never touched.
  - An unclosed trailing <think> is left intact (visible) rather than
    risk deleting legitimate content.
  - A response containing ONLY think blocks yields "" so callers treat it
    as an invalid/empty model response. Reasoning is never summarized.
"""

import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_BLANK_RE = re.compile(r"\n{3,}")


def strip_think_blocks(text: str) -> str:
    """Remove complete think blocks; return the cleaned answer."""
    if not text:
        return ""
    cleaned = _THINK_RE.sub("", text)
    cleaned = _BLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def is_empty_response(text: str) -> bool:
    """True when nothing usable remains after normalization."""
    return not strip_think_blocks(text or "")
