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

# Prefixes of "<think>" / "</think>" (case-insensitive). A trailing buffer
# matching one of these may be a tag split across chunks, so it is held
# back until more input arrives. Anything else starting with "<" is
# emitted immediately (e.g. "a < b").
_THINK_PREFIXES = tuple(
    "<think"[:i].lower() for i in range(1, len("<think") + 1)
) + tuple(
    "</think"[:i].lower() for i in range(1, len("</think") + 1)
)


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


class ThinkStreamSanitizer:
    """Incremental think-block filter with chunk-boundary safety.

    Buffers enough state to guarantee forbidden content is never emitted
    prematurely: complete think blocks are removed, reasoning between an
    open and close tag is held, and a trailing fragment that could still
    become a tag is withheld until the next chunk (or flush()).

    Usage: feed() each raw chunk and render what it returns; call flush()
    at stream end and render that too. The concatenation equals what
    strip_think_blocks() produces for the whole text (modulo blank-line
    collapsing, applied once by the caller on the final answer).
    """

    def __init__(self):
        self._buf = ""
        self._in_think = False

    def feed(self, chunk: str):
        """Return newly-safe text for one raw chunk (may be "")."""
        data = self._buf + (chunk or "")
        self._buf = ""
        out = []
        while data:
            if self._in_think:
                end = data.lower().find("</think>")
                if end == -1:
                    self._buf = data
                    return "".join(out)
                data = data[end + len("</think>"):]
                self._in_think = False
                continue
            low = data.lower()
            idx = low.find("<think")
            close_only = low.find("</think")
            if idx == -1 and close_only == -1:
                # No tag yet, but a trailing "<..." may still grow into
                # one (split across chunks): hold just that tail.
                tail = low[low.rfind("<"):] if "<" in low else ""
                if tail and tail in _THINK_PREFIXES:
                    out.append(data[:len(data) - len(tail)])
                    self._buf = data[len(data) - len(tail):]
                    return "".join(out)
                out.append(data)
                return "".join(out)
            # Earliest tag-like event wins.
            if close_only != -1 and (idx == -1 or close_only < idx):
                out.append(data[:close_only])
                data = data[close_only + len("</think>"):]
                continue
            # Complete open tag present? Only "<think>" (optional attrs)
            # counts; lookalikes like "<thinker>" stay literal text.
            rest = low[idx:]
            tag = re.match(r"<think(\s[^>]*)?>", rest)
            if tag:
                out.append(data[:idx])
                data = data[idx + tag.end():]
                self._in_think = True
                continue
            # Hold only if the tail could still grow into a tag
            # ("<thi" + "nk>" later); lookalikes like "<thinker>"
            # emit immediately as literal text.
            tail = data[idx:]
            if tail.lower() in _THINK_PREFIXES:
                out.append(data[:idx])
                self._buf = tail
                return "".join(out)
            out.append(data)
            return "".join(out)
            # Trailing fragment that could still become a tag: hold it.
            tail = data[idx:]
            if tail.lower() in _THINK_PREFIXES:
                out.append(data[:idx])
                self._buf = tail
                return "".join(out)
            # A "<" that cannot become a think tag: literal text.
            out.append(data)
            return "".join(out)
        return "".join(out)

    def flush(self):
        """Emit the remainder.

        A still-open think block at stream end is dropped (it is
        indistinguishable from reasoning in progress). A merely held
        tag PREFIX ("<thi") is emitted literally. This deliberately
        differs from strip_think_blocks() for the open-block case:
        during streaming, emitting it would risk leaking reasoning
        whose close tag simply hasn't arrived yet.
        """
        tail, self._buf = self._buf, ""
        if self._in_think:
            self._in_think = False
            return ""
        return tail
