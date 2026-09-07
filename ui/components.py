"""Presentation helpers for the RAG-4i frontend.

Pure functions only (no Streamlit calls) so they are unit-testable.
Backend data passes through untouched; only wording/styling adapts.
"""

import html
import os

_BASE = os.path.dirname(os.path.abspath(__file__))

WELCOME_TITLE = "RAG-4i"
WELCOME_BODY = (
    "Your document intelligence workspace. Ask questions about the "
    "connected legal documents and get grounded answers with sources."
)

# Backend message -> human wording. Backend strings are unchanged;
# unknown messages pass through verbatim (never masked silently).
ERROR_COPY = {
    "unreachable": "The document search is available, but the assistant "
                   "is temporarily unable to generate a response.",
    "no-context": "I couldn't find enough information in the connected "
                  "documents to answer that.",
}


def load_styles() -> str:
    """Combined <style> block: tokens first, then section stylesheets."""
    from ui.tokens import css_variables

    parts = [css_variables()]
    for name in ("base.css", "chat.css"):
        path = os.path.join(_BASE, "css", name)
        with open(path, encoding="utf-8") as f:
            parts.append(f.read())
    return "<style>\n" + "\n".join(parts) + "\n</style>"


def escape(text: str) -> str:
    return html.escape(text or "")


def format_source_rows(sources):
    """Source rows preserving filename/page/score exactly."""
    rows = []
    seen = set()
    for s in sources or []:
        key = (s.get("file_name"), s.get("page"))
        if key in seen:
            continue
        seen.add(key)
        score = s.get("score")
        rows.append({
            "file_name": s.get("file_name"),
            "page": s.get("page"),
            "score": score,
            "score_text": f"{score:.2f}" if score is not None else "—",
        })
    return rows


def friendly_error(backend_message: str) -> str:
    """Human wording for known backend failures; passthrough otherwise."""
    text = backend_message or ""
    low = text.lower()
    if "unreachable" in low or "lm studio" in low:
        return ERROR_COPY["unreachable"]
    if "could not find enough" in low or "cannot find this information" in low:
        return ERROR_COPY["no-context"]
    return text


def suggestion_button_label(text: str, limit: int = 80) -> str:
    """Full suggestion text (Streamlit wraps); shortened only for keys."""
    return text or ""
