"""Presentation helpers for the RAG-4i frontend.

Pure functions only (no Streamlit calls) so they are unit-testable.
Backend data passes through untouched; only wording/styling adapts.
"""

import html
import os
import re

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


def load_styles(theme: str = "auto") -> str:
    """Combined <style> block: tokens first, then section stylesheets."""
    from ui.tokens import css_variables

    parts = [css_variables("dark" if theme == "dark" else "auto")]
    for name in ("base.css", "chat.css"):
        path = os.path.join(_BASE, "css", name)
        with open(path, encoding="utf-8") as f:
            parts.append(f.read())
    return "<style>\n" + "\n".join(parts) + "\n</style>"


def escape(text: str) -> str:
    return html.escape(text or "")


def format_source_rows(sources):
    """Source rows preserving filename/page/score exactly.

    Filenames come from document metadata (untrusted input) and are
    HTML-escaped here; scores render with fixed 2-decimal formatting.
    """
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
            "content": s.get("content", "") or "",
        })
    return rows


def source_row_html(row) -> str:
    """One escaped source row (filename/page/score + chunk preview)."""
    name = escape(row["file_name"]) if row["file_name"] else "unknown"
    page = f" · p. {int(row['page'])}" if row["page"] is not None else ""
    out = (f'<div class="rag-source-row">{name}{page} '
           f"<span class='rag-source-score'>· relevance "
           f"{row['score_text']}</span></div>")
    if row.get("content"):
        out += (f'<div class="rag-source-chunk">'
                f'{escape(row["content"][:500])}</div>')
    return out


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


def export_chat_markdown(messages) -> str:
    """Conversation export (user/assistant text + cited sources)."""
    lines = ["# RAG-4i conversation", ""]
    for message in messages or []:
        role = message.get("role", "")
        lines.append("## User" if role == "user" else "## Assistant")
        lines.append("")
        lines.append(message.get("content", "") or "")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def response_label(prompt: str, sources, needs_retrieval: bool):
    """Small answer-type label, or None for plain conversation.

    Recomputes support from the same evidence (no backend change):
    grounded -> "Grounded answer", scoped -> "Partial answer",
    no evidence -> "Not enough context".
    """
    if not needs_retrieval:
        return None
    if not sources:
        return "Not enough context"
    from answer_support import DIRECT, assess_support
    level = assess_support(prompt, sources)["level"]
    return "Grounded answer" if level == DIRECT else "Partial answer"


def copy_button_html(text: str, key: str) -> str:
    """Vanilla-JS copy button bound by explicit element ID.

    The button carries id="rag-copy-<key>" and looks itself up — no
    sibling-DOM assumptions. Clipboard API with legacy fallback.
    """
    import json

    payload = json.dumps(text or "")
    safe_key = re.sub(r"[^A-Za-z0-9_-]", "_", key or "x")
    element_id = f"rag-copy-{safe_key}"
    return (
        f"<button class='rag-copy' id='{element_id}' "
        f"data-payload='{html.escape(payload)}'>Copy answer</button>"
        "<script>(function(){"
        f"var b=document.getElementById('{element_id}');"
        "if(!b||b.dataset.done)return;b.dataset.done='1';"
        "b.addEventListener('click',function(){"
        "var t=JSON.parse(b.dataset.payload);"
        "function ok(){b.textContent='Copied';"
        "setTimeout(function(){b.textContent='Copy answer';},1500);}"
        "if(navigator.clipboard&&navigator.clipboard.writeText){"
        "navigator.clipboard.writeText(t).then(ok,function(){legacy();});}"
        "else{legacy();}"
        "function legacy(){var a=document.createElement('textarea');"
        "a.value=t;document.body.appendChild(a);a.select();"
        "try{document.execCommand('copy');ok();}catch(e){}a.remove();}"
        "});})();</script>"
    )
