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


def excerpt_text(content: str, limit: int = 500) -> str:
    """Compact excerpt: word-boundary trim with ellipsis when cut."""
    text = re.sub(r"\s+", " ", content or "").strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    if cut < limit // 2:
        cut = limit
    return text[:cut].rstrip() + "…"


def source_row_html(row) -> str:
    """One escaped source row (filename/page/score + chunk preview)."""
    name = escape(row["file_name"]) if row["file_name"] else "unknown"
    page = f" · p. {int(row['page'])}" if row["page"] is not None else ""
    out = (f'<div class="rag-source-row">{name}{page} '
           f"<span class='rag-source-score'>· relevance "
           f"{row['score_text']}</span></div>")
    if row.get("content"):
        out += (f'<div class="rag-source-chunk">'
                f'{escape(excerpt_text(row["content"]))}</div>')
    return out


def full_source_html(row) -> str:
    """Full cited page text (escaped). Only filename/page/score/content
    are rendered — never storage paths or other metadata."""
    name = escape(row["file_name"]) if row["file_name"] else "unknown"
    page = f" · p. {int(row['page'])}" if row["page"] is not None else ""
    out = (f'<div class="rag-source-row">{name}{page} '
           f"<span class='rag-source-score'>· relevance "
           f"{row['score_text']}</span></div>")
    if row.get("content"):
        # Note: the sub result lives outside the f-string: backslash
        # sequences inside f-string expressions are a SyntaxError on the
        # EC2 Python 3.11 (allowed only from 3.12).
        cleaned = escape(re.sub(r"\s+", " ", row["content"]).strip())
        out += (f'<div class="rag-source-chunk">'
                f'{cleaned}</div>')
    return out


def sibling_pages(sources, file_name, exclude_page=None):
    """Distinct cited pages for a file within one answer's sources.

    Pure helper for the page viewer (retrieved evidence only).
    Order kept, duplicates removed.
    """
    pages = []
    for s in sources or []:
        if (s or {}).get("file_name") != file_name:
            continue
        page = (s or {}).get("page")
        if page == exclude_page or page in pages:
            continue
        pages.append(page)
    return pages


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


def retrieval_strength(sources) -> str | None:
    """Compact retrieval-strength label from the top relevance score.

    Terminology is deliberate: a similarity score, not a probability of
    correctness. Returns None when no scored source exists.
    """
    scores = [s.get("score") for s in (sources or [])
              if isinstance(s.get("score"), (int, float))]
    if not scores:
        return None
    return f"Retrieval strength · {max(scores):.2f}"


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


def ask_about_document_question(file_name: str) -> str:
    """Normal-pipeline question for 'Ask about this document'.

    Contains the real filename so the existing broad-scope/file path
    handles it; no special retrieval, no extra LLM call.
    """
    name = (file_name or "").strip() or "this document"
    return f"Tell me more about {name}"


def document_welcome_summary(sources, limit: int = 6) -> str | None:
    """Compact 'N documents indexed' + filename list for the welcome state.

    Uses existing list_sources() output only; None when nothing indexed.
    Filenames are returned raw (callers escape for HTML; Streamlit text
    rendering escapes automatically).
    """
    names = []
    for s in sources or []:
        name = (s or {}).get("file_name")
        if name and name not in names:
            names.append(name)
    if not names:
        return None
    shown = names[:max(1, limit)]
    n = len(names)
    noun = "document" if n == 1 else "documents"
    line1 = f"{n} {noun} indexed"
    line2 = " · ".join(shown)
    if len(names) > len(shown):
        line2 += f" · +{len(names) - len(shown)} more"
    return f"{line1}\n{line2}"


def localstorage_saver_html(payload_json: str, element_id: str = "rag-restore") -> str:
    """One-way browser-local snapshot saver (no Python read-back).

    Writes the serialized conversation to localStorage on every render.
    Bound by explicit element ID; no Streamlit-internal selectors.
    Restore itself stays explicit + Python-side (paste/choice) so there
    is no fragile JS→Python bridge and no silent data flow.
    """
    import json

    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", element_id or "rag-restore")
    # payload_json is already JSON; embed safely as JS string via dumps.
    blob = json.dumps(payload_json or "{}")
    return (
        f"<div id='{safe_id}' style='display:none'></div>"
        "<script>(function(){"
        f"var el=document.getElementById('{safe_id}');"
        "if(!el||el.dataset.done)return;el.dataset.done='1';"
        f"var raw={blob};"
        "try{"
        "if(raw&&raw!=='{}'){localStorage.setItem('rag4i.backup.v1',raw);}"
        "}catch(e){}"
        "})();</script>"
    )


def localstorage_clearer_html(element_id: str = "rag-restore-clear") -> str:
    """One-shot localStorage clearer for New Chat / Start fresh."""
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", element_id or "rag-restore-clear")
    return (
        f"<div id='{safe_id}' style='display:none'></div>"
        "<script>(function(){"
        f"var el=document.getElementById('{safe_id}');"
        "if(!el||el.dataset.done)return;el.dataset.done='1';"
        "try{localStorage.removeItem('rag4i.backup.v1');}catch(e){}"
        "})();</script>"
    )


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
