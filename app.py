import streamlit as st

from backend import (
    check_llm_status,
    get_knowledge_base_status,
    ingest_with_report,
    query_documents,
)
from config import get_config
from ui.components import (
    WELCOME_BODY,
    WELCOME_TITLE,
    copy_button_html,
    format_source_rows,
    friendly_error,
    load_styles,
)

# --- PAGE SETUP ---
st.set_page_config(page_title="RAG-4i — Document Assistant", layout="centered")
st.markdown(load_styles(), unsafe_allow_html=True)

cfg = get_config()

# --- SIDEBAR: KNOWLEDGE BASE SETUP (controls unchanged) ---
is_cloud = (getattr(cfg, "document_storage", "local") or "local").lower() == "s3"
with st.sidebar:
    st.header("Knowledge base")

    def _show_ingest_details(details):
        st.write(f"Documents found: **{details['found']}**")
        st.write(f"Pages processed: **{details['pages']}**")
        st.write(f"Chunks created: **{details['chunks']}**")
        st.write(f"Embeddings created: **{details['embeddings']}**")
        st.write(f"Vectors stored: **{details['vectors_stored']}**")
        st.write(f"Failed files: **{details['failed']}**")
        for item in details.get("failed_files", []):
            st.write(f"❌ {item['file']}: {item['reason']}")

    if is_cloud:
        st.write(f"S3 source: `s3://{cfg.s3_bucket}/{cfg.s3_prefix}`")
        st.caption("Cloud mode: ingestion reads from S3. Local filesystem "
                   "paths are not available here.")
        if st.button("Build/Update Database from S3"):
            with st.spinner("Reading from S3 and building index..."):
                success, message, details = ingest_with_report(None)
                if success:
                    st.success(message)
                else:
                    st.error(message)
                _show_ingest_details(details)
    else:
        st.write("Point this to your client's legal folder.")

        # Input for Folder Path (local mode only: path is on the app host).
        folder_path = st.text_input("Folder Path (local host only):",
                                    placeholder=r"D:\Clients\ABC_Ltd\Legal")
        if getattr(cfg, "app_env", "local") != "local":
            st.caption("Note: this path is on the machine running the app, "
                       "not your browser computer.")

        if st.button("Build/Update Database"):
            if folder_path:
                with st.spinner("Scanning documents and building index..."):
                    success, message, details = ingest_with_report(folder_path)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                    _show_ingest_details(details)
            else:
                st.warning("Please enter a folder path.")

    st.markdown("---")
    # Real status from actual checks (no hardcoded claims).
    st.write(f"Environment: **{cfg.app_env.capitalize()}**")

    kb = get_knowledge_base_status()
    kb_ready = bool(kb.get("ready"))
    if kb_ready:
        chunks = kb.get("chunk_count")
        docs = kb.get("document_count")
        parts = []
        if docs is not None:
            parts.append(f"{docs} docs")
        if chunks is not None:
            parts.append(f"{chunks} chunks")
        count_text = f" ({', '.join(parts)})" if parts else ""
        st.success(f"Knowledge Base: **Ready**{count_text}")
    elif kb.get("exists"):
        st.warning("Knowledge Base: **Empty** (no chunks indexed)")
    else:
        st.warning("Knowledge Base: **Not built**")

    llm = check_llm_status()
    if llm.get("reachable"):
        st.success(f"LLM: **Reachable** ({llm.get('model')})")
    else:
        st.error(f"LLM: **Unreachable** ({llm.get('base_url')})")

    st.caption(f"Vector store: {cfg.vector_store} @ {cfg.chroma_path}")
    storage_label = (getattr(cfg, "document_storage", "local") or "local")
    st.caption(f"Document storage: **{storage_label.capitalize()}**")

# --- HEADER: quiet identity + live document status ---
if kb_ready:
    _docline = "Documents ready"
    _counts = []
    if kb.get("document_count") is not None:
        _counts.append(f"{kb['document_count']} documents")
    if kb.get("chunk_count") is not None:
        _counts.append(f"{kb['chunk_count']} indexed chunks")
    if _counts:
        _docline += " · " + " · ".join(_counts)
else:
    _docline = "Document workspace"
st.markdown(
    f'<div class="rag-header"><h1>RAG-4i</h1>'
    f"<p>Ask questions about your connected documents.</p>"
    f'<p class="rag-docline">{_docline}</p></div>',
    unsafe_allow_html=True,
)

# --- MAIN CHAT INTERFACE ---
# Initialize chat history + bounded conversation memory (recent turns only).
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    from conversation_memory import ConversationMemory
    st.session_state.memory = ConversationMemory()
if "model_state" not in st.session_state:
    st.session_state.model_state = None

# --- MODEL GATE: real readiness, never simulated ---
if st.session_state.model_state is None:
    st.markdown(
        '<div class="rag-welcome"><h2>RAG-4i</h2>'
        "<p>Your document intelligence workspace.</p>"
        "<p>Loading the assistant model…</p></div>",
        unsafe_allow_html=True,
    )
    from model_warmup import READY, warmup
    with st.spinner("Loading model…"):
        result = warmup(cfg)
    st.session_state.model_state = result
    st.rerun()

_model = st.session_state.model_state
model_ready = isinstance(_model, dict) and _model.get("state") == "ready"
if not model_ready:
    detail = _model.get("detail", "The assistant is unavailable.") \
        if isinstance(_model, dict) else "The assistant is unavailable."
    st.markdown(
        '<div class="rag-welcome"><h2>RAG-4i</h2>'
        f"<p>{detail}</p></div>",
        unsafe_allow_html=True,
    )
    if st.button("Retry connection"):
        st.session_state.model_state = None
        st.toast("Retrying connection…")
        st.rerun()
    st.stop()

# Display previous chat messages (decorations re-render from stored
# metadata every run, so reruns never wipe labels/sources/copy buttons).
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("label"):
            st.markdown(f"<p class='rag-answer-label'>{message['label']}</p>",
                        unsafe_allow_html=True)
        if message.get("notice") == "no-context":
            st.warning(
                f"Retrieved **0 chunks** above the relevance "
                f"threshold ({cfg.relevance_threshold}) "
                f"(top-k={cfg.retrieval_k}). The answer below is "
                f"the no-context fallback, not a grounded answer."
            )
        st.markdown(message["content"])
        if message.get("sources"):
            import streamlit.components.v1 as _components

            _rows = format_source_rows(message["sources"])
            _bits = []
            if _rows:
                _first = _rows[0]
                _bits.append(_first["file_name"] or "unknown")
                if _first["page"] is not None:
                    _bits.append(f"p. {_first['page']}")
                _bits.append(f"relevance {_first['score_text']}")
            with st.expander(f"Sources · {' · '.join(_bits)}" if _rows else "Sources"):
                for _row in _rows:
                    _page = f" · p. {_row['page']}" if _row["page"] is not None else ""
                    st.markdown(
                        f'<div class="rag-source-row">{_row["file_name"] or "unknown"}'
                        f"{_page} <span class='rag-source-score'>· relevance "
                        f"{_row['score_text']}</span></div>",
                        unsafe_allow_html=True,
                    )
            _components.html(
                copy_button_html(message["content"],
                                 f"copy_{message.get('seq', 0)}"),
                height=44,
            )


def _initial_suggestions():
    """Starter questions from the real index (generic fallback if empty)."""
    from suggestions import initial_suggestions
    try:
        from vector_store import get_vector_store
        from embeddings import get_embedding_provider
        vs = get_vector_store(cfg, get_embedding_provider(cfg))
        files = [s["file_name"] for s in vs.list_sources()]
    except Exception:
        files = []
    return initial_suggestions(files)


def _handle_prompt(prompt_text):
    """Single conversation pipeline for typed and suggested questions.

    State-only: appends to session_state; the display loop above renders
    everything, so reruns never wipe labels/sources/copy buttons.
    """
    if "msg_seq" not in st.session_state:
        st.session_state.msg_seq = 0

    def _push(role, content, label=None, sources=None, notice=None):
        st.session_state.msg_seq += 1
        st.session_state.messages.append({
            "role": role, "content": content, "seq": st.session_state.msg_seq,
            "label": label, "sources": sources or [], "notice": notice,
        })

    # 1. Record User Message
    _push("user", prompt_text)
    _count_before = len(st.session_state.messages)

    # 2. Generate Assistant Response (honest staged loading).
    with st.status("Working on your question…", expanded=False) as status:
        def _phase(name):
            status.update(
                label="Finding relevant information…"
                if name == "retrieving" else "Generating response…")

        try:
            kb = get_knowledge_base_status()
            if not kb.get("ready"):
                status.update(label="Knowledge base not ready", state="error")
                st.error("⚠️ Knowledge Base is not ready. Please build it in the sidebar first.")
                return
            from query_router import observe_query
            from ui.components import response_label
            memory = st.session_state.memory
            memory.add("user", prompt_text)
            history = memory.as_context()
            needs_retrieval = observe_query(prompt_text, history).needs_retrieval
            response_text, sources = query_documents(
                prompt_text, conversation_context=history, on_phase=_phase)

            # Explicit retrieval status for document questions only;
            # routed replies (chat/out-of-scope) intentionally skip retrieval.
            notice = None
            if not sources and needs_retrieval:
                notice = "no-context"
                display_text = friendly_error(response_text)
            else:
                display_text = response_text

            # Sources: quiet expander, citation data preserved exactly.
            if sources:
                rows = format_source_rows(sources)
                source_text = ""
                for row in rows:
                    label = row["file_name"] or "unknown"
                    if row["page"] is not None:
                        label += f" (p. {row['page']})"
                    label += f" [{row['score_text']}]"
                    source_text += label + ", "
                source_text = f"\n\n**Sources:** *{source_text.rstrip(', ')}*" if rows else ""
            else:
                source_text = ""

            full_response = display_text + source_text

            # 3. Record Assistant Message (label/sources/notice ride along).
            _label = response_label(prompt_text, sources, needs_retrieval)
            _push("assistant", full_response, label=_label,
                  sources=sources, notice=notice)
            st.session_state.memory.add("assistant", full_response)
            st.session_state.pop("last_failed", None)

            # 4. Persist for top-level suggestion rendering (buttons must
            # exist on every rerun, not only inside prompt handling).
            from suggestions import remember_followups
            remember_followups(st.session_state, prompt_text, sources)
            status.update(label="Done", state="complete")

        except Exception:
            # No raw stack trace for normal users; details go to console/log.
            print("UI query failed (see logs for details).")
            status.update(label="Something went wrong", state="error")
            st.session_state.last_failed = prompt_text
            st.error(friendly_error("An error occurred while answering. "
                                    "Make sure LM Studio Server is running!"))

    # Refresh only when new messages exist (errors stay visible as-is).
    if len(st.session_state.messages) > _count_before:
        st.rerun()


# Retry card for the last failed question (same pipeline, no new logic).
if st.session_state.get("last_failed") and not st.session_state.get("pending_prompt"):
    st.markdown("<p class='rag-prompt-label'>The last answer failed.</p>",
                unsafe_allow_html=True)
    if st.button("Retry answer", key="retry_last"):
        st.session_state.pending_prompt = st.session_state.pop("last_failed")
        st.rerun()


# Follow-up buttons for the latest grounded answer. Rendered at top level
# on EVERY rerun so clicks always materialize (this was the bug: buttons
# previously existed only inside prompt handling, so clicks were lost).
from suggestions import current_followups as _current_followups
_followup_specs = _current_followups(st.session_state)
if _followup_specs:
    st.markdown("**You may also ask:**")
    _cols = st.columns(len(_followup_specs))
    for _col, _spec in zip(_cols, _followup_specs):
        if _col.button(_spec["label"], key=_spec["key"]):
            if "pending_prompt" not in st.session_state:
                st.session_state.pending_prompt = _spec["label"]
            st.rerun()

# Welcome hero + starter suggestions for a fresh conversation.
if not st.session_state.messages:
    st.markdown(
        f'<div class="rag-welcome"><h2>{WELCOME_TITLE}</h2><p>{WELCOME_BODY}</p></div>',
        unsafe_allow_html=True,
    )
    try:
        if get_knowledge_base_status().get("ready"):
            st.markdown('<p class="rag-prompt-label">Try asking</p>',
                        unsafe_allow_html=True)
            starters = _initial_suggestions()
            cols = st.columns(len(starters))
            for i, (col, sug) in enumerate(zip(cols, starters)):
                if col.button(sug, key=f"starter_{i}"):
                    st.session_state.pending_prompt = sug
                    st.rerun()
        else:
            st.info("Add documents using the Knowledge base panel, "
                    "then come back and ask away.")
    except Exception:
        print("Starter suggestions unavailable (see logs).")

# Handle User Input (typed or clicked suggestion, one pipeline).
pending = st.session_state.pop("pending_prompt", None)
typed = st.chat_input(
    "Ex: What is the lock-in period in the lease deed?",
    disabled=not (kb_ready and model_ready),
)
if pending or typed:
    _handle_prompt(pending or typed)
