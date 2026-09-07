import os

import streamlit as st

from backend import (
    check_llm_status,
    get_knowledge_base_status,
    ingest_with_report,
    query_documents,
)
from config import get_config

# --- PAGE SETUP ---
st.set_page_config(page_title="CA Legal Assistant", layout="wide")
st.title("⚖️ CA Firm Legal Document Assistant")
st.markdown("---")

cfg = get_config()

# --- SIDEBAR: KNOWLEDGE BASE SETUP ---
is_cloud = (getattr(cfg, "document_storage", "local") or "local").lower() == "s3"
with st.sidebar:
    st.header("📂 Knowledge Base")

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
    if kb.get("ready"):
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

# --- MAIN CHAT INTERFACE ---
st.subheader("💬 Ask a question about the documents")

# Initialize chat history + bounded conversation memory (recent turns only).
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    from conversation_memory import ConversationMemory
    st.session_state.memory = ConversationMemory()

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


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
    """Single conversation pipeline for typed and suggested questions."""
    # 1. Show User Message
    st.chat_message("user").markdown(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    # 2. Generate Assistant Response
    with st.spinner("Analyzing legal context..."):
        try:
            kb = get_knowledge_base_status()
            if not kb.get("ready"):
                st.error("⚠️ Knowledge Base is not ready. Please build it in the sidebar first.")
                return
            from query_router import observe_query
            memory = st.session_state.memory
            memory.add("user", prompt_text)
            history = memory.as_context()
            needs_retrieval = observe_query(prompt_text, history).needs_retrieval
            response_text, sources = query_documents(
                prompt_text, conversation_context=history)

            # Explicit retrieval status for document questions only;
            # routed replies (chat/out-of-scope) intentionally skip retrieval.
            if not sources and needs_retrieval:
                st.warning(
                    f"Retrieved **0 chunks** above the relevance "
                    f"threshold ({cfg.relevance_threshold}) "
                    f"(top-k={cfg.retrieval_k}). The answer below is "
                    f"the no-context fallback, not a grounded answer."
                )

            # Structured sources: filename + page + score.
            if sources:
                parts = []
                seen = set()
                for s in sources:
                    key = (s.get("file_name"), s.get("page"))
                    if key in seen:
                        continue
                    seen.add(key)
                    label = s.get("file_name") or "unknown"
                    if s.get("page") is not None:
                        label += f" (p. {s.get('page')})"
                    if s.get("score") is not None:
                        label += f" [{s.get('score'):.2f}]"
                    parts.append(label)
                source_text = f"\n\n**Sources:** *{', '.join(parts)}*" if parts else ""
            else:
                source_text = ""

            full_response = response_text + source_text

            # 3. Show Assistant Message
            with st.chat_message("assistant"):
                st.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.memory.add("assistant", full_response)

                # 4. Persist for top-level suggestion rendering (buttons must
                # exist on every rerun, not only inside prompt handling).
                from suggestions import remember_followups
                remember_followups(st.session_state, prompt_text, sources)

        except Exception:
            # No raw stack trace for normal users; details go to console/log.
            print("UI query failed (see logs for details).")
            st.error("An error occurred while answering. Make sure LM Studio Server is running!")

    # Refresh so fresh suggestion buttons render immediately.
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

# Starter suggestions for a fresh conversation (from the real index).
if not st.session_state.messages:
    try:
        if get_knowledge_base_status().get("ready"):
            st.markdown("**Try asking:**")
            starters = _initial_suggestions()
            cols = st.columns(len(starters))
            for col, sug in zip(cols, starters):
                if col.button(sug, key=f"starter_{sug[:16]}"):
                    st.session_state.pending_prompt = sug
                    st.rerun()
    except Exception:
        print("Starter suggestions unavailable (see logs).")

# Handle User Input (typed or clicked suggestion, one pipeline).
pending = st.session_state.pop("pending_prompt", None)
typed = st.chat_input("Ex: What is the lock-in period in the lease deed?")
if pending or typed:
    _handle_prompt(pending or typed)
