import os

import streamlit as st

from backend import (
    check_llm_status,
    create_vector_db_from_folder,
    get_knowledge_base_status,
    query_documents,
)
from config import get_config

# --- PAGE SETUP ---
st.set_page_config(page_title="CA Legal Assistant", layout="wide")
st.title("⚖️ CA Firm Legal Document Assistant")
st.markdown("---")

cfg = get_config()

# --- SIDEBAR: KNOWLEDGE BASE SETUP ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    st.write("Point this to your client's legal folder.")

    # Input for Folder Path
    folder_path = st.text_input("Folder Path:", placeholder=r"D:\Clients\ABC_Ltd\Legal")

    if st.button("Build/Update Database"):
        if folder_path:
            with st.spinner("Scanning documents and building index..."):
                success, message = create_vector_db_from_folder(folder_path)
                if success:
                    st.success(message)
                else:
                    st.error(message)
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

# --- MAIN CHAT INTERFACE ---
st.subheader("💬 Ask a question about the documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ex: What is the lock-in period in the lease deed?"):
    # 1. Show User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Assistant Response
    with st.spinner("Analyzing legal context..."):
        try:
            kb = get_knowledge_base_status()
            if not kb.get("ready"):
                st.error("⚠️ Knowledge Base is not ready. Please build it in the sidebar first.")
            else:
                response_text, sources = query_documents(prompt)

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

        except Exception:
            # No raw stack trace for normal users; details go to console/log.
            print("UI query failed (see logs for details).")
            st.error("An error occurred while answering. Make sure LM Studio Server is running!")
