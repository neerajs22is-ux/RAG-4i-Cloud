import logging
import time
import streamlit as st

logger = logging.getLogger(__name__)

from backend import (
    check_llm_status,
    generate_answer,
    get_knowledge_base_status,
    ingest_with_report,
)
from config import get_config
from ui.components import (
    WELCOME_BODY,
    WELCOME_TITLE,
    ask_about_document_question,
    copy_button_html,
    document_welcome_summary,
    export_chat_markdown,
    format_source_rows,
    friendly_error,
    load_styles,
    localstorage_clearer_html,
    localstorage_saver_html,
    source_row_html,
)

# --- PAGE SETUP ---
st.set_page_config(page_title="RAG-4i — Document Assistant", layout="centered")

cfg = get_config()

# Theme: OS preference by default, toggle persists for the session.
if "theme" not in st.session_state:
    st.session_state.theme = "auto"
st.markdown(load_styles("dark" if st.session_state.theme == "dark" else "auto"),
            unsafe_allow_html=True)

# --- SIDEBAR: KNOWLEDGE BASE SETUP (controls unchanged) ---
is_cloud = (getattr(cfg, "document_storage", "local") or "local").lower() == "s3"
with st.sidebar:
    # Live checks first so the one-line summary below uses real data.
    _kb_summary = get_knowledge_base_status()
    _llm_summary = check_llm_status()
    _storage_label = (getattr(cfg, "document_storage", "local") or "local")

    with st.expander("Knowledge base & status", expanded=False):
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

        def _run_ingest(label, folder):
            """Run ingestion with live per-document progress (real loop only)."""
            progress = st.progress(0, text="Starting…")
            status = st.status("Preparing…", expanded=False)

            def _on_progress(current, total, filename, outcome):
                progress.progress(current / total if total else 0,
                                  text=f"{current} / {total} documents processed")
                status.update(label=f"Processing {filename}…")

            try:
                success, message, details = ingest_with_report(
                    folder, on_progress=_on_progress)
            finally:
                progress.empty()
                status.update(label="Ingestion finished", state="complete")
            if success:
                st.success(message)
            else:
                st.error(message)
            _show_ingest_details(details)

        if is_cloud:
            st.write(f"S3 source: `s3://{cfg.s3_bucket}/{cfg.s3_prefix}`")
            st.caption("Cloud mode: ingestion reads from S3. Local filesystem "
                       "paths are not available here.")
            if st.button("Build/Update Database from S3"):
                _run_ingest("s3", None)
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
                    _run_ingest("local", folder_path)
                else:
                    st.warning("Please enter a folder path.")

        st.markdown("---")
        # Real status from actual checks (no hardcoded claims).
        st.write(f"Environment: **{cfg.app_env.capitalize()}**")

        kb = _kb_summary
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

        llm = _llm_summary
        if llm.get("reachable"):
            st.success(f"LLM: **Reachable** ({llm.get('model')})")
        else:
            st.error(f"LLM: **Unreachable** ({llm.get('base_url')})")

        st.caption(f"Vector store: {cfg.vector_store} @ {cfg.chroma_path}")
        storage_label = (getattr(cfg, "document_storage", "local") or "local")
        st.caption(f"Document storage: **{storage_label.capitalize()}**")

        # Unified recovery hint (same readiness model as the main area;
        # actionable, no stack traces, retry guidance included).
        try:
            from readiness import recovery_message, summarize_readiness
            _side_readiness = summarize_readiness(
                st.session_state.get("model_state"), _kb_summary)
            if _side_readiness.get("state") != "READY":
                _rec = recovery_message(
                    _side_readiness, _kb_summary,
                    st.session_state.get("model_state"), is_cloud)
                st.caption(f"⚠ {_rec['title']}: {_rec['what']} {_rec['next']}")
        except Exception:
            logger.warning("Readiness summary unavailable (see logs).")

    # Persistent one-line summary (outside the expander, always visible).
    # Driven by the unified readiness model (same source as welcome/composer).
    try:
        from readiness import summarize_readiness as _summ
        _unified = _summ(st.session_state.get("model_state"), _kb_summary)
        _ustate = _unified.get("state")
    except Exception:
        _unified = {"state": "DEGRADED" if _kb_summary.get("error") else
                    ("READY" if kb_ready else "KB_NOT_READY")}
        _ustate = _unified["state"]
    if kb_ready:
        _kb_word = "Ready"
    elif kb.get("exists"):
        _kb_word = "Empty"
    else:
        _kb_word = "Not built"
    st.caption(f"KB: {_kb_word} · LLM: "
               f"{'Reachable' if llm.get('reachable') else 'Unreachable'} · "
               f"{cfg.vector_store} + {storage_label.capitalize()}")

    with st.expander("Manage indexed documents"):
        st.caption("Removing a document deletes its indexed vectors. "
                   "Source files are left untouched.")
        try:
            from vector_store import get_vector_store
            from embeddings import get_embedding_provider
            _vs = get_vector_store(cfg, get_embedding_provider(cfg))
            _srcs = _vs.list_sources()
        except Exception:
            logger.warning("Source listing unavailable (see logs).")
            _srcs = []
        if not _srcs:
            st.caption("No indexed documents found.")
        for _s in _srcs:
            _fname = _s.get("file_name") or "unknown"
            _col_a, _col_b = st.columns([3, 1])
            _col_a.write(_fname)
            if _col_b.button("Remove", key=f"rm_{_fname}",
                             help=f"Delete indexed vectors for {_fname}."):
                from backend import delete_indexed_document
                _res = delete_indexed_document(_s.get("document_id"))
                st.toast(f"Removed {_res['vectors_removed']} vectors for {_fname}.")
                st.rerun()

    st.markdown("---")
    _theme_options = ["auto", "light", "dark"]
    _current = st.session_state.theme if st.session_state.theme in _theme_options else "auto"
    _picked = st.radio("Appearance", _theme_options,
                       index=_theme_options.index(_current),
                       help="Follow your system setting or force light/dark.",
                       horizontal=True)
    if _picked != st.session_state.theme:
        st.session_state.theme = _picked
        st.rerun()

    st.markdown("---")
    if st.session_state.get("messages"):
        from ui.components import export_chat_markdown
        st.download_button(
            "Export chat",
            data=export_chat_markdown(st.session_state.messages),
            file_name="rag4i-chat.md",
            mime="text/markdown",
            help="Download this conversation as Markdown.",
        )
    if st.button("New chat", help="Clear messages and conversation memory."):
        st.session_state.messages = []
        from conversation_memory import ConversationMemory
        st.session_state.memory = ConversationMemory()
        for _key in ("last_followups", "last_failed", "pending_prompt",
                     "starter_cache", "feedback_by_seq", "telemetry_events",
                     "_restore_dismissed", "_restore_snapshot",
                     "_checklist_dismissed"):
            if _key == "telemetry_events":
                # Telemetry buffer is per-session ephemeral; a new chat
                # starts a fresh pilot-evaluation window.
                st.session_state.pop(_key, None)
            else:
                st.session_state.pop(_key, None)
        # msg_seq stays monotonic so message IDs are never reused
        # (telemetry + feedback keys stay unique within the session).
        st.session_state["_clear_backup"] = True
        st.rerun()

# --- HEADER: quiet identity + live document status (native components) ---
st.title("RAG-4i")
st.caption("Ask questions about your connected documents.")
if kb_ready:
    _counts = []
    if kb.get("document_count") is not None:
        _counts.append(f"{kb['document_count']} documents")
    if kb.get("chunk_count") is not None:
        _counts.append(f"{kb['chunk_count']} indexed chunks")
    st.caption("Documents ready" + (f" · {' · '.join(_counts)}" if _counts else ""))
else:
    st.caption("Document workspace")

# --- MAIN CHAT INTERFACE ---
# Initialize chat history + bounded conversation memory (recent turns only).
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    from conversation_memory import ConversationMemory
    st.session_state.memory = ConversationMemory()
if "model_state" not in st.session_state:
    st.session_state.model_state = None
if "feedback_by_seq" not in st.session_state:
    st.session_state.feedback_by_seq = {}
if "telemetry_events" not in st.session_state:
    st.session_state.telemetry_events = []
if "msg_seq" not in st.session_state:
    st.session_state.msg_seq = 0

# --- MODEL GATE: real readiness, never simulated ---
if st.session_state.model_state is None:
    st.header(WELCOME_TITLE)
    st.write(WELCOME_BODY)
    st.caption("Loading the assistant model…")
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
    st.header(WELCOME_TITLE)
    st.write(detail)
    # Actionable recovery (what / what-to-do / retry), no stack trace.
    try:
        from readiness import recovery_message, summarize_readiness
        _gate_readiness = summarize_readiness(_model, _kb_summary)
        _gate_rec = recovery_message(_gate_readiness, _kb_summary, _model, is_cloud)
        st.caption(f"{_gate_rec['what']} {_gate_rec['next']}")
    except Exception:
        pass
    if st.button("Retry connection"):
        st.session_state.model_state = None
        st.toast("Retrying connection…")
        st.rerun()
    st.stop()

# Unified readiness for welcome / composer / recovery (no extra backend checks).
from readiness import composer_state, summarize_readiness
_readiness = summarize_readiness(st.session_state.model_state, _kb_summary)
_composer = composer_state(_readiness)

# Display previous chat messages (decorations re-render from stored
# metadata every run, so reruns never wipe labels/sources/copy buttons).
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"<div class='rag-msg rag-msg-{message['role']}'>",
                    unsafe_allow_html=True)
        if message.get("label"):
            st.caption(message["label"])
        if message.get("notice") == "no-context":
            st.warning(
                f"Retrieved **0 chunks** above the relevance "
                f"threshold ({cfg.relevance_threshold}) "
                f"(top-k={cfg.retrieval_k}). The answer below is "
                f"the no-context fallback, not a grounded answer."
            )
        st.markdown(message["content"])
        if message.get("strength"):
            st.caption(message["strength"])
        if message.get("guard_note"):
            st.caption(message["guard_note"])
        # User-visible latency (real elapsed, never "confidence").
        _lat = message.get("latency_ms", message.get("total_ms"))
        if message.get("role") == "assistant" and isinstance(_lat, (int, float)):
            try:
                from pilot_telemetry import format_answer_meta
                st.caption(format_answer_meta(
                    _lat, len(message.get("sources") or [])))
            except Exception:
                pass
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
                for _idx, _row in enumerate(_rows):
                    _ptitle = _row["file_name"] or "unknown"
                    if _row["page"] is not None:
                        _ptitle += f" p. {_row['page']}"
                    _ptitle += f" · relevance {_row['score_text']}"
                    with st.expander(_ptitle, expanded=False):
                        st.markdown(source_row_html(_row), unsafe_allow_html=True)
                        # Ask about this source: normal pipeline only.
                        _ask_q = ask_about_document_question(_row.get("file_name"))
                        _ask_key = (f"ask_{message.get('seq', 0)}_{_idx}_"
                                    f"{(_row.get('file_name') or 'x')[:20]}")
                        if st.button("Ask about this document", key=_ask_key,
                                     help=f"Ask a follow-up about {_row.get('file_name') or 'this document'}."):
                            st.session_state.pending_prompt = _ask_q
                            st.rerun()
            _components.html(
                copy_button_html(message["content"],
                                 f"copy_{message.get('seq', 0)}"),
                height=44,
            )
        # Lightweight answer feedback (assistant turns only; no LLM;
        # rerun-safe; never alters answer/source/label state).
        if message.get("role") == "assistant" and isinstance(message.get("seq"), int):
            _seq = message["seq"]
            _fb = (st.session_state.get("feedback_by_seq") or {}).get(str(_seq))
            _fb_val = (_fb or {}).get("value") if isinstance(_fb, dict) else None
            _col1, _col2, _col3 = st.columns([1, 1, 6])
            with _col1:
                if st.button("👍" if _fb_val != 1 else "👍 ✓", key=f"fb_up_{_seq}",
                             help="This answer was helpful."):
                    from pilot_telemetry import record_feedback_state
                    record_feedback_state(st.session_state, _seq, 1)
                    try:
                        from pilot_telemetry import PilotTelemetryStore
                        for _ev in reversed(st.session_state.get("telemetry_events", [])):
                            if _ev.get("message_id") == _seq:
                                _ev["feedback"] = 1
                                _ev["feedback_category"] = None
                                break
                    except Exception:
                        pass
                    st.rerun()
            with _col2:
                if st.button("👎" if _fb_val != -1 else "👎 ✓", key=f"fb_down_{_seq}",
                             help="This answer was not helpful."):
                    from pilot_telemetry import record_feedback_state
                    record_feedback_state(st.session_state, _seq, -1,
                                          (_fb or {}).get("category") if isinstance(_fb, dict) else None)
                    try:
                        for _ev in reversed(st.session_state.get("telemetry_events", [])):
                            if _ev.get("message_id") == _seq:
                                _ev["feedback"] = -1
                                break
                    except Exception:
                        pass
                    st.rerun()
            with _col3:
                if _fb_val == 1:
                    st.caption("Thanks for the feedback.")
                elif _fb_val == -1:
                    st.caption("Thanks — what went wrong?")
            if _fb_val == -1:
                try:
                    from pilot_telemetry import FEEDBACK_CATEGORIES
                    _cats = list(FEEDBACK_CATEGORIES)
                    _current_cat = (_fb or {}).get("category")
                    _idx = _cats.index(_current_cat) if _current_cat in _cats else None
                    _picked = st.selectbox(
                        "What went wrong?",
                        options=_cats,
                        index=_idx if _idx is not None else 0,
                        key=f"fb_cat_{_seq}",
                        help="Pick the closest reason (no extra questions asked).")
                    # Persist the selection (idempotent; selectbox reruns on change).
                    if _picked != _current_cat:
                        from pilot_telemetry import record_feedback_state
                        record_feedback_state(st.session_state, _seq, -1, _picked)
                        try:
                            for _ev in reversed(st.session_state.get("telemetry_events", [])):
                                if _ev.get("message_id") == _seq:
                                    _ev["feedback_category"] = _picked
                                    break
                        except Exception:
                            pass
                        st.rerun()
                except Exception:
                    logger.warning("Feedback category render failed (see logs).")
        st.markdown("</div>", unsafe_allow_html=True)


def _initial_suggestions():
    """Starter questions, probed against the index with per-session cache.

    Only questions that actually retrieve (>=1 hit through the real
    thresholded pipeline) are shown; the plain filename fallback fills
    any remaining slots, so output is never worse than before.
    """
    from suggestions import grounded_initial_suggestions, initial_suggestions
    try:
        from vector_store import get_vector_store
        from embeddings import get_embedding_provider
        from backend import retrieve_documents
        vs = get_vector_store(cfg, get_embedding_provider(cfg))
        files = [s["file_name"] for s in vs.list_sources()]
    except Exception:
        return initial_suggestions([])
    cache = st.session_state.get("starter_cache")
    if not isinstance(cache, dict):
        cache = {}
    key = tuple(files)
    if key not in cache:
        from backend import preview_answer

        def _probe(question):
            return preview_answer(question, vector_store=vs)["will_generate"]

        cache[key] = grounded_initial_suggestions(_probe, files)
        st.session_state.starter_cache = cache
    return cache[key]


def _handle_prompt(prompt_text):
    """Single conversation pipeline for typed and suggested questions.

    State-only: appends to session_state; the display loop above renders
    everything, so reruns never wipe labels/sources/copy buttons.
    """
    if "msg_seq" not in st.session_state:
        st.session_state.msg_seq = 0

    def _push(role, content, label=None, sources=None, notice=None,
              strength=None, guard_note=None, latency_ms=None,
              timings=None):
        st.session_state.msg_seq += 1
        st.session_state.messages.append({
            "role": role, "content": content, "seq": st.session_state.msg_seq,
            "label": label, "sources": sources or [], "notice": notice,
            "strength": strength, "guard_note": guard_note,
            "latency_ms": latency_ms,
            "total_ms": latency_ms,
            "timings": dict(timings or {}) if timings else None,
        })
        return st.session_state.msg_seq

    _t_total0 = time.monotonic()
    # 1. Record User Message
    _push("user", prompt_text)
    _count_before = len(st.session_state.messages)

    # 2. Generate Assistant Response (honest staged loading).
    # Preparation (routing/retrieval/assessment) runs inside the status
    # block; the streaming chat renders OUTSIDE it so status-nesting can
    # never swallow or misplace streamed output.
    _prep = {"ok": False}
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
            from ui.components import response_label
            memory = st.session_state.memory
            memory.add("user", prompt_text)
            history = memory.as_context()

            # Streaming answer: preparation (routing/retrieval/assessment)
            # runs once inside stream_answer; chunks render incrementally.
            from backend import EMPTY_RESPONSE_MESSAGE, stream_answer
            from output_safety import is_empty_response
            info, stream = stream_answer(
                prompt_text, conversation_context=history, on_phase=_phase)
            _prep.update(ok=True, info=info, stream=stream,
                         needs_retrieval=info["needs_retrieval"],
                         sources=info["retrieved"])
            status.update(label="Preparation done", state="complete")
        except Exception:
            # No raw stack trace for normal users; details go to console/log.
            logger.warning("UI query failed (see logs for details).")
            status.update(label="Something went wrong", state="error")
            st.session_state.last_failed = prompt_text
            st.error(friendly_error("An error occurred while answering. "
                                    "Make sure LM Studio Server is running!"))
            return

    if not _prep["ok"]:
        return
    info, stream = _prep["info"], _prep["stream"]
    needs_retrieval, sources = _prep["needs_retrieval"], info["retrieved"]
    _prep_timings = dict(info.get("timings") or {})
    _t_gen0 = time.monotonic()
    try:
        if info["answer"] is not None:
            # Decided without generation (routed/unsupported/fallback).
            response_text = info["answer"]
            _generation_ms = 0
        else:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                pieces = []
                try:
                    for piece in stream:
                        pieces.append(piece)
                        placeholder.markdown("".join(pieces) + "▍")
                except Exception:
                    # Discard the broken partial render; fall back to
                    # non-streaming generation with the SAME evidence.
                    logger.warning("Stream failed mid-answer; falling back.")
                    pieces = [generate_answer(
                        prompt_text, sources,
                        support_level=info["support_level"])]
                response_text = "".join(pieces)
                placeholder.empty()
            _generation_ms = max(0, int((time.monotonic() - _t_gen0) * 1000))
        if is_empty_response(response_text):
            response_text = EMPTY_RESPONSE_MESSAGE

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
    # The live stream above already showed this turn; the trailing
    # rerun re-renders everything from state (no duplication: each
    # run renders live output once, then state once).
        _label = response_label(prompt_text, sources, needs_retrieval)
        from answer_support import citation_guard
        from ui.components import retrieval_strength
        _guard = citation_guard(display_text, sources) \
            if sources else {"flagged": [], "count": 0, "total": 0}
        _guard_note = None
        if _guard["count"]:
            if _label == "Grounded answer" and _guard["count"] > _guard["total"] / 2:
                _label = "Partial answer"
            _guard_note = (f"({_guard['count']} of {_guard['total']} "
                           f"statements unverified against sources)")
        from pilot_telemetry import top_retrieval_score
        _total_ms = max(0, int((time.monotonic() - _t_total0) * 1000))
        _timings = dict(_prep_timings)
        _timings["generation_ms"] = _generation_ms
        _timings["total_ms"] = _total_ms
        _seq = _push("assistant", full_response, label=_label,
                     sources=sources, notice=notice,
                     strength=retrieval_strength(sources),
                     guard_note=_guard_note,
                     latency_ms=_total_ms, timings=_timings)
        st.session_state.memory.add("assistant", full_response)
        st.session_state.pop("last_failed", None)

        # Pilot telemetry: metadata only (never question/answer text).
        try:
            from pilot_telemetry import build_telemetry_event
            _strength_val = top_retrieval_score(sources)
            _strength_txt = retrieval_strength(sources)
            _ev = build_telemetry_event(
                message_id=_seq, answer_label=_label,
                source_count=len(sources or []),
                retrieval_strength=_strength_val if _strength_val is not None else _strength_txt,
                total_ms=_total_ms, timings=_timings)
            st.session_state.telemetry_events.append(_ev)
            # Structured log (counts/timings only; no content/secrets).
            logger.info("answer telemetry msg=%s label=%s sources=%d "
                        "retrieval_ms=%s support_ms=%s generation_ms=%s total_ms=%d",
                        _seq, _label, len(sources or []),
                        _timings.get("retrieval_ms"), _timings.get("support_ms"),
                        _generation_ms, _total_ms)
        except Exception:
            logger.warning("Telemetry record failed (see logs).")

        # 4. Persist for top-level suggestion rendering (buttons must
        # exist on every rerun, not only inside prompt handling).
        from suggestions import remember_followups
        remember_followups(st.session_state, prompt_text, sources)
    except Exception:
        # No raw stack trace for normal users; details go to console/log.
        logger.warning("UI query failed (see logs for details).")
        st.session_state.last_failed = prompt_text
        st.error(friendly_error("An error occurred while answering. "
                                "Make sure LM Studio Server is running!"))

    # Refresh only when new messages exist (errors stay visible as-is).
    if len(st.session_state.messages) > _count_before:
        st.rerun()


# Retry card for the last failed question (same pipeline, no new logic).
if st.session_state.get("last_failed") and not st.session_state.get("pending_prompt"):
    st.caption("The last answer failed.")
    # Actionable hint: retry is appropriate here (transient model/retrieval fault).
    st.caption("Retry is safe to try — your documents and chat are unchanged.")
    if st.button("Retry answer", key="retry_last"):
        st.session_state.pending_prompt = st.session_state.pop("last_failed")
        st.rerun()


# Follow-up buttons for the latest grounded answer. Rendered at top level
# on EVERY rerun so clicks always materialize (this was the bug: buttons
# previously existed only inside prompt handling, so clicks were lost).
# Stacked full-width: readable on every screen size without media queries.
from suggestions import current_followups as _current_followups
_followup_specs = _current_followups(st.session_state)
if _followup_specs:
    st.caption("You may also ask")
    for _spec in _followup_specs:
        if st.button(_spec["label"], key=_spec["key"], use_container_width=True):
            if "pending_prompt" not in st.session_state:
                st.session_state.pending_prompt = _spec["label"]
            st.rerun()

# Welcome hero + first-run experience for a fresh conversation.
if not st.session_state.messages:
    st.header(WELCOME_TITLE)
    st.write(WELCOME_BODY)

    # Guided setup checklist (real readiness only; dismissible; hides when READY).
    try:
        from readiness import checklist_items
        if _readiness.get("state") != "READY" and not st.session_state.get("_checklist_dismissed"):
            _items = checklist_items(_readiness, _kb_summary, is_cloud)
            with st.expander("Get started", expanded=True):
                for _it in _items:
                    _icon = "✓" if _it["done"] else "○"
                    st.write(f"{_icon} {_it['label']}")
                    if _it.get("hint"):
                        st.caption(_it["hint"])
                if st.button("Dismiss", key="checklist_dismiss"):
                    st.session_state["_checklist_dismissed"] = True
                    st.rerun()
    except Exception:
        logger.warning("First-run checklist unavailable (see logs).")

    try:
        _kb_is_ready = bool(_kb_summary.get("ready"))
        if _kb_is_ready:
            # Document-aware welcome (existing list_sources only; compact).
            try:
                from vector_store import get_vector_store as _gvs
                from embeddings import get_embedding_provider as _gep
                _wvs = _gvs(cfg, _gep(cfg))
                _wsrcs = _wvs.list_sources()
                _wline = document_welcome_summary(_wsrcs)
                if _wline:
                    st.caption(_wline)
            except Exception:
                logger.warning("Document welcome unavailable (see logs).")
            # Capability discovery reuses the grounded starter machinery
            # (no second engine; only validated, supported capabilities).
            st.caption("Try asking")
            starters = _initial_suggestions()
            for i, sug in enumerate(starters):
                if st.button(sug, key=f"starter_{i}", use_container_width=True):
                    st.session_state.pending_prompt = sug
                    st.rerun()
        else:
            # Intelligent empty state: 3 steps adapted to local/S3 mode.
            from readiness import empty_state_steps
            for _step in empty_state_steps(is_cloud, cfg.s3_bucket, cfg.s3_prefix):
                st.write(_step)
            st.info("Add documents using the Knowledge base panel, "
                    "then come back and ask away.")
    except Exception:
        logger.warning("Starter suggestions unavailable (see logs).")

    # Lightweight browser-local restore (accidental-refresh recovery only).
    # Pure serialization is fully tested; the browser half is a one-way
    # localStorage saver (no fragile JS→Python bridge). Restore is an
    # explicit paste + choice so no legal text ever flows silently.
    if not st.session_state.get("_restore_dismissed"):
        with st.expander("Recover previous session?", expanded=False):
            st.caption("If the page reloaded accidentally, paste a previously "
                       "copied session backup to restore this conversation only. "
                       "Backups stay in your browser (localStorage); nothing is "
                       "stored on the server.")
            _paste = st.text_area("Session backup (JSON)", key="restore_paste",
                                  placeholder='Paste {"schema": 1, ...} here',
                                  help="Paste a backup copied from this browser.")
            _c1, _c2 = st.columns(2)
            with _c1:
                if st.button("Restore", key="restore_confirm"):
                    try:
                        import json as _json
                        from session_restore import deserialize_conversation, restore_messages_into_state
                        _raw = _json.loads(_paste or "{}")
                        _snap = deserialize_conversation(_raw)
                        if _snap.get("ok") and _snap.get("messages"):
                            _n = restore_messages_into_state(st.session_state, _snap)
                            # Rebuild memory anchors from restored user turns
                            # (assistant text never becomes retrieval evidence).
                            try:
                                from conversation_memory import ConversationMemory
                                _mem = ConversationMemory()
                                for _m in st.session_state.messages:
                                    if _m.get("role") in ("user", "assistant"):
                                        _mem.add(_m["role"], _m.get("content", ""))
                                st.session_state.memory = _mem
                            except Exception:
                                pass
                            st.session_state["_restore_dismissed"] = True
                            st.toast(f"Restored {_n} messages.")
                            st.rerun()
                        else:
                            st.warning("That backup could not be read. Check the pasted JSON.")
                    except Exception:
                        logger.warning("Session restore failed (see logs).")
                        st.warning("That backup could not be read. Check the pasted JSON.")
            with _c2:
                if st.button("Start fresh", key="restore_fresh"):
                    st.session_state["_restore_dismissed"] = True
                    st.session_state["_clear_backup"] = True
                    st.rerun()
            if _paste:
                try:
                    import json as _json
                    from session_restore import deserialize_conversation
                    _prev = deserialize_conversation(_json.loads(_paste))
                    if _prev.get("saved_at"):
                        st.caption(f"Backup saved at {_prev['saved_at']} · "
                                   f"{len(_prev.get('messages', []))} messages")
                except Exception:
                    pass

# Browser-local snapshot saver (one-way; runs whenever there is a conversation).
try:
    import streamlit.components.v1 as _components2
    import json as _json2
    from session_restore import serialize_conversation as _ser
    if st.session_state.get("messages"):
        _snap2 = _ser(st.session_state.messages)
        _components2.html(localstorage_saver_html(_json2.dumps(_snap2)), height=0)
    if st.session_state.pop("_clear_backup", False):
        _components2.html(localstorage_clearer_html(), height=0)
except Exception:
    pass

# Handle User Input (typed or clicked suggestion, one pipeline).
pending = st.session_state.pop("pending_prompt", None)
_composer_disabled = bool(_composer.get("disabled"))
if _composer_disabled:
    st.caption(_composer.get("reason") or "Chat is disabled until ready.")
typed = st.chat_input(
    "Ex: What is the lock-in period in the lease deed?",
    disabled=_composer_disabled,
)
if pending or typed:
    _handle_prompt(pending or typed)
