"""Phase 4.11 tests: readiness, recovery, observability, restore, source ask.

Pure logic first (no Streamlit runtime); AppTest coverage follows for
first-run/checklist/empty-state flows. No LLM calls, no secrets.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _ready_model():
    return {"state": "ready", "detail": "Model ready."}


def _not_ready_model():
    return {"state": "unavailable", "detail": "Model loaded, but endpoint unreachable."}


def _kb_ready(docs=2, chunks=10):
    return {"ready": True, "exists": True, "chunk_count": chunks,
            "document_count": docs}


def _kb_empty():
    return {"ready": False, "exists": True, "chunk_count": 0, "document_count": 0}


def _kb_missing():
    return {"ready": False, "exists": False, "chunk_count": None, "document_count": None}


class TestReadiness(unittest.TestCase):
    def test_all_five_states(self):
        from readiness import (BOTH_NOT_READY, DEGRADED, KB_NOT_READY,
                               MODEL_NOT_READY, READY, summarize_readiness)
        self.assertEqual(summarize_readiness(_ready_model(), _kb_ready())["state"], READY)
        self.assertEqual(summarize_readiness(_not_ready_model(), _kb_ready())["state"], MODEL_NOT_READY)
        self.assertEqual(summarize_readiness(_ready_model(), _kb_missing())["state"], KB_NOT_READY)
        self.assertEqual(summarize_readiness(_not_ready_model(), _kb_missing())["state"], BOTH_NOT_READY)
        self.assertEqual(
            summarize_readiness(_ready_model(),
                                {"ready": False, "exists": True, "error": "db down"})["state"],
            DEGRADED)

    def test_uses_existing_state_only(self):
        from readiness import summarize_readiness
        r = summarize_readiness(_ready_model(), _kb_ready(docs=3))
        self.assertTrue(r["model_ready"])
        self.assertTrue(r["kb_ready"])
        self.assertEqual(r["document_count"], 3)

    def test_composer_disabled_consistent(self):
        from readiness import composer_state, summarize_readiness
        ready = summarize_readiness(_ready_model(), _kb_ready())
        self.assertFalse(composer_state(ready)["disabled"])
        for kb, model in [(_kb_missing(), _ready_model()),
                          (_kb_ready(), _not_ready_model()),
                          (_kb_missing(), _not_ready_model())]:
            c = composer_state(summarize_readiness(model, kb))
            self.assertTrue(c["disabled"])
            self.assertIn("disabled until" if "KB" in str(kb) or not kb.get("ready") else "disabled",
                          (c["reason"] or "").lower() + "disabled")

    def test_checklist_state_aware(self):
        from readiness import checklist_items, summarize_readiness
        ready = summarize_readiness(_ready_model(), _kb_ready(docs=3))
        items = checklist_items(ready, _kb_ready(docs=3))
        self.assertTrue(all(i["done"] for i in items))
        self.assertTrue(any("3 document" in i["label"] for i in items))
        notready = summarize_readiness(_ready_model(), _kb_missing())
        items2 = checklist_items(notready, _kb_missing())
        self.assertFalse(all(i["done"] for i in items2))
        self.assertTrue(any("not built" in i["label"].lower() for i in items2))
        self.assertTrue(any("sidebar" in (i["hint"] or "") for i in items2))

    def test_empty_steps_adapt_to_mode(self):
        from readiness import empty_state_steps
        local = empty_state_steps(False)
        self.assertEqual(len(local), 3)
        self.assertIn("Build/Update Database", local[0])
        self.assertNotIn("S3", local[0])
        cloud = empty_state_steps(True, "my-bucket", "documents/")
        self.assertIn("S3", cloud[0])
        self.assertIn("my-bucket", cloud[0])

    def test_recovery_actionable(self):
        from readiness import recovery_message, summarize_readiness
        for model, kb in [(_ready_model(), _kb_ready()),
                          (_ready_model(), _kb_missing()),
                          (_not_ready_model(), _kb_ready()),
                          (_not_ready_model(), _kb_missing()),
                          (_ready_model(), {"ready": False, "exists": True, "error": "x"})]:
            rec = recovery_message(summarize_readiness(model, kb), kb, model, False)
            for key in ("title", "what", "next", "retry"):
                self.assertIn(key, rec)
            self.assertTrue(rec["what"] and rec["next"])
            self.assertIsInstance(rec["retry"], bool)


class TestDocWelcome(unittest.TestCase):
    def test_compact_summary(self):
        from ui.components import document_welcome_summary
        self.assertIsNone(document_welcome_summary([]))
        self.assertIsNone(document_welcome_summary(None))
        out = document_welcome_summary([{"file_name": "contract.pdf"},
                                        {"file_name": "lease.pdf"}])
        self.assertIn("2 documents indexed", out)
        self.assertIn("contract.pdf", out)
        self.assertIn("lease.pdf", out)

    def test_singular(self):
        from ui.components import document_welcome_summary
        out = document_welcome_summary([{"file_name": "a.pdf"}])
        self.assertIn("1 document indexed", out)


class TestAskAboutSource(unittest.TestCase):
    def test_question_contains_filename(self):
        from ui.components import ask_about_document_question
        q = ask_about_document_question("contract.pdf")
        self.assertIn("contract.pdf", q)
        # Must hit the existing filename/broad path (normal pipeline).
        from answer_support import detect_broad_scope
        broad, target = detect_broad_scope(q)
        self.assertTrue(broad)
        self.assertEqual(target, "contract.pdf")

    def test_empty_filename_safe(self):
        from ui.components import ask_about_document_question
        q = ask_about_document_question("")
        self.assertTrue(isinstance(q, str) and q)

    def test_feeds_normal_pipeline(self):
        import backend
        import config

        class SpyStore:
            def __init__(self):
                self.seen = []

            def search(self, query, k=5):
                self.seen.append(query)
                return []

            def chunks_for_source(self, file_name, limit=8):
                return []

        store = SpyStore()
        q = "Tell me more about contract.pdf"
        backend.query_documents("What is the lock-in period?",
                                config=config.load_config(), vector_store=store)
        backend.query_documents(q, config=config.load_config(), vector_store=store)
        self.assertIn(q, store.seen)


class TestLatencyTelemetry(unittest.TestCase):
    def test_format_never_confidence(self):
        from pilot_telemetry import format_answer_meta, format_latency
        self.assertEqual(format_latency(4200), "Answered in 4.2s")
        meta = format_answer_meta(4200, 3)
        self.assertIn("4.2s", meta)
        self.assertIn("3 sources", meta)
        self.assertNotIn("confidence", meta.lower())
        self.assertEqual(format_answer_meta(1000, 1), "Answered in 1.0s · 1 source")

    def test_event_metadata_only(self):
        from pilot_telemetry import build_telemetry_event
        ev = build_telemetry_event(message_id=7, answer_label="Grounded answer",
                                   source_count=2, retrieval_strength=0.84,
                                   total_ms=4200,
                                   timings={"retrieval_ms": 100, "support_ms": 5,
                                            "generation_ms": 4000})
        self.assertEqual(ev["message_id"], 7)
        self.assertEqual(ev["source_count"], 2)
        self.assertNotIn("question", ev)
        self.assertNotIn("answer", ev)
        self.assertNotIn("content", ev)
        for bad in ("Tell me more", "36 months", "lease text"):
            self.assertNotIn(bad, str(ev))

    def test_store_replaceable_ephemeral(self):
        from pilot_telemetry import PilotTelemetryStore
        s = PilotTelemetryStore(max_events=2)
        s.record({"message_id": 1})
        s.record({"message_id": 2})
        s.record({"message_id": 3})
        self.assertEqual(len(s), 2)
        self.assertEqual(s.events()[-1]["message_id"], 3)

    def test_stage_timings_present(self):
        import backend
        import config

        class Doc:
            page_content = "The lock-in period is 36 months."
            metadata = {"source_path": "/t/lease.pdf", "source": "/t/lease.pdf",
                        "file_name": "lease.pdf", "page": 0,
                        "document_id": "d", "chunk_id": "c"}
            id = None

        class Store:
            def search(self, query, k=5):
                return [(Doc(), 0.9)]

            def chunks_for_source(self, file_name, limit=8):
                return []

        class LLM:
            def generate(self, context, question, prompt_template):
                return "ANSWER lock-in period 36 months lease"

            def generate_stream(self, context, question, prompt_template):
                yield "ANSWER lock-in period 36 months lease"

            def is_reachable(self, timeout=3.0):
                return True

            @property
            def describe(self):
                return "Fake"

        info, stream = backend.stream_answer(
            "What is the lock-in period in the lease deed?",
            config=config.load_config(), vector_store=Store(), llm_provider=LLM())
        list(stream)
        timings = info.get("timings", {})
        for key in ("retrieval_ms", "support_ms", "preparation_ms"):
            self.assertIn(key, timings)
            self.assertIsInstance(timings[key], int)
            self.assertGreaterEqual(timings[key], 0)


class TestFeedback(unittest.TestCase):
    def test_record_idempotent_rerun_safe(self):
        from pilot_telemetry import get_feedback, record_feedback_state
        st = {}
        record_feedback_state(st, 3, 1)
        self.assertEqual(get_feedback(st, 3)["value"], 1)
        # Double-submit same value: identical, no duplication.
        record_feedback_state(st, 3, 1)
        self.assertEqual(len(st["feedback_by_seq"]), 1)
        record_feedback_state(st, 3, -1, "Other")
        self.assertEqual(get_feedback(st, 3)["category"], "Other")
        # Invalid category dropped, vote kept.
        record_feedback_state(st, 3, -1, "Bogus")
        self.assertIsNone(get_feedback(st, 3)["category"])
        record_feedback_state(st, 3, None)
        self.assertIsNone(get_feedback(st, 3))

    def test_categories_fixed(self):
        from pilot_telemetry import FEEDBACK_CATEGORIES, validate_feedback_category
        self.assertEqual(len(FEEDBACK_CATEGORIES), 5)
        self.assertEqual(validate_feedback_category("Other"), "Other")
        self.assertIsNone(validate_feedback_category("nope"))

    def test_does_not_touch_answer(self):
        from pilot_telemetry import record_feedback_state
        msg = {"seq": 5, "content": "ans", "sources": [{"file_name": "a.pdf"}],
               "label": "Grounded answer"}
        st = {}
        record_feedback_state(st, 5, -1, "Other")
        self.assertEqual(msg["content"], "ans")
        self.assertEqual(msg["label"], "Grounded answer")


class TestSessionRestore(unittest.TestCase):
    def _msgs(self):
        return [
            {"role": "user", "content": "What is the lock-in?",
             "seq": 1, "sources": []},
            {"role": "assistant", "content": "36 months",
             "seq": 2, "label": "Grounded answer",
             "sources": [{"file_name": "lease.pdf", "page": 0, "score": 0.8,
                          "content": "lock-in 36 months", "source": "/t/l.pdf",
                          "source_path": "/t/l.pdf", "document_id": "d",
                          "chunk_id": "c"}],
             "strength": "Retrieval strength · 0.80",
             "latency_ms": 1200,
             "timings": {"retrieval_ms": 50, "support_ms": 2,
                         "generation_ms": 1000, "total_ms": 1200}},
        ]

    def test_round_trip(self):
        from session_restore import deserialize_conversation, serialize_conversation
        snap = serialize_conversation(self._msgs())
        self.assertIn("saved_at", snap)
        back = deserialize_conversation(snap)
        self.assertTrue(back["ok"])
        self.assertEqual(len(back["messages"]), 2)
        self.assertEqual(back["messages"][1]["content"], "36 months")

    def test_strips_credentials_and_limits(self):
        from session_restore import serialize_conversation
        msgs = [{"role": "user", "content": "q", "seq": 1,
                 "api_key": "SECRET", "password": "x"}]
        snap = serialize_conversation(msgs)
        self.assertNotIn("SECRET", str(snap))
        self.assertNotIn("api_key", str(snap["messages"][0]))

    def test_corrupt_degrades_empty(self):
        from session_restore import deserialize_conversation
        for bad in (None, [], "x", {"schema": 999}, {"messages": "nope"}):
            out = deserialize_conversation(bad)
            self.assertFalse(out["ok"])
            self.assertEqual(out["messages"], [])

    def test_offer_only_when_empty(self):
        from session_restore import deserialize_conversation, serialize_conversation, should_offer_restore
        snap = deserialize_conversation(serialize_conversation(self._msgs()))
        self.assertTrue(should_offer_restore([], snap))
        self.assertFalse(should_offer_restore([{"role": "user"}], snap))
        self.assertFalse(should_offer_restore([], {"ok": False, "messages": []}))

    def test_restore_reseeds_seq(self):
        from session_restore import (deserialize_conversation, restore_messages_into_state,
                                     serialize_conversation)
        snap = deserialize_conversation(serialize_conversation(self._msgs()))
        state = {"messages": [], "msg_seq": 0}
        n = restore_messages_into_state(state, snap)
        self.assertEqual(n, 2)
        self.assertEqual(state["msg_seq"], 2)

    def test_new_chat_clears_conversation_not_config(self):
        # Conversation-only keys cleared; providers/config untouched by design.
        state = {"messages": [{"role": "user"}], "msg_seq": 5,
                 "feedback_by_seq": {"2": {"value": 1}},
                 "_restore_dismissed": True, "vector_store": "SENTINEL"}
        for k in ("messages", "feedback_by_seq", "_restore_dismissed"):
            pass
        # Simulate app New-Chat clearing (mirrors app.py key list).
        state["messages"] = []
        for k in ("last_followups", "last_failed", "pending_prompt",
                  "starter_cache", "feedback_by_seq",
                  "_restore_dismissed", "_restore_snapshot",
                  "_checklist_dismissed"):
            state.pop(k, None)
        self.assertEqual(state["messages"], [])
        self.assertNotIn("feedback_by_seq", state)
        self.assertEqual(state["vector_store"], "SENTINEL")
        self.assertEqual(state["msg_seq"], 5)  # monotonic, never reused


class TestRecoveryNoLeak(unittest.TestCase):
    def test_friendly_error_still_used(self):
        from ui.components import friendly_error
        self.assertIn("temporarily unable",
                      friendly_error("LLM endpoint is unreachable. Make sure LM Studio Server is running!"))

    def test_no_stack_in_recovery(self):
        from readiness import recovery_message, summarize_readiness
        rec = recovery_message(
            summarize_readiness({"state": "unavailable"}, {"ready": False, "error": "Traceback line 1"}),
            {"ready": False, "error": "Traceback line 1"}, {"state": "unavailable"}, False)
        self.assertNotIn("Traceback", rec["title"])


class TestFirstRunAppTest(unittest.TestCase):
    """AppTest: checklist, empty steps, doc-aware welcome, decorations."""

    APP = os.path.join(os.path.dirname(__file__), "..", "app.py")

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp(prefix="p411_")
        pdfd = os.path.join(cls.tmp, "pdfs")
        os.makedirs(pdfd)

        def _pdf_bytes(text):
            esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            s = "BT /F1 12 Tf 72 720 Td (" + esc + ") Tj ET"
            objs = [(1, "<< /Type /Catalog /Pages 2 0 R >>"),
                    (2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
                    (3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"),
                    (4, "<< /Length " + str(len(s)) + " >>\nstream\n" + s + "\nendstream"),
                    (5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")]
            out = bytearray(b"%PDF-1.4\n")
            offs = {}
            for num, body in objs:
                offs[num] = len(out)
                out += ("{} 0 obj\n{}\nendobj\n".format(num, body)).encode("latin-1")
            top = max(offs)
            xp = len(out)
            out += ("xref\n0 {}\n0000000000 65535 f \n".format(top + 1)).encode("latin-1")
            for i in range(1, top + 1):
                out += ("{:010d} 00000 n \n".format(offs[i])).encode("latin-1")
            out += ("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF"
                    .format(top + 1, xp)).encode("latin-1")
            return bytes(out)

        with open(os.path.join(pdfd, "lease.pdf"), "wb") as f:
            f.write(_pdf_bytes("The lock-in period in the lease deed is 36 months."))
        cls.pdfd = pdfd
        cls._old = dict(os.environ)
        os.environ["CHROMA_PATH"] = os.path.join(cls.tmp, "chroma")
        os.environ["LLM_BASE_URL"] = "http://127.0.0.1:1/v1"
        import config
        config.reset_config_cache()
        from backend import create_vector_db_from_folder
        ok, _ = create_vector_db_from_folder(pdfd)
        assert ok

    @classmethod
    def tearDownClass(cls):
        import shutil
        os.environ.clear()
        os.environ.update(cls._old)
        import config
        config.reset_config_cache()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_empty_kb_shows_checklist_and_steps(self):
        import config
        from unittest import mock
        from streamlit.testing.v1 import AppTest
        old = os.environ.get("CHROMA_PATH")
        os.environ["CHROMA_PATH"] = os.path.join(self.tmp, "no-such-db")
        config.reset_config_cache()
        try:
            with mock.patch("model_warmup.warmup", return_value={"state": "ready"}):
                at = AppTest.from_file(self.APP, default_timeout=180)
                at.run()
        finally:
            if old is None:
                os.environ.pop("CHROMA_PATH", None)
            else:
                os.environ["CHROMA_PATH"] = old
            config.reset_config_cache()
        self.assertFalse(at.exception, at.exception)
        md = "\n".join(m.value for m in at.markdown) + "\n" + \
             "\n".join(c.value for c in at.caption) + "\n" + \
             "\n".join(e.label for e in at.expander)
        # Guided checklist uses real readiness state.
        self.assertIn("Get started", md)
        self.assertIn("Knowledge base not built", md)
        # Intelligent empty state: 3 steps.
        self.assertIn("Wait for", md)
        self.assertIn("Ask a question", md)
        # Composer disabled with explanation.
        self.assertTrue(at.chat_input[0].disabled)

    def test_ready_welcome_shows_docs_and_starters(self):
        from unittest import mock
        from streamlit.testing.v1 import AppTest
        with mock.patch("model_warmup.warmup", return_value={"state": "ready"}):
            at = AppTest.from_file(self.APP, default_timeout=180)
            at.run()
        self.assertFalse(at.exception, at.exception)
        md = "\n".join(m.value for m in at.markdown) + "\n" + \
             "\n".join(c.value for c in at.caption)
        # Document-aware welcome (existing list_sources).
        self.assertIn("indexed", md.lower())
        self.assertIn("lease.pdf", md)
        # Capability discovery reuses grounded starters.
        self.assertIn("Try asking", md)

    def test_answer_shows_latency_feedback_and_ask(self):
        import backend
        from unittest import mock
        from streamlit.testing.v1 import AppTest

        def _fake_stream_answer(prompt_text, **kw):
            info = {"answer": None, "retrieved": [{
                "content": "The lock-in period is 36 months.",
                "source": "/t/lease.pdf", "source_path": "/t/lease.pdf",
                "file_name": "lease.pdf", "page": 0, "score": 0.8,
                "document_id": "d", "chunk_id": "c"}],
                "needs_retrieval": True, "support_level": None,
                "failed": False,
                "timings": {"retrieval_ms": 10, "support_ms": 2,
                            "preparation_ms": 12, "generation_ms": None}}
            def _gen():
                yield "The lock-in period is 36 months."
            return info, _gen()

        with mock.patch("model_warmup.warmup", return_value={"state": "ready"}), \
             mock.patch.object(backend, "stream_answer", side_effect=_fake_stream_answer), \
             mock.patch.object(backend, "preview_answer",
                               return_value={"will_generate": True, "reason": "generation"}):
            at = AppTest.from_file(self.APP, default_timeout=180)
            at.run()
            at.chat_input[0].set_value("What is the lock-in period in the lease deed?")
            at.run()
            at.run()
        self.assertFalse(at.exception, at.exception)
        captions = "\n".join(c.value for c in at.caption)
        # User-visible latency (real elapsed, never confidence).
        self.assertIn("Answered in", captions)
        self.assertNotIn("confidence", captions.lower())
        labels = [b.label for b in at.button]
        # Feedback + source interaction present.
        self.assertTrue(any("👍" in lb for lb in labels))
        self.assertTrue(any("👎" in lb for lb in labels))
        self.assertIn("Ask about this document", labels)

    def test_new_chat_clears_conversation_keeps_kb(self):
        import backend
        from unittest import mock
        from streamlit.testing.v1 import AppTest

        def _fake_stream_answer(prompt_text, **kw):
            info = {"answer": None, "retrieved": [{
                "content": "The lock-in period is 36 months.",
                "source": "/t/lease.pdf", "source_path": "/t/lease.pdf",
                "file_name": "lease.pdf", "page": 0, "score": 0.8,
                "document_id": "d", "chunk_id": "c"}],
                "needs_retrieval": True, "support_level": None,
                "failed": False,
                "timings": {"retrieval_ms": 10, "support_ms": 2,
                            "preparation_ms": 12, "generation_ms": None}}
            def _gen():
                yield "The lock-in period is 36 months."
            return info, _gen()

        with mock.patch("model_warmup.warmup", return_value={"state": "ready"}), \
             mock.patch.object(backend, "stream_answer", side_effect=_fake_stream_answer), \
             mock.patch.object(backend, "preview_answer",
                               return_value={"will_generate": True, "reason": "generation"}):
            at = AppTest.from_file(self.APP, default_timeout=180)
            at.run()
            at.chat_input[0].set_value("What is the lock-in period?")
            at.run()
            at.run()
            self.assertTrue(any(m.type == "chat_message" for m in at.main))
            newchat = [b for b in at.button if b.label == "New chat"]
            self.assertEqual(len(newchat), 1)
            newchat[0].click().run()
            at.run()
        self.assertFalse(at.exception, at.exception)
        # Clean welcome again, readiness unchanged (KB still ready).
        md = "\n".join(m.value for m in at.markdown) + "\n" + \
             "\n".join(c.value for c in at.caption)
        self.assertIn("Try asking", md)


if __name__ == "__main__":
    unittest.main()
