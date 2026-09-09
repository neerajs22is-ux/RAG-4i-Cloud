"""Headless UI flow tests (AppTest): model gate, clicks, notices, export."""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


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


class UIFlowCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="uiflow_")
        pdfd = os.path.join(cls.tmp, "pdfs")
        os.makedirs(pdfd)
        with open(os.path.join(pdfd, "lease.pdf"), "wb") as f:
            f.write(_pdf_bytes(
                "The lock-in period in the lease deed is 36 months."))
        cls._old = dict(os.environ)
        os.environ["CHROMA_PATH"] = os.path.join(cls.tmp, "chroma")
        os.environ["LLM_BASE_URL"] = "http://127.0.0.1:1/v1"  # dead: no LLM
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

    def _run(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(APP, default_timeout=180)
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    def test_model_gate_retry_flow(self):
        import model_warmup
        at = self._run()  # real warmup: embeddings ok, LLM dead
        md = "\n".join(m.value for m in at.markdown)
        self.assertIn("unreachable", md.lower())
        retry = [b for b in at.button if b.label == "Retry connection"]
        self.assertEqual(len(retry), 1)
        with mock.patch.object(model_warmup, "warmup",
                               return_value={"state": "ready"}) as w:
            retry[0].click().run()
            self.assertTrue(w.called)
        self.assertFalse(at.exception, at.exception)

    def test_no_context_notice_renders(self):
        from streamlit.testing.v1 import AppTest
        with mock.patch("model_warmup.warmup",
                        return_value={"state": "ready"}):
            at = AppTest.from_file(APP, default_timeout=180)
            at.run()
            at.chat_input[0].set_value("What is the zebra's nickname?")
            at.run()
        self.assertFalse(at.exception, at.exception)
        warnings = "\n".join(w.value for w in at.warning)
        self.assertIn("0 chunks", warnings)

    def test_suggestion_click_materializes(self):
        from streamlit.testing.v1 import AppTest
        with mock.patch("model_warmup.warmup",
                        return_value={"state": "ready"}):
            at = AppTest.from_file(APP, default_timeout=180)
            at.run()
            starters = [b for b in at.button
                        if b.label not in ("Build/Update Database",
                                           "Build/Update Database from S3",
                                           "Retry connection", "Retry answer",
                                           "New chat", "Remove",
                                           "Restore", "Start fresh", "Dismiss",
                                           "Ask about this document",
                                           "👍", "👎", "👍 ✓", "👎 ✓")]
            self.assertEqual(len(starters), 3)
            starters[0].click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(any(m.type == "chat_message" for m in at.main))

    def test_starter_click_produces_answer_not_refusal(self):
        import backend
        from streamlit.testing.v1 import AppTest

        def _fake_stream_answer(prompt_text, **kw):
            info = {"answer": None, "retrieved": [{
                "content": "The lock-in period is 36 months.",
                "source": "/t/lease.pdf", "source_path": "/t/lease.pdf",
                "file_name": "lease.pdf", "page": 0, "score": 0.8,
                "document_id": "d", "chunk_id": "c"}],
                "needs_retrieval": True, "support_level": None,
                "failed": False}

            def _gen():
                yield "The lock-in period is 36 months."

            return info, _gen()

        with mock.patch("model_warmup.warmup",
                        return_value={"state": "ready"}), \
             mock.patch.object(backend, "stream_answer",
                               side_effect=_fake_stream_answer), \
             mock.patch.object(backend, "preview_answer",
                               return_value={"will_generate": True,
                                             "reason": "generation"}):
            at = AppTest.from_file(APP, default_timeout=180)
            at.run()
            starters = [b for b in at.button
                        if b.label not in ("Build/Update Database",
                                           "Build/Update Database from S3",
                                           "Retry connection", "Retry answer",
                                           "New chat", "Remove",
                                           "Restore", "Start fresh", "Dismiss",
                                           "Ask about this document",
                                           "👍", "👎", "👍 ✓", "👎 ✓")]
            self.assertEqual(len(starters), 3)
            starters[0].click().run()
            # Click runs can take an extra rerun to settle.
            at.run()
        self.assertFalse(at.exception, at.exception)
        texts = [m.value for m in at.markdown]
        self.assertTrue(any("36 months" in t for t in texts), texts)
        joined = "\n".join(texts)
        self.assertNotIn("could not find", joined.lower())
        self.assertNotIn("unavailable", joined.lower())

    def test_disabled_composer_explains_why(self):
        import config
        from streamlit.testing.v1 import AppTest
        old = os.environ.get("CHROMA_PATH")
        os.environ["CHROMA_PATH"] = os.path.join(self.tmp, "no-such-db")
        config.reset_config_cache()
        try:
            with mock.patch("model_warmup.warmup",
                            return_value={"state": "ready"}):
                at = AppTest.from_file(APP, default_timeout=180)
                at.run()
        finally:
            if old is None:
                os.environ.pop("CHROMA_PATH", None)
            else:
                os.environ["CHROMA_PATH"] = old
            config.reset_config_cache()
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(at.chat_input[0].disabled)
        captions = "\n".join(c.value for c in at.caption)
        self.assertIn("disabled until", captions)


if __name__ == "__main__":
    unittest.main()
