"""Ingestion reporting + error handling (mocked, no real PDFs/models)."""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})


class TestIngestionReporting(unittest.TestCase):
    def test_partial_failure_message_lists_files(self):
        import backend
        import config
        cfg = config.load_config()
        with tempfile.TemporaryDirectory() as tmp:
            # Fake loader: 2 found, 1 ok, 1 fail
            fake_docs = [FakeDoc("text", {"source_path": tmp + "/ok.pdf"})]
            fake_report = {"found": 2, "succeeded": 1, "failed": 1,
                           "failed_files": ["bad.pdf"]}

            class FakeStore:
                def build_index(self, chunks):
                    return len(chunks)

            with mock.patch("document_loader.load_documents_from_folder",
                            return_value=(fake_docs, fake_report)), \
                 mock.patch("chunking.chunk_documents",
                            return_value=[FakeDoc("c")]):
                ok, msg = backend.create_vector_db_from_folder(
                    tmp, config=cfg, vector_store=FakeStore())
            self.assertTrue(ok)
            self.assertIn("1/2", msg)
            self.assertIn("bad.pdf", msg)

    def test_no_pdfs_found(self):
        import backend
        import config
        cfg = config.load_config()
        with tempfile.TemporaryDirectory() as tmp:
            ok, msg = backend.create_vector_db_from_folder(tmp, config=cfg)
            self.assertFalse(ok)
            self.assertIn("No PDF", msg)

    def test_missing_folder(self):
        import backend
        import config
        cfg = config.load_config()
        ok, msg = backend.create_vector_db_from_folder(
            os.path.join(tempfile.gettempdir(), "no-such-dir-xyz"), config=cfg)
        self.assertFalse(ok)

    def test_all_failed(self):
        import backend
        import config
        cfg = config.load_config()
        with tempfile.TemporaryDirectory() as tmp:
            fake_report = {"found": 1, "succeeded": 0, "failed": 1,
                           "failed_files": ["bad.pdf"]}
            with mock.patch("document_loader.load_documents_from_folder",
                            return_value=([], fake_report)):
                ok, msg = backend.create_vector_db_from_folder(
                    tmp, config=cfg, vector_store=mock.MagicMock())
            self.assertFalse(ok)
            self.assertIn("bad.pdf", msg)

    def test_query_handles_db_unavailable(self):
        import backend
        import config
        cfg = config.load_config()

        class BrokenStore:
            def search(self, q, k=5):
                raise RuntimeError("db down")

            def get_status(self):
                raise RuntimeError("down")

        ans, sources = backend.query_documents("hi", config=cfg,
                                               vector_store=BrokenStore())
        self.assertEqual(sources, [])
        self.assertIn("unavailable", ans.lower())

    def test_ui_has_no_offline_claim(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("100% Offline", src)
        self.assertIn("Environment", src)


if __name__ == "__main__":
    unittest.main()
