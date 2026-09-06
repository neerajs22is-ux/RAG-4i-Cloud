"""Ingestion/chunking/metadata tests (mocked where heavy deps missing)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})


class TestChunking(unittest.TestCase):
    def test_chunk_size_overlap_preserved(self):
        import chunking
        self.assertEqual(chunking.CHUNK_SIZE, 1000)
        self.assertEqual(chunking.CHUNK_OVERLAP, 200)

    def test_chunking_splits_and_adds_chunk_id(self):
        try:
            import chunking
        except ImportError as e:
            self.skipTest(f"chunking deps missing: {e}")
        try:
            long_text = "Hello world. " * 200  # ~2600 chars -> multiple chunks
            docs = [FakeDoc(long_text, {"source_path": "/tmp/a.pdf",
                                        "source": "/tmp/a.pdf",
                                        "file_name": "a.pdf",
                                        "page": 3,
                                        "document_id": "abc"})]
            chunks = chunking.chunk_documents(docs)
        except ImportError as e:
            self.skipTest(f"splitter dep missing: {e}")
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertIn("chunk_id", c.metadata)
            self.assertEqual(c.metadata["file_name"], "a.pdf")
            self.assertEqual(c.metadata["page"], 3)
            self.assertEqual(c.metadata["document_id"], "abc")
        # chunk_ids unique
        ids = [c.metadata["chunk_id"] for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_page_none_when_missing(self):
        try:
            import chunking
            chunks = chunking.chunk_documents([FakeDoc("short text", {})])
        except ImportError as e:
            self.skipTest(f"splitter dep missing: {e}")
        self.assertIsNone(chunks[0].metadata["page"])


class TestDocumentStorage(unittest.TestCase):
    def test_list_save_delete_roundtrip(self):
        from document_storage import LocalDocumentStorage
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalDocumentStorage(tmp)
            self.assertEqual(store.list_documents(), [])
            # save a fake pdf
            p = store.save_document("sub/doc.pdf", b"%PDF-1.4 fake")
            self.assertTrue(os.path.isfile(p))
            listed = store.list_documents()
            self.assertEqual(len(listed), 1)
            self.assertTrue(listed[0].endswith("doc.pdf"))
            data = store.get_document(p)
            self.assertEqual(data, b"%PDF-1.4 fake")
            store.delete_document(p)
            self.assertEqual(store.list_documents(), [])

    def test_list_ignores_non_pdf(self):
        from document_storage import LocalDocumentStorage
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a.txt"), "w", encoding="utf-8") as f:
                f.write("hi")
            store = LocalDocumentStorage(tmp)
            self.assertEqual(store.list_documents(), [])


class TestLoaderMetadata(unittest.TestCase):
    def test_standardize_metadata(self):
        from document_loader import stable_document_id, standardize_document_metadata
        m = standardize_document_metadata("/tmp/Lease.pdf", 14)
        self.assertEqual(m["file_name"], "Lease.pdf")
        self.assertEqual(m["page"], 14)
        self.assertIn("document_id", m)
        self.assertIn("source_path", m)
        # stable id
        self.assertEqual(m["document_id"], stable_document_id("/tmp/Lease.pdf"))

    def test_page_none_fallback(self):
        from document_loader import standardize_document_metadata
        m = standardize_document_metadata("/tmp/x.pdf", "not-an-int")
        self.assertIsNone(m["page"])
        m2 = standardize_document_metadata("/tmp/x.pdf", None)
        self.assertIsNone(m2["page"])


if __name__ == "__main__":
    unittest.main()
