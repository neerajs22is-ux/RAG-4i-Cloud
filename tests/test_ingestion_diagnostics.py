"""Ingestion reliability tests: paths, PDFs, chunks, cloud-mode, reporting."""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_pdf_bytes(texts):
    """Minimal valid PDF bytes; texts=[] -> blank page without text."""
    n = max(len(texts), 1)
    objs = [(1, "<< /Type /Catalog /Pages 2 0 R >>"),
            (2, f"<< /Type /Pages /Kids [{' '.join(f'{3+i} 0 R' for i in range(n))}] /Count {n} >>")]
    for i in range(n):
        objs.append((3 + i,
                     "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                     f"/Contents {3+n+i} 0 R "
                     f"/Resources << /Font << /F1 {3+2*n} 0 R >> >> >>"))
    for i in range(n):
        s = ""
        if i < len(texts):
            esc = texts[i].replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            s = f"BT /F1 12 Tf 72 720 Td ({esc}) Tj ET"
        objs.append((3 + n + i, f"<< /Length {len(s)} >>\nstream\n{s}\nendstream"))
    objs.append((3 + 2 * n, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))
    out = bytearray(b"%PDF-1.4\n")
    offs = {}
    for num, body in objs:
        offs[num] = len(out)
        out += f"{num} 0 obj\n{body}\nendobj\n".encode("latin-1")
    top = max(offs)
    xp = len(out)
    out += f"xref\n0 {top+1}\n0000000000 65535 f \n".encode("latin-1")
    for i in range(1, top + 1):
        out += f"{offs[i]:010d} 00000 n \n".encode("latin-1")
    out += (f"trailer\n<< /Size {top+1} /Root 1 0 R >>\n"
            f"startxref\n{xp}\n%%EOF").encode("latin-1")
    return bytes(out)


def write_pdf(path, texts):
    with open(path, "wb") as f:
        f.write(make_pdf_bytes(texts))


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})
        self.id = None


class FakeVectorStore:
    def __init__(self):
        self.chunks = []

    def build_index(self, chunks):
        self.chunks.extend(list(chunks))
        return len(chunks)

    def search(self, query, k=5):
        return [(c, 0.9) for c in self.chunks[:k]]

    def get_status(self):
        return {"ready": bool(self.chunks), "exists": bool(self.chunks),
                "chunk_count": len(self.chunks), "document_count": 1,
                "persist_directory": "fake"}


class TestLocalPathIngestion(unittest.TestCase):
    def test_readable_pdf_ingests_with_counts(self):
        import backend
        import config
        cfg = config.load_config()
        with tempfile.TemporaryDirectory() as tmp:
            write_pdf(os.path.join(tmp, "lease.pdf"),
                      ["The lock-in period is 36 months."])
            ok, msg, det = backend.ingest_with_report(
                tmp, config=cfg, vector_store=FakeVectorStore())
            self.assertTrue(ok)
            self.assertEqual(det["source"], "local")
            self.assertEqual(det["found"], 1)
            self.assertEqual(det["pages"], 1)
            self.assertGreater(det["chars"], 0)
            self.assertGreater(det["chunks"], 0)
            self.assertEqual(det["embeddings"], det["chunks"])
            self.assertEqual(det["vectors_stored"], det["chunks"])
            self.assertEqual(det["failed"], 0)

    def test_nonexistent_path_explicit(self):
        import backend
        import config
        ok, msg, det = backend.ingest_with_report(
            os.path.join(tempfile.gettempdir(), "no-such-dir-xyz"),
            config=config.load_config(), vector_store=FakeVectorStore())
        self.assertFalse(ok)
        self.assertIn("does not exist", msg)
        self.assertEqual(det["found"], 0)

    def test_empty_folder_explicit(self):
        import backend
        import config
        with tempfile.TemporaryDirectory() as tmp:
            ok, msg, det = backend.ingest_with_report(
                tmp, config=config.load_config(),
                vector_store=FakeVectorStore())
            self.assertFalse(ok)
            self.assertIn("No PDF", msg)

    def test_windows_path_rejected_off_windows(self):
        import backend
        import config
        with mock.patch.object(backend.os, "name", "posix"):
            ok, msg, det = backend.ingest_with_report(
                r"C:\Users\bob\Documents",
                config=config.load_config(),
                vector_store=FakeVectorStore())
            self.assertFalse(ok)
            self.assertIn("Windows path", msg)

    def test_no_text_pdf_says_so(self):
        from pypdf import PdfWriter
        import backend
        import config
        with tempfile.TemporaryDirectory() as tmp:
            w = PdfWriter()
            w.add_blank_page(width=612, height=792)
            with open(os.path.join(tmp, "blank.pdf"), "wb") as f:
                w.write(f)
            ok, msg, det = backend.ingest_with_report(
                tmp, config=config.load_config(),
                vector_store=FakeVectorStore())
            self.assertFalse(ok)
            self.assertIn("zero usable text", msg)
            self.assertEqual(det["chars"], 0)

    def test_corrupt_pdf_reports_reason(self):
        import backend
        import config
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "bad.pdf"), "wb") as f:
                f.write(b"%PDF-1.4\nNOT REAL \x00\xff")
            ok, msg, det = backend.ingest_with_report(
                tmp, config=config.load_config(),
                vector_store=FakeVectorStore())
            self.assertFalse(ok)
            self.assertEqual(det["failed"], 1)
            self.assertEqual(det["failed_files"][0]["file"], "bad.pdf")
            self.assertTrue(det["failed_files"][0]["reason"])


class TestCloudModeIgnoresLocalPath(unittest.TestCase):
    def _s3_cfg(self, **kw):
        import config
        base = config.load_config().__dict__
        base.update({"document_storage": "s3", "s3_bucket": "bkt"})
        base.update(kw)
        return config.load_config().__class__(**base)

    def test_s3_uses_bucket_not_folder(self):
        import backend

        class FakeS3:
            bucket = "bkt"

            def list_documents(self, pattern="**/*.pdf"):
                return ["documents/lease.pdf"]

            def get_document(self, key):
                return make_pdf_bytes(["The lock-in period is 36 months."])

        ok, msg, det = backend.ingest_with_report(
            r"C:\Users\bob\Documents",  # must be ignored, not fail
            config=self._s3_cfg(), storage=FakeS3(),
            vector_store=FakeVectorStore())
        self.assertTrue(ok)
        self.assertEqual(det["source"], "s3")
        self.assertEqual(det["found"], 1)

    def test_s3_metadata_uses_s3_identifier(self):
        from document_loader import load_documents_from_s3

        class FakeS3:
            bucket = "bkt"

            def list_documents(self, pattern="**/*.pdf"):
                return ["documents/lease.pdf"]

            def get_document(self, key):
                return make_pdf_bytes(["hello world"])

        docs, rep = load_documents_from_s3(FakeS3())
        self.assertEqual(rep["pages"], 1)
        self.assertTrue(rep["chars"] > 0)
        self.assertTrue(docs[0].metadata["source_path"].startswith("s3://bkt/"))
        self.assertEqual(docs[0].metadata["file_name"], "lease.pdf")
        self.assertEqual(docs[0].metadata["page"], 0)


class TestRetrievalKnownContent(unittest.TestCase):
    def test_known_content_retrieved_above_threshold(self):
        import backend
        import config
        cfg = config.load_config()
        store = FakeVectorStore()
        # Build chunks directly for determinism.
        from chunking import chunk_documents
        docs = [FakeDoc("The lock-in period is 36 months. " * 5,
                        {"source_path": "/t/lease.pdf", "source": "/t/lease.pdf",
                         "file_name": "lease.pdf", "page": 2,
                         "document_id": "d1"})]
        chunks = chunk_documents(docs)
        self.assertTrue(len(chunks) >= 1)
        store.build_index(chunks)
        res = backend.retrieve_documents("lock-in period", config=cfg,
                                         vector_store=store)
        self.assertTrue(res)
        self.assertEqual(res[0]["file_name"], "lease.pdf")
        self.assertEqual(res[0]["page"], 2)
        self.assertGreaterEqual(res[0]["score"], cfg.relevance_threshold)

    def test_report_shape_complete(self):
        import backend
        import config
        with tempfile.TemporaryDirectory() as tmp:
            write_pdf(os.path.join(tmp, "a.pdf"), ["hello"])
            _, _, det = backend.ingest_with_report(
                tmp, config=config.load_config(),
                vector_store=FakeVectorStore())
            for key in ("source", "found", "pages", "chars", "chunks",
                        "embeddings", "vectors_stored", "succeeded",
                        "failed", "failed_files"):
                self.assertIn(key, det)


if __name__ == "__main__":
    unittest.main()
