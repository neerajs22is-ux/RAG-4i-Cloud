"""Ingestion progress tests: ordering, values, outcomes (deterministic)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_pdf_bytes(text):
    esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = "BT /F1 12 Tf 72 720 Td (" + esc + ") Tj ET"
    objs = [(1, "<< /Type /Catalog /Pages 2 0 R >>"),
            (2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
            (3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"),
            (4, "<< /Length " + str(len(stream)) + " >>\nstream\n"
                + stream + "\nendstream"),
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


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, current, total, filename, outcome):
        self.calls.append((current, total, filename, outcome))


class TestFolderProgress(unittest.TestCase):
    def test_ordering_values_outcomes(self):
        from document_loader import load_documents_from_folder
        from document_storage import LocalDocumentStorage
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a.pdf"), "wb") as f:
                f.write(make_pdf_bytes("Alpha content here."))
            with open(os.path.join(tmp, "b.pdf"), "wb") as f:
                f.write(make_pdf_bytes("Beta content here."))
            with open(os.path.join(tmp, "bad.pdf"), "wb") as f:
                f.write(b"%PDF-1.4\nbroken \x00\xff")
            rec = Recorder()
            docs, report = load_documents_from_folder(
                tmp, LocalDocumentStorage(tmp), on_progress=rec)
            self.assertEqual(report["found"], 3)
            # Fires once per file, in sorted processing order, 1-based.
            self.assertEqual([c[0] for c in rec.calls], [1, 2, 3])
            self.assertTrue(all(c[1] == 3 for c in rec.calls))
            self.assertEqual([c[2] for c in rec.calls],
                             ["a.pdf", "b.pdf", "bad.pdf"])
            by_file = {c[2]: c[3] for c in rec.calls}
            self.assertEqual(by_file, {"a.pdf": "ok", "b.pdf": "ok",
                                       "bad.pdf": "failed"})
            self.assertTrue(docs)

    def test_no_callback_without_listener(self):
        from document_loader import load_documents_from_folder
        from document_storage import LocalDocumentStorage
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a.pdf"), "wb") as f:
                f.write(make_pdf_bytes("Alpha content here."))
            docs, report = load_documents_from_folder(
                tmp, LocalDocumentStorage(tmp))
            self.assertEqual(report["succeeded"], 1)


class FakeS3Store:
    bucket = "bkt"

    def __init__(self, objects):
        self._objects = dict(objects)

    def list_documents(self, pattern="**/*.pdf"):
        return sorted(self._objects)

    def get_document(self, path):
        return self._objects[path]


class TestS3Progress(unittest.TestCase):
    def test_s3_progress_consistent(self):
        from document_loader import load_documents_from_s3
        store = FakeS3Store({
            "documents/lease.pdf": make_pdf_bytes("Lease text here."),
            "documents/contract.pdf": make_pdf_bytes("Contract text here."),
        })
        rec = Recorder()
        docs, report = load_documents_from_s3(store, on_progress=rec)
        self.assertEqual(report["found"], 2)
        self.assertEqual([(c[0], c[1], c[3]) for c in rec.calls],
                         [(1, 2, "ok"), (2, 2, "ok")])
        self.assertEqual(len(docs), 2)

    def test_backend_forwards_callback(self):
        import backend
        import config

        class FakeVS:
            def build_index(self, chunks):
                return len(chunks)

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a.pdf"), "wb") as f:
                f.write(make_pdf_bytes("Alpha content here."))
            rec = Recorder()
            ok, _, details = backend.ingest_with_report(
                tmp, config=config.load_config(), vector_store=FakeVS(),
                on_progress=rec)
            self.assertTrue(ok)
            self.assertEqual(rec.calls, [(1, 1, "a.pdf", "ok")])
            self.assertEqual(details["found"], 1)


if __name__ == "__main__":
    unittest.main()
