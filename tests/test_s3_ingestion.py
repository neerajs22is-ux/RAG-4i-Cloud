"""Mocked-boto3 S3 ingestion wiring: storage -> loader -> migrate.

boto3.client is mocked (no AWS/network); a real minimal PDF flows through
PyPDFLoader so extraction is genuinely exercised.
"""

import io
import os
import sys
import tempfile
import unittest
from unittest import mock

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


class MockS3Client:
    """In-memory boto3 S3 stand-in (paginated list, get/put/delete)."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken) if ContinuationToken else 0
        page = keys[start:start + 1]
        resp = {"Contents": [{"Key": k} for k in page]}
        if start + 1 < len(keys):
            resp["NextContinuationToken"] = str(start + 1)
        return resp

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise Exception("NoSuchKey: " + Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):
        data = Body.read() if hasattr(Body, "read") else bytes(Body)
        self.objects[Key] = data
        return {}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)
        return {}


LEASE_PDF = make_pdf_bytes("The lock-in period is 36 months.")


class S3IngestionCase(unittest.TestCase):
    def _store(self, objects=None, **kw):
        from s3_document_storage import S3DocumentStorage
        kw.setdefault("bucket", "test-bucket")
        kw.setdefault("prefix", "documents/")
        kw.setdefault("region", "ap-south-2")
        with mock.patch("boto3.client", return_value=MockS3Client(objects)):
            store = S3DocumentStorage(**kw)
            # Pin the mocked client (factory would rebuild it per call).
            store._client = MockS3Client(objects)
        return store

    def test_loader_reads_s3_pdfs_with_metadata(self):
        from document_loader import load_documents_from_s3
        store = self._store({"documents/lease.pdf": LEASE_PDF})
        docs, report = load_documents_from_s3(store)
        self.assertEqual(report["found"], 1)
        self.assertEqual(report["succeeded"], 1)
        self.assertGreater(report["chars"], 0)
        self.assertIn("36 months", docs[0].page_content)
        self.assertTrue(docs[0].metadata["source_path"].startswith("s3://"))
        self.assertEqual(docs[0].metadata["file_name"], "lease.pdf")

    def test_loader_records_per_file_failure(self):
        from document_loader import load_documents_from_s3
        store = self._store({"documents/bad.pdf": b"not a pdf at all"})
        _, report = load_documents_from_s3(store)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["failed_files"], ["bad.pdf"])

    def test_migrate_uploads_local_pdfs(self):
        import config
        import migrate_to_s3
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("lease.pdf", "contract.pdf"):
                with open(os.path.join(tmp, name), "wb") as f:
                    f.write(LEASE_PDF)
            store = self._store()
            cfg = config.load_config().__class__(
                **{**config.load_config().__dict__, "s3_bucket": "test-bucket"})
            with mock.patch("s3_document_storage.get_s3_store",
                            return_value=store), \
                 mock.patch("migrate_to_s3.get_config", return_value=cfg):
                ok, msg = migrate_to_s3.migrate(tmp)
            self.assertTrue(ok)
            self.assertIn("Found=2 uploaded=2", msg)
            self.assertEqual(
                store.get_document("documents/contract.pdf"), LEASE_PDF)


if __name__ == "__main__":
    unittest.main()
