"""S3 storage provider tests (mocked; no AWS credentials or network)."""

import io
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeS3Client:
    """Minimal boto3 S3 stand-in (paginated list, get/put/delete)."""

    def __init__(self, objects=None):
        # objects: {key: bytes}
        self.objects = dict(objects or {})
        self.put_calls = []
        self.delete_calls = []

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        # Force pagination: one key per page.
        start = int(ContinuationToken) if ContinuationToken else 0
        page = keys[start:start + 1]
        resp = {"Contents": [{"Key": k} for k in page]}
        if start + 1 < len(keys):
            resp["NextContinuationToken"] = str(start + 1)
        return resp

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise Exception(f"NoSuchKey: {Key}")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):
        data = Body.read() if hasattr(Body, "read") else bytes(Body)
        self.objects[Key] = data
        self.put_calls.append(Key)
        return {}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)
        self.delete_calls.append(Key)
        return {}


def make_store(objects=None, **kw):
    from s3_document_storage import S3DocumentStorage
    kw.setdefault("bucket", "test-bucket")
    kw.setdefault("prefix", "documents/")
    kw.setdefault("region", "ap-south-2")
    kw.setdefault("client", FakeS3Client(objects))
    return S3DocumentStorage(**kw)


class TestS3Config(unittest.TestCase):
    def setUp(self):
        self._old = dict(os.environ)
        for k in ["DOCUMENT_STORAGE", "S3_BUCKET", "S3_REGION", "S3_PREFIX"]:
            os.environ.pop(k, None)
        import config
        config.reset_config_cache()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old)
        import config
        config.reset_config_cache()

    def test_defaults_keep_local(self):
        import config
        cfg = config.load_config()
        self.assertEqual(cfg.document_storage, "local")
        self.assertEqual(cfg.s3_region, "ap-south-2")
        self.assertEqual(cfg.s3_prefix, "documents/")
        self.assertEqual(cfg.s3_bucket, "")

    def test_env_override(self):
        os.environ.update({"DOCUMENT_STORAGE": "s3",
                           "S3_BUCKET": "my-bucket",
                           "S3_PREFIX": "docs"})
        import config
        config.reset_config_cache()
        cfg = config.load_config()
        self.assertEqual(cfg.document_storage, "s3")
        self.assertEqual(cfg.s3_bucket, "my-bucket")


class TestS3Keys(unittest.TestCase):
    def test_key_generation_preserves_filename(self):
        s = make_store()
        self.assertEqual(s.key_for("lease.pdf"), "documents/lease.pdf")
        self.assertEqual(s.key_for("sub/lease.pdf"), "documents/lease.pdf")

    def test_prefix_normalized(self):
        from s3_document_storage import normalize_prefix
        self.assertEqual(normalize_prefix("documents"), "documents/")
        self.assertEqual(normalize_prefix("/documents//"), "documents/")
        self.assertEqual(normalize_prefix(""), "")

    def test_bucket_required(self):
        from s3_document_storage import S3DocumentStorage
        with self.assertRaises(ValueError):
            S3DocumentStorage(bucket="")


class TestS3Operations(unittest.TestCase):
    def test_list_filters_pdfs_sorted(self):
        s = make_store({"documents/lease.pdf": b"a",
                        "documents/notes.txt": b"b",
                        "documents/contract.PDF": b"c",
                        "other/x.pdf": b"d"})
        self.assertEqual(s.list_documents(),
                         ["documents/contract.PDF", "documents/lease.pdf"])

    def test_get_accepts_key_or_filename(self):
        s = make_store({"documents/lease.pdf": b"PDFBYTES"})
        self.assertEqual(s.get_document("documents/lease.pdf"), b"PDFBYTES")
        self.assertEqual(s.get_document("lease.pdf"), b"PDFBYTES")

    def test_save_and_delete_roundtrip(self):
        s = make_store()
        key = s.save_document("lease.pdf", b"data")
        self.assertEqual(key, "documents/lease.pdf")
        self.assertEqual(s.get_document(key), b"data")
        s.delete_document(key)
        self.assertEqual(s.list_documents(), [])

    def test_get_missing_raises(self):
        s = make_store()
        with self.assertRaises(Exception):
            s.get_document("missing.pdf")


class TestProviderSwitching(unittest.TestCase):
    def test_factory_local_default_s3_opt_in(self):
        import config
        from document_storage import LocalDocumentStorage, get_document_storage
        from s3_document_storage import S3DocumentStorage
        config.reset_config_cache()
        self.assertIsInstance(get_document_storage(config.load_config(), "/tmp"),
                              LocalDocumentStorage)
        os.environ["DOCUMENT_STORAGE"] = "s3"
        os.environ["S3_BUCKET"] = "bkt"
        try:
            config.reset_config_cache()
            vs = get_document_storage(config.load_config())
            self.assertIsInstance(vs, S3DocumentStorage)
            self.assertEqual(vs.describe(), "S3 s3://bkt/documents/")
        finally:
            del os.environ["DOCUMENT_STORAGE"]
            del os.environ["S3_BUCKET"]
            config.reset_config_cache()


class TestS3Migration(unittest.TestCase):
    def _pdf_dir(self, tmp, names):
        for n in names:
            with open(os.path.join(tmp, n), "wb") as f:
                f.write(b"%PDF-1.4 fake " + n.encode())
        return tmp

    def test_reports_found_uploaded_keys(self):
        import config
        import migrate_to_s3
        with tempfile.TemporaryDirectory() as tmp:
            self._pdf_dir(tmp, ["lease.pdf", "contract.pdf"])
            cfg = config.load_config().__class__(
                **{**config.load_config().__dict__, "s3_bucket": "bkt"})
            with mock.patch("s3_document_storage.get_s3_store",
                            return_value=make_store()), \
                 mock.patch("migrate_to_s3.get_config", return_value=cfg):
                ok, msg = migrate_to_s3.migrate(tmp)
            self.assertTrue(ok)
            self.assertIn("Found=2 uploaded=2", msg)
            self.assertIn("documents/lease.pdf", msg)

    def test_skips_existing_never_overwrites(self):
        import config
        import migrate_to_s3
        with tempfile.TemporaryDirectory() as tmp:
            self._pdf_dir(tmp, ["lease.pdf", "contract.pdf"])
            store = make_store({"documents/lease.pdf": b"old"})
            cfg = config.load_config().__class__(
                **{**config.load_config().__dict__, "s3_bucket": "bkt"})
            with mock.patch("s3_document_storage.get_s3_store",
                            return_value=store), \
                 mock.patch("migrate_to_s3.get_config", return_value=cfg):
                ok, msg = migrate_to_s3.migrate(tmp)
            self.assertTrue(ok)
            self.assertIn("uploaded=1 skipped=1", msg)
            # Existing object untouched.
            self.assertEqual(store.get_document("documents/lease.pdf"), b"old")

    def test_error_handling_lists_failures(self):
        import config
        import migrate_to_s3
        with tempfile.TemporaryDirectory() as tmp:
            self._pdf_dir(tmp, ["lease.pdf"])
            store = make_store()
            with mock.patch.object(store, "save_document",
                                   side_effect=Exception("denied")):
                cfg = config.load_config().__class__(
                    **{**config.load_config().__dict__, "s3_bucket": "bkt"})
                with mock.patch("s3_document_storage.get_s3_store",
                                return_value=store), \
                     mock.patch("migrate_to_s3.get_config",
                                return_value=cfg):
                    ok, msg = migrate_to_s3.migrate(tmp)
                self.assertFalse(ok)
                self.assertIn("lease.pdf", msg)


if __name__ == "__main__":
    unittest.main()
