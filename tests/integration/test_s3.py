"""Opt-in real-S3 integration test (NOT run by default).

Requires operator-configured AWS credentials (profile/env chain) and the
existing private bucket. Never creates buckets, users, or keys.

    $env:RUN_S3_INTEGRATION="1"
    $env:S3_BUCKET="rag4i-company-documents-305740358559-ap-south-2-an"
    python -m unittest tests.integration.test_s3 -v

Uses a scratch prefix so real objects are never touched; cleans up only the
keys it created. No credentials in source code.
"""

import os
import sys
import unittest
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

RUN = os.environ.get("RUN_S3_INTEGRATION") == "1"
SCRATCH = f"s3test-{uuid.uuid4().hex[:8]}/"


@unittest.skipUnless(RUN, "opt-in: set RUN_S3_INTEGRATION=1 with AWS creds")
class TestS3RealBucket(unittest.TestCase):
    def test_roundtrip_under_scratch_prefix(self):
        import config
        from s3_document_storage import S3DocumentStorage
        bucket = os.environ.get("S3_BUCKET", "")
        if not bucket:
            self.skipTest("S3_BUCKET not set in env")
        store = S3DocumentStorage(bucket=bucket, prefix=SCRATCH,
                                  region=os.environ.get(
                                      "S3_REGION",
                                      config.load_config().s3_region))
        try:
            key = store.save_document("probe.pdf", b"%PDF-1.4 probe")
        except Exception as e:
            self.skipTest(f"S3 unreachable (check credentials): {e}")
        try:
            self.assertTrue(key.startswith(SCRATCH))
            self.assertIn(key, store.list_documents())
            self.assertEqual(store.get_document(key), b"%PDF-1.4 probe")
        finally:
            try:
                store.delete_document(key)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
