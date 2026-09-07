"""Opt-in real-RDS integration test (NOT run by default).

Run only with an SSH tunnel up (localhost:15432 -> rag4i-db:5432):

    $env:RUN_PG_INTEGRATION="1"
    python -m unittest tests.integration.test_postgres_rds -v

Reads connection from env (DB_* / DATABASE_URL); no credentials in source.
Uses a temp table so the real index is never disturbed.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

RUN = os.environ.get("RUN_PG_INTEGRATION") == "1"


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})
        self.id = None


@unittest.skipUnless(RUN, "opt-in: set RUN_PG_INTEGRATION=1 with tunnel up")
class TestPostgresRDS(unittest.TestCase):
    TABLE = "chunks_e2e_tmp"

    def test_lease_contract_roundtrip(self):
        import config
        from embeddings import get_embedding_provider
        from postgres_vector_store import get_postgres_store

        cfg = config.load_config()
        if not (cfg.db_user and cfg.db_password) and not cfg.database_url:
            self.skipTest("DB credentials not configured in env")
        store = get_postgres_store(cfg, get_embedding_provider(cfg),
                                   table=self.TABLE)
        try:
            status = store.get_status()
        except Exception as e:
            self.skipTest(f"RDS unreachable (is the tunnel up?): {e}")
        if status.get("error") and not status.get("exists"):
            self.skipTest(f"RDS unreachable: {status.get('error')}")
        store.ensure_schema()  # base chunks table + extension
        # Create isolated table mirroring chunks schema.
        import psycopg2
        conn = psycopg2.connect(cfg.postgres_dsn())
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    cur.execute(
                        f"CREATE TABLE IF NOT EXISTS {self.TABLE} "
                        "(LIKE chunks INCLUDING ALL)")
        finally:
            conn.close()
        docs = [
            FakeDoc("The lock-in period in the lease deed is 36 months.",
                    {"chunk_id": "e2e-lease-1", "document_id": "lease.pdf",
                     "source_path": "lease.pdf", "source": "lease.pdf",
                     "file_name": "lease.pdf", "page": 4}),
            FakeDoc("The service contract renews annually with 30 days notice.",
                    {"chunk_id": "e2e-contract-1", "document_id": "contract.pdf",
                     "source_path": "contract.pdf", "source": "contract.pdf",
                     "file_name": "contract.pdf", "page": 1}),
        ]
        try:
            n = store.build_index(docs)
            self.assertGreaterEqual(n, 1)
            res = store.search("What is the lock-in period?", k=5)
            self.assertLessEqual(len(res), 5)
            self.assertTrue(res)
            top_doc, top_score = res[0]
            self.assertIn("36 months", top_doc.page_content)
            self.assertEqual(top_doc.metadata["file_name"], "lease.pdf")
            self.assertIsNotNone(top_score)
            st = store.get_status()
            # Status of main table still reportable; temp table check:
            self.assertIn("chunk_count", st)
        finally:
            conn = psycopg2.connect(cfg.postgres_dsn())
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(f"DROP TABLE IF EXISTS {self.TABLE}")
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
