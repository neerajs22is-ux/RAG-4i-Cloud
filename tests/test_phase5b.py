"""Phase 5B tests: scope primitives, isolation, migration, frozen semantics.

Chroma tests use a real local index (tmp dir) with dummy embeddings;
PostgreSQL tests mock psycopg2 (RDS stays opt-in). No network, no secrets.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SID_A = "a" * 32
SID_B = "b" * 32


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})
        self.id = None


class DummyEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.0]


def lease_chunk(cid="c1"):
    return FakeDoc("lease lock-in period text alpha",
                   {"file_name": "lease.pdf", "page": 0,
                    "document_id": "d1", "chunk_id": cid})


def session_chunk(cid="c2", fname="sess.pdf"):
    return FakeDoc("session alpha confidential terms beta",
                   {"file_name": fname, "page": 0,
                    "document_id": "d2", "chunk_id": cid})


class ScopeStore:
    """In-memory scope-aware store (mirrors provider filtering rules)."""

    def __init__(self):
        self.rows = []  # (content, file, scope, sid)

    def add(self, content, file, scope="persistent", sid=None):
        self.rows.append((content, file, scope, sid))

    def search(self, query, k=5, session_id=None):
        from document_scope import validate_session_id
        if session_id is not None:
            validate_session_id(session_id)
        out = []

        class D:
            def __init__(self, content, meta):
                self.page_content = content
                self.metadata = meta
                self.id = None

        for content, file, scope, sid in self.rows:
            if scope == "session" and sid != session_id:
                continue
            if scope not in ("persistent",) and scope != "session":
                continue
            if scope == "session" and session_id is None:
                continue
            out.append((D(content, {"file_name": file, "page": 0,
                                    "score": 0.9, "source": "/t/" + file,
                                    "source_path": "/t/" + file,
                                    "document_id": "d-" + file,
                                    "chunk_id": "c-" + file + content[:3]}),
                        0.9))
        return out[:k]

    def chunks_for_source(self, file_name, limit=8, session_id=None):
        return [(d, None) for d, _s in
                self.search("x", k=limit, session_id=session_id)
                if d.metadata["file_name"] == file_name][:limit]


class TestSessionIds(unittest.TestCase):
    def test_format_and_uniqueness(self):
        from document_scope import is_valid_session_id, new_session_id
        a, b = new_session_id(), new_session_id()
        self.assertEqual(len(a), 32)
        self.assertTrue(is_valid_session_id(a))
        self.assertNotEqual(a, b)

    def test_opaque_not_derived(self):
        from document_scope import new_session_id
        sid = new_session_id()
        for fragment in ("lease", "pdf", "user", "operator", "ec2-user"):
            self.assertNotIn(fragment, sid)

    def test_invalid_rejected(self):
        from document_scope import validate_session_id
        for bad in ("", None, "short", "z" * 32, "A" * 32,
                    "/home/ec2-user/x.pdf", "lease.pdf", 12345,
                    "a" * 31, "a" * 33, "a" * 31 + " "):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_session_id(bad)
                from document_scope import is_valid_session_id
                self.assertFalse(is_valid_session_id(bad))


class TestResolveChunkScope(unittest.TestCase):
    def test_defaults_persistent(self):
        from document_scope import PERSISTENT, resolve_chunk_scope
        scope, sid = resolve_chunk_scope({}, None)
        self.assertEqual((scope, sid), (PERSISTENT, None))

    def test_session_batch_stamps(self):
        from document_scope import SESSION, resolve_chunk_scope
        scope, sid = resolve_chunk_scope({}, SID_A)
        self.assertEqual((scope, sid), (SESSION, SID_A))

    def test_rejections(self):
        from document_scope import resolve_chunk_scope
        # Session scope without any session id.
        with self.assertRaises(ValueError):
            resolve_chunk_scope({"scope": "session"}, None)
        # Batch sid with pre-stamped persistent chunk.
        with self.assertRaises(ValueError):
            resolve_chunk_scope({"scope": "persistent"}, SID_A)
        # Cross-session write.
        with self.assertRaises(ValueError):
            resolve_chunk_scope(
                {"scope": "session", "session_id": SID_A}, SID_B)
        # Unknown scope value.
        with self.assertRaises(ValueError):
            resolve_chunk_scope({"scope": "other"}, None)
        # Malformed batch sid.
        with self.assertRaises(ValueError):
            resolve_chunk_scope({}, "bogus")
        # Persistent chunk carrying a stray session id.
        with self.assertRaises(ValueError):
            resolve_chunk_scope(
                {"scope": "persistent", "session_id": SID_A}, None)


class TestChromaScope(unittest.TestCase):
    def setUp(self):
        from vector_store import ChromaVectorStore
        self.tmp = tempfile.mkdtemp(prefix="scope5b_")
        self.vs = ChromaVectorStore(
            persist_directory=os.path.join(self.tmp, "db"),
            embedding_function=DummyEmbeddings())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ids(self, hits):
        return sorted(d.metadata.get("chunk_id") for d, _s in hits)

    def test_legacy_corpus_unchanged(self):
        # Pre-scope rows: searchable unbound AND stamped persistent.
        self.vs.build_index([lease_chunk()])
        hits = self.vs.search("lease lock-in alpha", k=10)
        self.assertEqual(self._ids(hits), ["c1"])
        db = self.vs._load_existing()
        metas = db._collection.get(include=["metadatas"])["metadatas"]
        self.assertTrue(all(m.get("scope") == "persistent" for m in metas))

    def test_session_isolation(self):
        self.vs.build_index([lease_chunk()])
        self.vs.build_index([session_chunk()], session_id=SID_A)
        self.vs.build_index([session_chunk("c3")], session_id=SID_B)
        unbound = self._ids(self.vs.search("lease session terms", k=10))
        self.assertNotIn("c2", unbound)
        self.assertNotIn("c3", unbound)
        both_a = self._ids(self.vs.search("lease session terms", k=10,
                                          session_id=SID_A))
        self.assertIn("c1", both_a)
        self.assertIn("c2", both_a)
        self.assertNotIn("c3", both_a)  # zero cross-session leakage
        both_b = self._ids(self.vs.search("lease session terms", k=10,
                                          session_id=SID_B))
        self.assertIn("c3", both_b)
        self.assertNotIn("c2", both_b)

    def test_list_and_topup_scoped(self):
        self.vs.build_index([lease_chunk()])
        self.vs.build_index([session_chunk()], session_id=SID_A)
        names = {s["file_name"] for s in self.vs.list_sources()}
        self.assertEqual(names, {"lease.pdf"})
        names_a = {s["file_name"]
                   for s in self.vs.list_sources(session_id=SID_A)}
        self.assertEqual(names_a, {"lease.pdf", "sess.pdf"})
        self.assertEqual(
            [d.metadata["chunk_id"] for d, _ in
             self.vs.chunks_for_source("sess.pdf", session_id=SID_A)], ["c2"])
        self.assertEqual(self.vs.chunks_for_source("sess.pdf"), [])

    def test_invalid_session_rejected(self):
        self.vs.build_index([lease_chunk()])
        with self.assertRaises(ValueError):
            self.vs.search("x", session_id="bogus")
        with self.assertRaises(ValueError):
            self.vs.list_sources(session_id="bogus")
        with self.assertRaises(ValueError):
            self.vs.chunks_for_source("lease.pdf", session_id="bogus")
        with self.assertRaises(ValueError):
            self.vs.build_index([lease_chunk("c9")], session_id="bogus")

    def test_contradictory_write_rejected(self):
        self.vs.build_index([lease_chunk()])
        bad = lease_chunk("c9")
        bad.metadata["scope"] = "session"  # no session_id anywhere
        with self.assertRaises(ValueError):
            self.vs.build_index([bad])

    def test_k_preserved_with_scope(self):
        for i in range(4):
            self.vs.build_index([session_chunk(f"c{i}")], session_id=SID_A)
        hits = self.vs.search("session terms", k=2, session_id=SID_A)
        self.assertLessEqual(len(hits), 2)

    def test_deterministic_ids_exclude_scope(self):
        from vector_store import ChromaVectorStore
        a = [lease_chunk("cx")]
        b = [lease_chunk("cx")]
        ids_a = ChromaVectorStore._chunk_ids(a)
        # Same content re-indexed under a session: identical IDs (upsert).
        self.vs.build_index(a)
        self.vs.build_index(b, session_id=SID_A)
        self.assertEqual(ids_a, ChromaVectorStore._chunk_ids(b))

    def test_backfill_failure_fallback(self):
        self.vs.build_index([lease_chunk()])
        with mock.patch.object(
                type(self.vs), "_ensure_persistent_scope",
                return_value=False):
            # Unbound degrades to legacy behavior, never hides data.
            self.assertEqual(
                self._ids(self.vs.search("lease lock-in", k=10)), ["c1"])
            # Bound reads refuse rather than filter un-migrated data.
            with self.assertRaises(RuntimeError):
                self.vs.search("lease", session_id=SID_A)


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.statements = []
        self._rows = []
        self._fail_first = None
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        if self._fail_first is not None:
            exc, self._fail_first = self._fail_first, None
            raise exc

    def fetchall(self):
        return list(self._rows)


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self._cursor

    def close(self):
        pass


def pg_store(cursor):
    from postgres_vector_store import PostgresVectorStore
    store = PostgresVectorStore(
        dsn="postgresql://u:p@localhost:15432/rag4i",
        embedding_function=DummyEmbeddings())
    store._connect = lambda: FakeConn(cursor)
    return store


class UndefinedColumn(Exception):
    def __init__(self):
        super().__init__('column "scope" does not exist')
        self.pgcode = "42703"


class TestPostgresScope(unittest.TestCase):
    def test_schema_and_migration_additive(self):
        from postgres_vector_store import SCHEMA_SQL, SCOPE_MIGRATION_SQL
        low = SCHEMA_SQL.lower()
        self.assertIn("scope", low)
        self.assertIn("session_id", low)
        self.assertIn("vector(384)", low)
        self.assertIn("if not exists", SCOPE_MIGRATION_SQL.lower())
        self.assertIn("default 'persistent'", SCOPE_MIGRATION_SQL.lower())

    def test_build_inserts_scope(self):
        cur = FakeCursor(None)
        store = pg_store(cur)
        store.ensure_schema = lambda: None
        store.build_index([lease_chunk()], session_id=None)
        sql, params = cur.statements[-1]
        self.assertIn("scope", sql)
        self.assertIn("session_id", sql)
        self.assertIn("persistent", params)
        store.build_index([session_chunk()], session_id=SID_A)
        _sql, params = cur.statements[-1]
        self.assertIn(SID_A, params)
        self.assertIn("session", params)

    def test_build_rejects_bad_scope(self):
        cur = FakeCursor(None)
        store = pg_store(cur)
        store.ensure_schema = lambda: None
        with self.assertRaises(ValueError):
            store.build_index([lease_chunk()], session_id="bogus")
        bad = lease_chunk("c9")
        bad.metadata["scope"] = "session"
        with self.assertRaises(ValueError):
            store.build_index([bad])

    def test_search_predicate_and_params(self):
        cur = FakeCursor(None)
        cur._rows = [("t", "d", "/p", "f.pdf", 0, "c1", 0.2)]
        store = pg_store(cur)
        store.search("q", k=5, session_id=SID_A)
        sql, params = cur.statements[-1]
        self.assertIn("scope IS NULL", sql)
        self.assertIn("session_id = %s", sql)
        self.assertEqual(params[1], SID_A)
        self.assertEqual(params[2], 5)

    def test_unbound_excludes_session_rows(self):
        cur = FakeCursor(None)
        cur._rows = []
        store = pg_store(cur)
        store.search("q", k=5)
        _sql, params = cur.statements[-1]
        self.assertIsNone(params[1])  # NULL never matches session rows

    def test_legacy_null_rows_visible(self):
        # Rows predating the migration (scope NULL) map to persistent.
        cur = FakeCursor(None)
        cur._rows = [("legacy text", "d9", "/p/o.pdf", "o.pdf", 1, "c9", 0.1)]
        store = pg_store(cur)
        res = store.search("legacy", k=5)
        self.assertEqual(len(res), 1)
        doc, _score = res[0]
        self.assertEqual(doc.metadata["chunk_id"], "c9")

    def test_missing_column_retry(self):
        cur = FakeCursor(None)
        cur._rows = [("t", "d", "/p", "f.pdf", 0, "c1", 0.2)]
        cur._fail_first = UndefinedColumn()
        store = pg_store(cur)
        migrated = []
        store.ensure_schema = lambda: migrated.append(True)
        res = store.search("q", k=5)
        self.assertTrue(migrated)
        self.assertEqual(len(res), 1)

    def test_list_and_topup_scoped(self):
        cur = FakeCursor(None)
        cur._rows = [("f.pdf", "d1")]
        store = pg_store(cur)
        store.list_sources(session_id=SID_A)
        sql, params = cur.statements[-1]
        self.assertIn("session_id = %s", sql)
        self.assertEqual(params[0], SID_A)
        cur2 = FakeCursor(None)
        cur2._rows = [("t", "d", "/p", "f.pdf", 0, "c1")]
        store2 = pg_store(cur2)
        store2.chunks_for_source("f.pdf", session_id=SID_A)
        sql, params = cur2.statements[-1]
        self.assertIn("file_name = %s", sql)
        self.assertIn("session_id = %s", sql)

    def test_invalid_session_before_connect(self):
        cur = FakeCursor(None)
        for method, args in (
                ("search", ("q",)),
                ("chunks_for_source", ("f.pdf",)),
                ("list_sources", ())):
            store = pg_store(cur)
            store._connect = mock.MagicMock(
                side_effect=AssertionError("must not connect"))
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    if method == "search":
                        store.search(*args, session_id="bogus")
                    elif method == "chunks_for_source":
                        store.chunks_for_source(*args, session_id="bogus")
                    else:
                        store.list_sources(session_id="bogus")
                store._connect.assert_not_called()


class TestBackendScopeThreading(unittest.TestCase):
    def test_retrieve_scoped_and_unscoped(self):
        import backend
        import config
        store = ScopeStore()
        store.add("lease lock-in alpha", "lease.pdf", "persistent", None)
        store.add("session alpha beta", "sess.pdf", "session", SID_A)
        store.add("session bravo gamma", "sess.pdf", "session", SID_B)
        cfg = config.load_config()
        got = backend.retrieve_documents("lease lock-in", config=cfg,
                                         vector_store=store)
        self.assertEqual({s["file_name"] for s in got}, {"lease.pdf"})
        got_a = backend.retrieve_documents(
            "session alpha", config=cfg, vector_store=store, session_id=SID_A)
        files_a = {s["file_name"] for s in got_a}
        self.assertIn("sess.pdf", files_a)
        self.assertNotIn("bravo", " ".join(
            s["content"] for s in got_a if s["file_name"] == "sess.pdf"))
        with self.assertRaises(ValueError):
            backend.retrieve_documents("x", config=cfg, vector_store=store,
                                       session_id="bogus")

    def test_k_threshold_unchanged_with_scope(self):
        import backend
        import config
        store = ScopeStore()
        for i in range(4):
            store.add(f"payment term number {i} alpha", "p.pdf",
                      "persistent", None)
        cfg = config.load_config()
        got = backend.retrieve_documents("payment term alpha", config=cfg,
                                         vector_store=store, k=2,
                                         threshold=0.3)
        self.assertLessEqual(len(got), 2)
        self.assertTrue(all(s["score"] >= 0.3 for s in got))

    def test_query_documents_rejects_loudly(self):
        import backend
        import config
        with self.assertRaises(ValueError):
            backend.query_documents("What is the lock-in period?",
                                    config=config.load_config(),
                                    vector_store=ScopeStore(),
                                    session_id="not-a-session")

    def test_query_documents_scoped(self):
        import backend
        import config

        class FakeLLM:
            def generate(self, context, question, prompt_template):
                return "ANSWER"

            def is_reachable(self, timeout=3.0):
                return True

            @property
            def describe(self):
                return "Fake"

        store = ScopeStore()
        store.add("lease lock-in period alpha", "lease.pdf",
                  "persistent", None)
        ans, sources = backend.query_documents(
            "What is the lock-in period?",
            config=config.load_config(), vector_store=store,
            llm_provider=FakeLLM(), session_id=SID_A)
        self.assertEqual(ans, "ANSWER")
        self.assertTrue(sources)


class TestWorkflowsScope(unittest.TestCase):
    def test_retrieve_for_document_scoped(self):
        import config
        from workflows import retrieve_for_document
        store = ScopeStore()
        store.add("lease lock-in alpha", "lease.pdf", "persistent", None)
        store.add("session alpha beta", "sess.pdf", "session", SID_A)
        store.add("session bravo gamma", "sess.pdf", "session", SID_B)
        ev, _ms = retrieve_for_document("session terms", "sess.pdf",
                                        config=config.load_config(),
                                        vector_store=store, session_id=SID_A)
        self.assertTrue(ev)
        self.assertNotIn("bravo", " ".join(s["content"] for s in ev))
        ev_none, _ms = retrieve_for_document(
            "session terms", "sess.pdf", config=config.load_config(),
            vector_store=store)
        self.assertEqual(ev_none, [])

    def test_known_files_compat(self):
        from workflows import known_file_names

        class LegacyStore:
            def list_sources(self, limit=50):
                return [{"file_name": "a.pdf"}]

        self.assertEqual(known_file_names(LegacyStore()), ["a.pdf"])
        with self.assertRaises(ValueError):
            known_file_names(LegacyStore(), session_id=SID_A)


class TestDeterministicIds(unittest.TestCase):
    def test_chunk_ids_stable(self):
        from vector_store import ChromaVectorStore
        self.assertEqual(ChromaVectorStore._chunk_ids([lease_chunk("cx")]),
                         ChromaVectorStore._chunk_ids([lease_chunk("cx")]))

    def test_session_ids_unique_opaque(self):
        from document_scope import new_session_id
        self.assertEqual(len({new_session_id() for _ in range(50)}), 50)


class TestDetachSemantics(unittest.TestCase):
    def test_new_chat_is_detach_only(self):
        from vector_store import ChromaVectorStore
        tmp = tempfile.mkdtemp(prefix="detach5b_")
        self.addCleanup(shutil.rmtree, tmp, True)
        vs = ChromaVectorStore(
            persist_directory=os.path.join(tmp, "db"),
            embedding_function=DummyEmbeddings())
        vs.build_index([lease_chunk()])
        vs.build_index([session_chunk()], session_id=SID_A)
        before = vs.get_status()["chunk_count"]
        # Simulated New Chat: drop the binding, delete nothing.
        bound = [d.metadata.get("chunk_id") for d, _ in
                 vs.search("session terms", k=10, session_id=SID_A)]
        self.assertIn("c2", bound)
        unbound = [d.metadata.get("chunk_id") for d, _ in
                   vs.search("session terms", k=10)]
        self.assertNotIn("c2", unbound)
        self.assertEqual(vs.get_status()["chunk_count"], before)


if __name__ == "__main__":
    unittest.main()
