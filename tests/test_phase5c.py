"""Phase 5C tests: session uploads (fakes/in-memory; S3/RDS opt-in only).

Covers validation, identity, quotas, ingest ordering, isolation,
detach, retry, telemetry hygiene, restore compat, and benchmarks.
"""

import os
import sys
import unittest
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SID_A = "a" * 32
SID_B = "b" * 32


def _pdf_bytes(text="Session memo terms alpha beta gamma."):
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


def _encrypted_pdf_bytes():
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(612, 792)
    w.encrypt("pw")
    buf = BytesIO()
    w.write(buf)
    return buf.getvalue()


class FakeStorage:
    """In-memory DocumentStorage (records deletes for cleanup checks)."""

    def __init__(self):
        self.objects = {}
        self.deleted = []

    def list_documents(self, pattern="**/*.pdf"):
        return sorted(self.objects)

    def get_document(self, path):
        return self.objects[path]

    def save_document(self, filename, data):
        self.objects[filename] = bytes(data)
        return filename

    def delete_document(self, path):
        self.deleted.append(path)
        self.objects.pop(path, None)


class FakeVector:
    """Scope-aware in-memory vector store with failure injection."""

    def __init__(self):
        self.rows = []  # dicts: content/file/scope/sid/chunk_id/page
        self.fail_build = False

    def build_index(self, chunks, session_id=None):
        from document_scope import resolve_chunk_scope
        if self.fail_build:
            raise RuntimeError("vector backend down")
        n = 0
        for c in chunks:
            scope, sid = resolve_chunk_scope(
                getattr(c, "metadata", {}) or {}, session_id)
            meta = dict(getattr(c, "metadata", {}) or {})
            meta["scope"] = scope
            if sid is None:
                meta.pop("session_id", None)
            else:
                meta["session_id"] = sid
            cid = meta.get("chunk_id") or f"c{len(self.rows)}"
            if any(r["chunk_id"] == cid for r in self.rows):
                continue  # idempotent upsert
            self.rows.append({"content": c.page_content,
                              "file": meta.get("file_name"),
                              "scope": scope, "sid": sid,
                              "chunk_id": cid,
                              "page": meta.get("page"),
                              "document_id": meta.get("document_id")})
            n += 1
        return n

    def _visible(self, r, session_id):
        if r["scope"] == "session":
            return r["sid"] is not None and r["sid"] == session_id
        return True

    def _doc(self, r):
        class D:
            pass
        d = D()
        d.page_content = r["content"]
        d.metadata = {"file_name": r["file"], "page": r["page"],
                      "score": 0.9, "source": "/t/" + str(r["file"]),
                      "source_path": "/t/" + str(r["file"]),
                      "document_id": r["document_id"],
                      "chunk_id": r["chunk_id"]}
        d.id = None
        return d

    def search(self, query, k=5, session_id=None):
        from document_scope import validate_session_id
        if session_id is not None:
            validate_session_id(session_id)
        return [(self._doc(r), 0.9) for r in self.rows
                if self._visible(r, session_id)][:k]

    def chunks_for_source(self, file_name, limit=8, session_id=None):
        return [(self._doc(r), None) for r in self.rows
                if r["file"] == file_name
                and self._visible(r, session_id)][:limit]

    def list_sources(self, limit=50, session_id=None):
        seen = {}
        for r in self.rows:
            if self._visible(r, session_id) and r["file"] not in seen:
                seen[r["file"]] = r["document_id"]
        return [{"file_name": f, "document_id": seen[f]}
                for f in list(seen)[:limit]]


def _cfg(**over):
    import config
    base = dict(document_storage="local")
    base.update(over)
    return config.AppConfig(**base)


def ingest(data, name, sid, **kw):
    from session_uploads import ingest_session_upload
    kw.setdefault("storage", FakeStorage())
    kw.setdefault("vector_store", FakeVector())
    kw.setdefault("registry", [])
    kw.setdefault("config", _cfg())
    return ingest_session_upload(data, name, sid, **kw)


class TestValidation(unittest.TestCase):
    def test_valid_pdf(self):
        from session_uploads import validate_upload
        ok, reason = validate_upload(_pdf_bytes(), "memo.pdf")
        self.assertTrue(ok, reason)

    def test_txt_name_with_pdf_bytes_accepted(self):
        # Magic bytes rule; browser names are labels only.
        from session_uploads import validate_upload
        ok, _reason = validate_upload(_pdf_bytes(), "notes.txt")
        self.assertTrue(ok)

    def test_non_pdf_rejected(self):
        from session_uploads import validate_upload
        ok, reason = validate_upload(b"hello world, not a pdf", "x.pdf")
        self.assertFalse(ok)
        self.assertIn("PDF", reason)

    def test_empty_rejected(self):
        from session_uploads import validate_upload
        ok, _reason = validate_upload(b"", "x.pdf")
        self.assertFalse(ok)

    def test_oversize_rejected_before_processing(self):
        from session_uploads import MAX_FILE_BYTES, validate_upload
        big = b"%PDF-" + b"\x00" * MAX_FILE_BYTES
        ok, reason = validate_upload(big, "big.pdf")
        self.assertFalse(ok)
        self.assertIn("10 MB", reason)

    def test_corrupt_rejected(self):
        from session_uploads import validate_upload
        ok, reason = validate_upload(b"%PDF-1.4\nbroken\xff\xff", "c.pdf")
        self.assertFalse(ok)
        self.assertTrue("corrupt" in reason or "PDF" in reason)

    def test_encrypted_rejected(self):
        from session_uploads import validate_upload
        ok, reason = validate_upload(_encrypted_pdf_bytes(), "locked.pdf")
        self.assertFalse(ok)
        self.assertIn("ncrypted", reason)


class TestIdentity(unittest.TestCase):
    def test_sanitize_and_keys(self):
        from session_uploads import (sanitize_filename, session_object_key,
                                     session_prefix)
        self.assertEqual(sanitize_filename("../evil.pdf"), "evil.pdf")
        self.assertEqual(sanitize_filename("a/b\\c.pdf"), "c.pdf")
        self.assertEqual(session_prefix(SID_A),
                         f"sessions/{SID_A}/uploads/")
        self.assertTrue(session_object_key(SID_A, "x.pdf").startswith(
            f"sessions/{SID_A}/uploads/"))
        with self.assertRaises(ValueError):
            session_prefix("bogus")

    def test_document_id_semantics(self):
        from session_uploads import content_sha256_of, session_document_id
        data = _pdf_bytes()
        sha = content_sha256_of(data)
        a1 = session_document_id(SID_A, sha, "x.pdf")
        a2 = session_document_id(SID_A, sha, "x.pdf")
        b = session_document_id(SID_B, sha, "x.pdf")
        c = session_document_id(SID_A, content_sha256_of(_pdf_bytes("other")),
                                "x.pdf")
        self.assertEqual(a1, a2)       # deterministic: retry-safe
        self.assertNotEqual(a1, b)     # sessions never collide
        self.assertNotEqual(a1, c)     # content change forks, never merges
        self.assertNotIn("x.pdf", a1)  # no raw names inside identity

    def test_collision_resistant_names(self):
        from session_uploads import content_sha256_of
        r1 = ingest(_pdf_bytes("version one text here"), "same.pdf", SID_A)
        r2 = ingest(_pdf_bytes("version two text here"), "same.pdf", SID_A)
        self.assertEqual(r1["status"], "ready")
        self.assertEqual(r2["status"], "ready")
        self.assertNotEqual(r1["safe_name"], r2["safe_name"])
        self.assertNotEqual(r1["document_id"], r2["document_id"])


class TestQuota(unittest.TestCase):
    def test_file_count_cap(self):
        from session_uploads import MAX_FILES_PER_SESSION, check_session_quota
        reg = [{"status": "ready", "size_bytes": 10, "object_stored": True,
                "document_id": f"d{i}"} for i in range(MAX_FILES_PER_SESSION)]
        ok, reason = check_session_quota(reg, 10)
        self.assertFalse(ok)
        self.assertIn(str(MAX_FILES_PER_SESSION), reason)

    def test_byte_cap(self):
        from session_uploads import MAX_SESSION_BYTES, check_session_quota
        reg = [{"status": "ready", "size_bytes": MAX_SESSION_BYTES,
                "object_stored": True, "document_id": "d0"}]
        ok, _reason = check_session_quota(reg, 10)
        self.assertFalse(ok)

    def test_conservative_unknowns(self):
        from session_uploads import MAX_FILE_BYTES, check_session_quota
        # Vector rows the registry never saw count at max-file weight.
        ok, _reason = check_session_quota([], 10, vector_doc_ids=["x"])
        self.assertTrue(ok)
        many = [f"d{i}" for i in range(6)]
        ok, _reason = check_session_quota([], 10, vector_doc_ids=many)
        self.assertFalse(ok)
        self.assertGreater(MAX_FILE_BYTES, 0)

    def test_failed_cleaned_not_counted(self):
        from session_uploads import check_session_quota
        reg = [{"status": "failed", "size_bytes": 999999,
                "object_stored": False}]
        ok, _reason = check_session_quota(reg, 10)
        self.assertTrue(ok)

    def test_ingest_enforces_counts(self):
        from session_uploads import MAX_FILES_PER_SESSION
        store, vs, reg = FakeStorage(), FakeVector(), []
        for i in range(MAX_FILES_PER_SESSION):
            r = ingest(_pdf_bytes(f"doc number {i} text here"), f"d{i}.pdf",
                       SID_A, storage=store, vector_store=vs, registry=reg)
            self.assertEqual(r["status"], "ready")
        r = ingest(_pdf_bytes("one more doc text here"), "extra.pdf",
                   SID_A, storage=store, vector_store=vs, registry=reg)
        self.assertEqual(r["status"], "failed")
        self.assertIn(str(MAX_FILES_PER_SESSION), r["error"])


class TestSessionStores(unittest.TestCase):
    def test_s3_session_prefix(self):
        from session_uploads import get_session_store

        class FakeClient:
            pass

        store = get_session_store(
            _cfg(document_storage="s3", s3_bucket="bkt"), SID_A,
            client=FakeClient())
        self.assertEqual(store.prefix, f"sessions/{SID_A}/uploads/")
        self.assertEqual(store.bucket, "bkt")
        self.assertEqual(
            store.key_for("x-abc.pdf"),
            f"sessions/{SID_A}/uploads/x-abc.pdf")
        self.assertNotIn("documents/", store.key_for("x-abc.pdf"))

    def test_s3_requires_bucket_and_sid(self):
        from session_uploads import get_session_store
        with self.assertRaises(ValueError):
            get_session_store(_cfg(document_storage="s3"), SID_A)
        with self.assertRaises(ValueError):
            get_session_store(_cfg(), "bogus")

    def test_local_session_dir(self):
        import tempfile
        from session_uploads import get_session_store
        tmp = tempfile.mkdtemp(prefix="sessroot_")
        try:
            store = get_session_store(
                _cfg(session_storage_dir=tmp), SID_A)
            dest = store.save_document("x.pdf", b"%PDF-1.4 hi")
            self.assertIn(SID_A, dest)
            self.assertNotIn("..", dest)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


class TestIngestFlow(unittest.TestCase):
    def test_upload_index_retrieve(self):
        import backend
        store, vs, reg = FakeStorage(), FakeVector(), []
        r = ingest(_pdf_bytes(), "memo.pdf", SID_A,
                   storage=store, vector_store=vs, registry=reg)
        self.assertEqual(r["status"], "ready")
        self.assertGreater(r["chunks_indexed"], 0)
        self.assertEqual(set(r), {"status", "display_name", "safe_name",
                                 "size_bytes", "content_sha256",
                                 "document_id", "chunks_indexed", "error",
                                 "object_stored"})
        got = backend.retrieve_documents(
            "memo terms alpha", config=_cfg(), vector_store=vs,
            session_id=SID_A)
        self.assertTrue(any(s["file_name"] == "memo.pdf" for s in got))
        # Scoped vector metadata on every row.
        for row in vs.rows:
            self.assertEqual(row["scope"], "session")
            self.assertEqual(row["sid"], SID_A)
        # Unbound retrieval sees nothing of the session doc.
        got_none = backend.retrieve_documents(
            "memo terms alpha", config=_cfg(), vector_store=vs)
        self.assertFalse(any(s["file_name"] == "memo.pdf"
                             for s in got_none))

    def test_persistent_plus_session(self):
        import backend
        store, vs, reg = FakeStorage(), FakeVector(), []
        vs.rows.append({"content": "lease lock-in alpha", "file": "lease.pdf",
                        "scope": "persistent", "sid": None, "chunk_id": "p1",
                        "page": 0, "document_id": "dp"})
        ingest(_pdf_bytes(), "memo.pdf", SID_A,
               storage=store, vector_store=vs, registry=reg)
        got = backend.retrieve_documents(
            "lease lock-in memo alpha", config=_cfg(), vector_store=vs,
            session_id=SID_A)
        files = {s["file_name"] for s in got}
        self.assertIn("lease.pdf", files)
        self.assertIn("memo.pdf", files)

    def test_cross_session_zero(self):
        import backend
        store, vs, reg = FakeStorage(), FakeVector(), []
        ingest(_pdf_bytes(), "memo.pdf", SID_A,
               storage=store, vector_store=vs, registry=reg)
        got = backend.retrieve_documents(
            "memo terms alpha", config=_cfg(), vector_store=vs,
            session_id=SID_B)
        self.assertFalse(any(s["file_name"] == "memo.pdf" for s in got))

    def test_duplicate_shortcut_no_dupes(self):
        store, vs, reg = FakeStorage(), FakeVector(), []
        data = _pdf_bytes()
        r1 = ingest(data, "memo.pdf", SID_A,
                    storage=store, vector_store=vs, registry=reg)
        n = len(vs.rows)
        r2 = ingest(data, "memo.pdf", SID_A,
                    storage=store, vector_store=vs, registry=reg)
        self.assertEqual(r1["status"], "ready")
        self.assertEqual(r2["status"], "already-indexed")
        self.assertEqual(len(vs.rows), n)

    def test_partial_failure_cleanup_and_retry(self):
        store, vs, reg = FakeStorage(), FakeVector(), []
        vs.fail_build = True
        r = ingest(_pdf_bytes(), "memo.pdf", SID_A,
                   storage=store, vector_store=vs, registry=reg)
        self.assertEqual(r["status"], "failed")
        self.assertIn("memo.pdf", r["display_name"])
        # Staged object cleaned; registry says failed (never ready).
        rec = [x for x in reg if x.get("document_id") == r["document_id"]]
        self.assertTrue(rec and rec[0]["status"] == "failed")
        vs.fail_build = False
        r2 = ingest(_pdf_bytes(), "memo.pdf", SID_A,
                    storage=store, vector_store=vs, registry=reg)
        self.assertEqual(r2["status"], "ready")
        # No uncontrolled duplicates after retry.
        self.assertEqual(len(vs.rows), r2["chunks_indexed"])

    def test_zero_text_not_ready(self):
        from pypdf import PdfWriter
        w = PdfWriter()
        w.add_blank_page(612, 792)
        buf = BytesIO()
        w.write(buf)
        store, vs, reg = FakeStorage(), FakeVector(), []
        r = ingest(buf.getvalue(), "blank.pdf", SID_A,
                   storage=store, vector_store=vs, registry=reg)
        self.assertEqual(r["status"], "failed")
        self.assertIn("extractable", r["error"])


class TestLifecycle(unittest.TestCase):
    def test_ensure_and_rotate(self):
        from session_uploads import ensure_session_id, rotate_session_id
        state = {}
        first = ensure_session_id(state)
        self.assertEqual(ensure_session_id(state), first)
        state["session_docs"] = [{"status": "ready"}]
        second = rotate_session_id(state)
        self.assertNotEqual(first, second)
        self.assertEqual(state["session_docs"], [])

    def test_ensure_resets_on_damage(self):
        from session_uploads import ensure_session_id
        state = {"session_id": "bogus",
                 "session_docs": [{"status": "ready"}]}
        fresh = ensure_session_id(state)
        self.assertNotEqual(fresh, "bogus")
        self.assertEqual(state["session_docs"], [])

    def test_new_chat_detach_keeps_vectors(self):
        import backend
        from session_uploads import rotate_session_id
        store, vs, reg = FakeStorage(), FakeVector(), []
        ingest(_pdf_bytes(), "memo.pdf", SID_A,
               storage=store, vector_store=vs, registry=reg)
        before = len(vs.rows)
        state = {"session_id": SID_A, "session_docs": reg}
        rotate_session_id(state)  # New Chat: detach, never delete
        self.assertEqual(len(vs.rows), before)
        got = backend.retrieve_documents(
            "memo terms alpha", config=_cfg(), vector_store=vs,
            session_id=state["session_id"])
        self.assertFalse(any(s["file_name"] == "memo.pdf" for s in got))


class TestTelemetryHygiene(unittest.TestCase):
    def test_result_has_no_contents_or_secrets(self):
        store, vs, reg = FakeStorage(), FakeVector(), []
        r = ingest(_pdf_bytes("secret alpha content here"), "memo.pdf",
                   SID_A, storage=store, vector_store=vs, registry=reg)
        blob = str(r).lower()
        self.assertNotIn("secret alpha content", blob)
        for key in ("content", "page_content", "password", "api_key"):
            self.assertNotIn(key, r)

    def test_registry_has_no_contents(self):
        store, vs, reg = FakeStorage(), FakeVector(), []
        ingest(_pdf_bytes(), "m.pdf", SID_A,
               storage=store, vector_store=vs, registry=reg)
        scrubbed = str(reg).replace("content_sha256", "")
        self.assertNotIn("content", scrubbed)
        self.assertNotIn("alpha beta", str(reg))


class TestRestoreCompat(unittest.TestCase):
    def test_session_block_round_trip(self):
        from session_restore import (deserialize_conversation,
                                     restore_session_into_state,
                                     serialize_conversation)
        snap = serialize_conversation(
            [{"role": "user", "content": "hi", "seq": 1}],
            session={"id": SID_A, "documents": [
                {"display_name": "m.pdf", "safe_name": "m-abc.pdf",
                 "size_bytes": 10, "content_sha256": "s",
                 "document_id": "d", "status": "ready",
                 "chunks_indexed": 2, "error": ""}]})
        back = deserialize_conversation(snap)
        self.assertTrue(back["ok"])
        state = {}
        self.assertTrue(restore_session_into_state(state, back))
        self.assertEqual(state["session_id"], SID_A)
        self.assertEqual(len(state["session_docs"]), 1)

    def test_old_snapshot_without_session(self):
        from session_restore import (deserialize_conversation,
                                     restore_session_into_state)
        back = deserialize_conversation({"messages": []})
        self.assertTrue(back["ok"])
        self.assertNotIn("session", back)
        self.assertFalse(restore_session_into_state({}, back))

    def test_invalid_session_dropped(self):
        from session_restore import deserialize_conversation
        back = deserialize_conversation(
            {"messages": [], "session": {"id": "bogus", "documents": []}})
        self.assertTrue(back["ok"])
        self.assertNotIn("session", back)


class TestBenchmarkSessions(unittest.TestCase):
    def test_session_cases_run_live(self):
        from tests.benchmarks.harness import run_all
        out = run_all()
        by_id = {r["id"]: r for r in out["results"]}
        self.assertEqual(by_id["session-upload-basic"]["status"], "pass")
        self.assertEqual(by_id["session-upload-isolation"]["status"], "pass")
        self.assertEqual(out["summary"]["skipped"], 0)


if __name__ == "__main__":
    unittest.main()
