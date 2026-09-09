"""Phase 4.12 tests: comparison, extraction, summaries, page navigation.

Deterministic unit tests (fakes, no LLM server) + focused AppTest
coverage. No secrets. Existing 4.10/4.11 behaviour is regression-checked
by the untouched suite plus explicit delegation tests here.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class Doc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})
        self.id = None


def _struct(content, fname, page, score, cid):
    return {"content": content, "file_name": fname, "page": page,
            "score": score, "source": "/t/" + fname,
            "source_path": "/t/" + fname, "document_id": "d-" + fname,
            "chunk_id": cid}


LEASE = "The lock-in period in the lease deed is 36 months. Early termination requires 3 months notice."
CONTRACT = "The service contract renews annually with 30 days notice. Payment of Rs 50000 is due monthly."


class FakeStore:
    """Threshold-simulating store over two fixed documents."""
    FILES = ["contract.pdf", "lease.pdf"]

    def __init__(self, scores=None):
        # (fname -> score) for search hits; None = no hit.
        self.scores = scores if scores is not None else {
            "lease.pdf": 0.9, "contract.pdf": 0.85}
        self.limits = []

    def _doc(self, fname):
        content = LEASE if fname == "lease.pdf" else CONTRACT
        return Doc(content, {"source_path": "/t/" + fname,
                             "source": "/t/" + fname, "file_name": fname,
                             "page": 0, "document_id": "d-" + fname,
                             "chunk_id": "c-" + fname})

    def search(self, query, k=5):
        q = (query or "").lower()
        out = []
        for fname in self.FILES:
            doc = self._doc(fname)
            words = [w for w in q.split() if len(w) >= 4]
            if any(w in doc.page_content.lower() for w in words):
                score = self.scores.get(fname)
                if score is not None:
                    out.append((doc, score))
        return out[:k]

    def chunks_for_source(self, file_name, limit=8):
        self.limits.append(limit)
        if file_name not in self.FILES:
            return []
        return [(self._doc(file_name), None)]

    def list_sources(self, limit=50):
        return [{"file_name": f, "document_id": "d-" + f}
                for f in self.FILES]

    def get_status(self):
        return {"ready": True}


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, context, question, prompt_template):
        self.calls.append((context, question, prompt_template))
        return "SYNTHESIZED lease contract months notice."

    def generate_stream(self, context, question, prompt_template):
        yield self.generate(context, question, prompt_template)

    def is_reachable(self, timeout=3.0):
        return True

    @property
    def describe(self):
        return "Fake"


def _cfg():
    import config
    return config.load_config()


class TestDetection(unittest.TestCase):
    def test_comparison_requests(self):
        from workflows import COMPARISON, detect_workflow
        for q in ["Compare the lock-in clauses in contract.pdf and lease.pdf.",
                  "How do the payment terms differ between these two documents?",
                  "Are the notice periods the same?"]:
            with self.subTest(q=q):
                self.assertEqual(detect_workflow(q)["workflow"], COMPARISON)

    def test_extraction_requests(self):
        from workflows import EXTRACTION, detect_workflow
        for q in ["List all payment deadlines.",
                  "Extract all notice periods.",
                  "List dates, deadlines, renewal periods and termination notice."]:
            with self.subTest(q=q):
                det = detect_workflow(q)
                self.assertEqual(det["workflow"], EXTRACTION)
                self.assertTrue(det["fields"])

    def test_summary_requests(self):
        from workflows import SUMMARY, detect_workflow
        for q in ["Summarize lease.pdf.",
                  "Give me an overview of contract.pdf."]:
            with self.subTest(q=q):
                det = detect_workflow(q)
                self.assertEqual(det["workflow"], SUMMARY)
                self.assertEqual(len(det["mentioned"]), 1)

    def test_normal_questions_unchanged(self):
        from workflows import NORMAL, detect_workflow
        for q in ["What is the lock-in period in the lease deed?",
                  "What does lease.pdf say about renewal?",
                  "Tell me more about lease.pdf.",
                  "Hi", "What can you do?",
                  "List your abilities"]:
            with self.subTest(q=q):
                self.assertEqual(detect_workflow(q)["workflow"], NORMAL)

    def test_no_llm_in_detection(self):
        import workflows
        self.assertNotIn("generate", workflows.detect_workflow.__code__.co_names)


class TestComparisonTargets(unittest.TestCase):
    def test_two_named_files(self):
        from workflows import resolve_comparison_targets
        r = resolve_comparison_targets(["contract.pdf", "lease.pdf"],
                                       ["contract.pdf", "lease.pdf"])
        self.assertEqual(r["targets"], ["contract.pdf", "lease.pdf"])
        self.assertIsNone(r["clarification"])

    def test_unknown_file_clarifies(self):
        from workflows import resolve_comparison_targets
        r = resolve_comparison_targets(["missing.pdf"],
                                       ["contract.pdf", "lease.pdf"])
        self.assertIsNone(r["targets"])
        self.assertIn("missing.pdf", r["clarification"])
        self.assertIn("contract.pdf", r["clarification"])

    def test_ambiguous_clarifies(self):
        from workflows import resolve_comparison_targets
        r = resolve_comparison_targets([], ["a.pdf", "b.pdf", "c.pdf"])
        self.assertIsNone(r["targets"])
        self.assertIn("To proceed", r["clarification"])

    def test_bare_same_with_two_docs(self):
        from workflows import (resolve_comparison_targets,
                               should_auto_resolve_bare_comparison)
        known = ["contract.pdf", "lease.pdf"]
        self.assertTrue(should_auto_resolve_bare_comparison(
            "Are the notice periods the same?", known))
        r = resolve_comparison_targets([], known)
        self.assertEqual(r["targets"], known)


class TestComparisonRun(unittest.TestCase):
    def test_two_document_comparison(self):
        from workflows import COMPARISON_LABEL, query_workflow
        ans, sources, info = query_workflow(
            "Compare the lock-in clauses in contract.pdf and lease.pdf.",
            config=_cfg(), vector_store=FakeStore(), llm_provider=FakeLLM())
        self.assertEqual(info["workflow"], "comparison")
        self.assertEqual(info["label"], COMPARISON_LABEL)
        files = {s["file_name"] for s in sources}
        self.assertEqual(files, {"contract.pdf", "lease.pdf"})
        self.assertIn("SYNTHESIZED", ans)

    def test_per_document_scoping(self):
        from workflows import query_workflow
        store = FakeStore()
        _ans, sources, _info = query_workflow(
            "Compare the lock-in clauses in contract.pdf and lease.pdf.",
            config=_cfg(), vector_store=store, llm_provider=FakeLLM())
        # Same-topic evidence kept per document (no cross-contamination).
        for s in sources:
            self.assertIn(s["file_name"], ("contract.pdf", "lease.pdf"))
            self.assertIsNotNone(s["chunk_id"])

    def test_partial_one_side_missing(self):
        from workflows import PARTIAL_COMPARISON_LABEL, query_workflow
        store = FakeStore(scores={"lease.pdf": 0.9, "contract.pdf": None})
        llm = FakeLLM()
        ans, sources, info = query_workflow(
            "Compare the lock-in clauses in contract.pdf and lease.pdf.",
            config=_cfg(), vector_store=store, llm_provider=llm)
        self.assertEqual(info["label"], PARTIAL_COMPARISON_LABEL)
        self.assertIn("incomplete", ans.lower())
        # Contract side has only file-browse (unscored) evidence: no
        # query-matched chunk, nothing invented about it.
        scored = [s for s in sources
                  if s.get("file_name") == "contract.pdf"
                  and isinstance(s.get("score"), (int, float))]
        self.assertEqual(scored, [])
        self.assertTrue(any(s["file_name"] == "lease.pdf"
                            and isinstance(s.get("score"), (int, float))
                            for s in sources))

    def test_ambiguous_target_clarifies_no_llm(self):
        from workflows import query_workflow
        store = FakeStore()
        store.FILES = ["a.pdf", "b.pdf", "c.pdf"]
        llm = FakeLLM()
        ans, sources, info = query_workflow(
            "Compare the notice periods.", config=_cfg(),
            vector_store=store, llm_provider=llm)
        self.assertEqual(sources, [])
        self.assertIn("To proceed", ans)
        self.assertEqual(llm.calls, [])

    def test_no_favorability_claim(self):
        from workflows import COMPARISON_PROMPT_TEMPLATE
        self.assertIn("more favorable", COMPARISON_PROMPT_TEMPLATE)
        self.assertIn("Never claim", COMPARISON_PROMPT_TEMPLATE)

    def test_stream_matches_nonstream(self):
        from workflows import query_workflow, stream_workflow_answer
        llm = FakeLLM()
        ans, _src, _info = query_workflow(
            "Compare the lock-in clauses in contract.pdf and lease.pdf.",
            config=_cfg(), vector_store=FakeStore(), llm_provider=FakeLLM())
        info, stream = stream_workflow_answer(
            "Compare the lock-in clauses in contract.pdf and lease.pdf.",
            config=_cfg(), vector_store=FakeStore(), llm_provider=llm)
        streamed = "".join(stream)
        self.assertIn("SYNTHESIZED", streamed)
        self.assertEqual(info["label"], _info["label"])


class TestExtraction(unittest.TestCase):
    def test_notice_periods_table(self):
        from workflows import EXTRACTION_LABEL, query_workflow
        ans, sources, info = query_workflow(
            "Extract all notice periods.", config=_cfg(),
            vector_store=FakeStore(), llm_provider=FakeLLM())
        self.assertEqual(info["label"], EXTRACTION_LABEL)
        self.assertIn("|", ans)  # markdown table
        self.assertIn("notice", ans.lower())
        self.assertTrue(sources)

    def test_values_grounded(self):
        from workflows import query_workflow
        ans, sources, _info = query_workflow(
            "List all payment deadlines.", config=_cfg(),
            vector_store=FakeStore(), llm_provider=FakeLLM())
        evidence = " ".join(s.get("content", "") for s in sources).lower()
        for line in ans.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 3 or cells[0] in ("#", "---") \
                    or cells[2].startswith("---"):
                continue
            self.assertIn(cells[2].lower()[:40], evidence)

    def test_missing_values_marked(self):
        from workflows import harvest_extraction_rows
        rows, missing = harvest_extraction_rows(
            [_struct("The deposit amount is Rs 50000.", "a.pdf", 0, 0.9, "c1")],
            ["amounts", "notice periods"])
        self.assertTrue(rows)
        self.assertIn("notice periods", missing)

    def test_duplicates_removed(self):
        from workflows import harvest_extraction_rows
        dup = _struct("Payment of Rs 50000 is due monthly.", "a.pdf", 0,
                      0.9, "c1")
        rows, _missing = harvest_extraction_rows([dup, dict(dup)],
                                                 ["payment terms"])
        self.assertEqual(len(rows), 1)

    def test_no_llm_for_extraction(self):
        from workflows import query_workflow
        llm = FakeLLM()
        query_workflow("List all payment deadlines.", config=_cfg(),
                       vector_store=FakeStore(), llm_provider=llm)
        self.assertEqual(llm.calls, [])

    def test_normal_routing_unchanged(self):
        from unittest import mock
        import backend
        import workflows
        with mock.patch.object(backend, "stream_answer") as m:
            m.return_value = ({"answer": "x", "retrieved": [],
                               "needs_retrieval": False, "support_level": None,
                               "failed": False, "timings": {}}, iter(["x"]))
            workflows.stream_workflow_answer("What is the lock-in period?")
            self.assertTrue(m.called)


class TestSummary(unittest.TestCase):
    def test_explicit_summary_targets_file(self):
        from workflows import SUMMARY_LABEL, query_workflow
        ans, sources, info = query_workflow(
            "Summarize lease.pdf.", config=_cfg(),
            vector_store=FakeStore(), llm_provider=FakeLLM())
        self.assertEqual(info["label"], SUMMARY_LABEL)
        self.assertTrue(all(s["file_name"] == "lease.pdf" for s in sources))
        self.assertIn("SYNTHESIZED", ans)

    def test_topical_question_stays_normal(self):
        from workflows import NORMAL, detect_workflow
        self.assertEqual(
            detect_workflow("What does lease.pdf say about renewal?")["workflow"],
            NORMAL)

    def test_missing_document_no_llm(self):
        from workflows import query_workflow
        llm = FakeLLM()
        ans, sources, _info = query_workflow(
            "Summarize ghost.pdf.", config=_cfg(),
            vector_store=FakeStore(), llm_provider=llm)
        self.assertEqual(sources, [])
        self.assertIn("ghost.pdf", ans)
        self.assertEqual(llm.calls, [])

    def test_chunk_budget_bounded(self):
        from workflows import MAX_SUMMARY_CHUNKS, summary_chunks

        class BigStore(FakeStore):
            def chunks_for_source(self, file_name, limit=8):
                self.limits.append(limit)
                return super().chunks_for_source(
                    file_name, limit=min(limit, MAX_SUMMARY_CHUNKS))

        store = BigStore()
        chunks = summary_chunks("lease.pdf", store)
        self.assertLessEqual(len(chunks), MAX_SUMMARY_CHUNKS)
        self.assertTrue(all(lim <= MAX_SUMMARY_CHUNKS for lim in store.limits))

    def test_staged_large_document(self):
        from workflows import SUMMARY_GROUP_SIZE, query_workflow

        class ManyStore(FakeStore):
            def chunks_for_source(self, file_name, limit=8):
                base = super().chunks_for_source(file_name, limit=8)
                if file_name != "lease.pdf":
                    return base
                out = []
                for i in range(SUMMARY_GROUP_SIZE + 2):
                    d = Doc(f"Clause {i}: the lock-in period term {i} months.",
                            {"source_path": "/t/lease.pdf", "source": "/t/lease.pdf",
                             "file_name": "lease.pdf", "page": i,
                             "document_id": "d-lease.pdf",
                             "chunk_id": f"c-big-{i}"})
                    out.append((d, None))
                return out

        llm = FakeLLM()
        ans, sources, info = query_workflow(
            "Summarize lease.pdf.", config=_cfg(),
            vector_store=ManyStore(), llm_provider=llm)
        # Staged: >1 part call + 1 final call, sequential same provider.
        self.assertGreaterEqual(len(llm.calls), 2)
        self.assertLessEqual(len(sources), 24)
        self.assertIn("SYNTHESIZED", ans)

    def test_new_prompts_leave_old_ones_alone(self):
        import backend
        import workflows
        self.assertNotEqual(workflows.SUMMARY_PROMPT_TEMPLATE,
                            backend.PROMPT_TEMPLATE)
        self.assertNotEqual(workflows.COMPARISON_PROMPT_TEMPLATE,
                            backend.PROMPT_TEMPLATE)


class TestNavigation(unittest.TestCase):
    def test_sibling_pages(self):
        from ui.components import sibling_pages
        sources = [_struct("a", "x.pdf", 0, 0.9, "c1"),
                   _struct("b", "x.pdf", 2, 0.8, "c2"),
                   _struct("c", "y.pdf", 0, 0.7, "c3")]
        self.assertEqual(sibling_pages(sources, "x.pdf", exclude_page=0), [2])
        self.assertEqual(sibling_pages(sources, "y.pdf"), [0])

    def test_full_html_no_path_leak(self):
        from ui.components import full_source_html
        row = {"file_name": "lease.pdf", "page": 1, "score_text": "0.84",
               "content": "The lock-in period is 36 months. " * 40}
        html = full_source_html(row)
        self.assertIn("36 months", html)
        self.assertNotIn("/t/", html)
        self.assertNotIn("source_path", html)

    def test_full_html_escapes(self):
        from ui.components import full_source_html
        row = {"file_name": "<img src=x>.pdf", "page": 0,
               "score_text": "0.50",
               "content": "<script>alert(1)</script> terms"}
        html = full_source_html(row)
        self.assertNotIn("<script>alert", html)
        self.assertNotIn("<img src=x", html)
        self.assertIn("terms", html)


class TestInvariants412(unittest.TestCase):
    def test_core_untouched(self):
        import backend
        import config
        cfg = config.load_config()
        self.assertEqual(cfg.llm_temperature, 0.0)
        self.assertEqual(cfg.retrieval_k, 5)
        self.assertEqual(cfg.relevance_threshold, 0.3)
        self.assertEqual(cfg.chunk_size, 1000)
        self.assertEqual(cfg.chunk_overlap, 200)
        self.assertEqual(cfg.embedding_model, "all-MiniLM-L6-v2")
        self.assertIn("You are an expert legal assistant", backend.PROMPT_TEMPLATE)

    def test_retrieval_params_preserved(self):
        import inspect
        import backend
        sig = inspect.signature(backend.retrieve_documents)
        self.assertIn("threshold", sig.parameters)
        self.assertIn("k", sig.parameters)

    def test_provider_abstraction_intact(self):
        import backend
        import inspect
        src = inspect.getsource(backend.retrieve_documents)
        self.assertIn("vector_store.search", src)
        import workflows
        wsrc = inspect.getsource(workflows.retrieve_for_document)
        self.assertIn("chunks_for_source", wsrc)
        self.assertNotIn("Chroma", wsrc)
        self.assertNotIn("psycopg", wsrc)


class TestWorkflowsAppTest(unittest.TestCase):
    """Focused AppTest: workflow rendering + viewer + normal regression."""

    APP = os.path.join(os.path.dirname(__file__), "..", "app.py")

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp(prefix="p412_")
        pdfd = os.path.join(cls.tmp, "pdfs")
        os.makedirs(pdfd)

        def _pdf_bytes(text):
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

        with open(os.path.join(pdfd, "lease.pdf"), "wb") as f:
            f.write(_pdf_bytes(
                "The lock-in period in the lease deed is 36 months. "
                "Early termination requires 3 months notice."))
        cls._old = dict(os.environ)
        os.environ["CHROMA_PATH"] = os.path.join(cls.tmp, "chroma")
        os.environ["LLM_BASE_URL"] = "http://127.0.0.1:1/v1"
        import config
        config.reset_config_cache()
        from backend import create_vector_db_from_folder
        ok, _ = create_vector_db_from_folder(pdfd)
        assert ok

    @classmethod
    def tearDownClass(cls):
        import shutil
        os.environ.clear()
        os.environ.update(cls._old)
        import config
        config.reset_config_cache()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_extraction_renders_table_no_llm(self):
        import backend
        from unittest import mock
        from streamlit.testing.v1 import AppTest

        chunks = [_struct(LEASE, "lease.pdf", 0, 0.85, "c1"),
                  _struct(CONTRACT, "lease.pdf", 1, 0.8, "c2")]

        with mock.patch("model_warmup.warmup",
                        return_value={"state": "ready"}), \
             mock.patch.object(backend, "retrieve_documents",
                               return_value=[dict(c) for c in chunks]):
            at = AppTest.from_file(self.APP, default_timeout=180)
            at.run()
            at.chat_input[0].set_value("List all notice periods.")
            at.run()
            at.run()
        self.assertFalse(at.exception, at.exception)
        md = "\n".join(m.value for m in at.markdown)
        caps = "\n".join(c.value for c in at.caption)
        self.assertIn("Structured answer", caps)
        self.assertIn("notice", md.lower())
        self.assertIn("lease.pdf", md)
        labels = [b.label for b in at.button]
        self.assertIn("Open source", labels)

    def test_comparison_renders_per_doc_sources(self):
        import workflows
        from unittest import mock
        from streamlit.testing.v1 import AppTest

        srcs = [_struct(LEASE, "lease.pdf", 0, 0.85, "c1"),
                _struct(CONTRACT, "contract.pdf", 0, 0.8, "c2")]

        def _fake(prompt_text, **kw):
            info = {"answer": None, "retrieved": [dict(s) for s in srcs],
                    "needs_retrieval": True, "support_level": "comparison",
                    "failed": False,
                    "timings": {"retrieval_ms": 5, "support_ms": 1,
                                "preparation_ms": 6, "generation_ms": None},
                    "workflow": "comparison", "label": "Comparison answer",
                    "fallback_template": None, "fallback_question": prompt_text}

            def _gen():
                yield ("| Topic | lease.pdf | contract.pdf |\n"
                       "36 months lease notice contract.")

            return info, _gen()

        with mock.patch("model_warmup.warmup",
                        return_value={"state": "ready"}), \
             mock.patch.object(workflows, "stream_workflow_answer",
                               side_effect=_fake):
            at = AppTest.from_file(self.APP, default_timeout=180)
            at.run()
            at.chat_input[0].set_value(
                "Compare the notice periods in lease.pdf and contract.pdf.")
            at.run()
            at.run()
        self.assertFalse(at.exception, at.exception)
        md = "\n".join(m.value for m in at.markdown)
        caps = "\n".join(c.value for c in at.caption)
        self.assertIn("Comparison answer", caps)
        self.assertIn("lease.pdf", md)
        self.assertIn("contract.pdf", md)

    def test_open_source_viewer(self):
        import backend
        from unittest import mock
        from streamlit.testing.v1 import AppTest

        chunks = [_struct(LEASE + " Extra detail sentence here.", "lease.pdf",
                          0, 0.85, "c1")]

        with mock.patch("model_warmup.warmup",
                        return_value={"state": "ready"}), \
             mock.patch.object(backend, "retrieve_documents",
                               return_value=[dict(c) for c in chunks]):
            at = AppTest.from_file(self.APP, default_timeout=180)
            at.run()
            at.chat_input[0].set_value("List all notice periods.")
            at.run()
            at.run()
            opens = [b for b in at.button if b.label == "Open source"]
            self.assertTrue(opens)
            opens[0].click().run()
            at.run()
        self.assertFalse(at.exception, at.exception)
        md = "\n".join(m.value for m in at.markdown)
        self.assertIn("Extra detail sentence here", md)
        self.assertIn("Close", [b.label for b in at.button])


if __name__ == "__main__":
    unittest.main()
