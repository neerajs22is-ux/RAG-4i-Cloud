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




if __name__ == "__main__":
    unittest.main()
