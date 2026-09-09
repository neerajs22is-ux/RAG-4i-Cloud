"""Starter-probe tests: probe agrees with the real answer path."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LEASE = ("The lock-in period in the lease deed is 36 months. Early "
         "termination requires 3 months notice.")


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})
        self.id = None


def lease_doc():
    return FakeDoc(LEASE, {"source_path": "/t/lease.pdf",
                           "source": "/t/lease.pdf",
                           "file_name": "lease.pdf", "page": 0,
                           "document_id": "d1", "chunk_id": "c1"})


class FakeStore:
    def __init__(self, chunks=None, topup_raises=False):
        self.chunks = list(chunks or [])
        self.topup_raises = topup_raises

    def build_index(self, chunks):
        self.chunks.extend(list(chunks))
        return len(chunks)

    def search(self, query, k=5):
        from answer_support import _stem, content_terms
        qw = {_stem(w) for w in content_terms(query)}
        out = []
        for c in self.chunks:
            cw = {_stem(w) for w in content_terms(c.page_content)}
            if qw & cw:
                out.append((c, 0.9))
        return out[:k]

    def chunks_for_source(self, file_name, limit=8):
        if self.topup_raises:
            raise RuntimeError("pgvector down")
        return [(c, None) for c in self.chunks
                if c.metadata.get("file_name") == file_name][:limit]

    def get_status(self):
        return {"ready": True}


class FakeLLM:
    def generate(self, context, question, prompt_template):
        return "GENERATED:" + question[:30]

    def generate_stream(self, context, question, prompt_template):
        yield "GENERATED:" + question[:30]

    def is_reachable(self, timeout=3.0):
        return True

    @property
    def describe(self):
        return "Fake"


class TestPreviewAgreement(unittest.TestCase):
    def test_probe_positive_implies_generated_answer(self):
        import backend
        import config
        store = FakeStore([lease_doc()])
        preview = backend.preview_answer(
            "What is the lock-in period in the lease deed?",
            config=config.load_config(), vector_store=store)
        self.assertTrue(preview["will_generate"], preview)
        ans, src = backend.query_documents(
            "What is the lock-in period in the lease deed?",
            config=config.load_config(),
            vector_store=store, llm_provider=FakeLLM())
        self.assertTrue(ans.startswith("GENERATED:"))
        self.assertTrue(src)

    def test_probe_positive_for_partial_clarifies(self):
        import backend
        import config
        store = FakeStore([lease_doc()])
        preview = backend.preview_answer(
            "What does lease.pdf cover?", config=config.load_config(),
            vector_store=store)
        # Broad overview with partial coverage still reaches generation
        # (scoping prompt), so the starter is correctly shown.
        self.assertTrue(preview["will_generate"], preview)
        ans, src = backend.query_documents(
            "What does lease.pdf cover?", config=config.load_config(),
            vector_store=store, llm_provider=FakeLLM())
        self.assertTrue(src)

    def test_probe_negative_for_routed(self):
        import backend
        import config
        preview = backend.preview_answer(
            "Hi", config=config.load_config(), vector_store=FakeStore())
        self.assertFalse(preview["will_generate"])

    def test_probe_negative_for_empty_retrieval(self):
        import backend
        import config
        preview = backend.preview_answer(
            "What is the lock-in period?", config=config.load_config(),
            vector_store=FakeStore())
        self.assertFalse(preview["will_generate"])

    def test_topup_failure_degrades_not_fails(self):
        import backend
        import config
        store = FakeStore([lease_doc()], topup_raises=True)
        ans, src = backend.query_documents(
            "What does lease.pdf cover?", config=config.load_config(),
            vector_store=store, llm_provider=FakeLLM())
        self.assertTrue(ans.startswith("GENERATED:"))
        self.assertTrue(src)


if __name__ == "__main__":
    unittest.main()
