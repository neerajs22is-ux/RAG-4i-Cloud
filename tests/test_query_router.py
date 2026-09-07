"""Conversational routing tests (no retrieval for chat, RAG intact)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class ExplodingStore:
    """Fails loudly if retrieval is attempted."""

    def search(self, query, k=5):
        raise AssertionError("retrieval must not run for routed replies")

    def get_status(self):
        raise AssertionError("retrieval must not run for routed replies")


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})
        self.id = None


class FakeStore:
    def __init__(self):
        self.chunks = []

    def build_index(self, chunks):
        self.chunks.extend(list(chunks))
        return len(chunks)

    def search(self, query, k=5):
        return [(c, 0.9) for c in self.chunks[:k]]

    def get_status(self):
        return {"ready": True, "exists": True, "chunk_count": 1,
                "document_count": 1, "persist_directory": "fake"}


class FakeLLM:
    def generate(self, context, question, prompt_template):
        return "GROUNDED:" + context[:20]

    def is_reachable(self, timeout=3.0):
        return True

    @property
    def describe(self):
        return "Fake"


class TestRouter(unittest.TestCase):
    def test_greetings_route_conversational(self):
        from query_router import CONVERSATIONAL, route_query
        for msg in ["Hi", "hello", "hey!", "good morning",
                    "how are you?", "HOW ARE YOU"]:
            self.assertEqual(route_query(msg), CONVERSATIONAL, msg)

    def test_document_questions_route_to_rag(self):
        from query_router import DOCUMENT_QUERY, route_query
        for msg in ["What is the lock-in period?",
                    "Who are the parties to the lease?",
                    "What are the termination conditions?",
                    "When does the agreement expire?",
                    "Tell me about the rent clause"]:
            self.assertEqual(route_query(msg), DOCUMENT_QUERY, msg)

    def test_out_of_scope_detected(self):
        from query_router import OUT_OF_SCOPE, route_query
        for msg in ["What is the capital of France?",
                    "Write me a poem.",
                    "What's today's weather?"]:
            self.assertEqual(route_query(msg), OUT_OF_SCOPE, msg)


class TestRoutedReplies(unittest.TestCase):
    def test_hi_no_retrieval(self):
        import backend
        import config
        ans, src = backend.query_documents(
            "Hi", config=config.load_config(),
            vector_store=ExplodingStore(), llm_provider=FakeLLM())
        self.assertIn("Hi!", ans)
        self.assertIn("documents", ans)
        self.assertEqual(src, [])

    def test_hello_no_retrieval(self):
        import backend
        import config
        ans, src = backend.query_documents(
            "Hello", config=config.load_config(),
            vector_store=ExplodingStore(), llm_provider=FakeLLM())
        self.assertIn("Hi!", ans)
        self.assertEqual(src, [])

    def test_thanks_no_retrieval(self):
        import backend
        import config
        ans, src = backend.query_documents(
            "Thanks", config=config.load_config(),
            vector_store=ExplodingStore(), llm_provider=FakeLLM())
        self.assertEqual(ans, "You're welcome!")
        self.assertEqual(src, [])

    def test_out_of_scope_no_retrieval(self):
        import backend
        import config
        ans, src = backend.query_documents(
            "What is the capital of France?", config=config.load_config(),
            vector_store=ExplodingStore(), llm_provider=FakeLLM())
        self.assertIn("documents connected", ans)
        self.assertNotIn("Paris", ans)
        self.assertEqual(src, [])

    def test_document_question_uses_rag(self):
        import backend
        import config
        store = FakeStore()
        store.chunks = [FakeDoc(
            "The lock-in period is 36 months.",
            {"source_path": "/d/lease.pdf", "source": "/d/lease.pdf",
             "file_name": "lease.pdf", "page": 1,
             "document_id": "d", "chunk_id": "c"})]
        ans, src = backend.query_documents(
            "What is the lock-in period?", config=config.load_config(),
            vector_store=store, llm_provider=FakeLLM())
        self.assertTrue(ans.startswith("GROUNDED:"))
        self.assertEqual(src[0]["file_name"], "lease.pdf")

    def test_irrelevant_question_keeps_refusal(self):
        import backend
        import config

        class EmptyStore(FakeStore):
            def search(self, query, k=5):
                return []

        ans, src = backend.query_documents(
            "What is the lock-in period for unicorns?",
            config=config.load_config(), vector_store=EmptyStore(),
            llm_provider=FakeLLM())
        self.assertIn("could not find", ans.lower())
        self.assertEqual(src, [])

    def test_citation_intact(self):
        import backend
        import config
        store = FakeStore()
        store.chunks = [FakeDoc(
            "Rent is Rs 50000.",
            {"source_path": "/d/rent.pdf", "source": "/d/rent.pdf",
             "file_name": "rent.pdf", "page": 3,
             "document_id": "d", "chunk_id": "c"})]
        ans, src = backend.query_documents(
            "What is the rent?", config=config.load_config(),
            vector_store=store, llm_provider=FakeLLM())
        self.assertEqual(src[0]["file_name"], "rent.pdf")
        self.assertEqual(src[0]["page"], 3)
        self.assertAlmostEqual(src[0]["score"], 0.9)


if __name__ == "__main__":
    unittest.main()
