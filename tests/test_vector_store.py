"""Vector-store factory + status (incl. document_count) + provider wiring.

Uses dummy embeddings + temp dirs so no model download or server is needed.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class DummyEmbeddings:
    """Minimal LangChain-compatible embeddings (fixed vectors, no download)."""

    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.0]


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})
        # Match langchain_core Document interface used by Chroma.add_documents.
        self.id = None


def make_chunks():
    return [
        FakeDoc("lease text part 1", {"source_path": "/tmp/a.pdf", "source": "/tmp/a.pdf",
                                      "file_name": "a.pdf", "page": 0,
                                      "document_id": "doc-A", "chunk_id": "c1"}),
        FakeDoc("lease text part 2", {"source_path": "/tmp/a.pdf", "source": "/tmp/a.pdf",
                                      "file_name": "a.pdf", "page": 1,
                                      "document_id": "doc-A", "chunk_id": "c2"}),
        FakeDoc("other doc", {"source_path": "/tmp/b.pdf", "source": "/tmp/b.pdf",
                              "file_name": "b.pdf", "page": 0,
                              "document_id": "doc-B", "chunk_id": "c3"}),
    ]


class TestVectorStoreFactory(unittest.TestCase):
    def test_factory_returns_chroma_store(self):
        from vector_store import ChromaVectorStore, get_vector_store
        vs = get_vector_store()
        self.assertIsInstance(vs, ChromaVectorStore)

    def test_factory_respects_config_path(self):
        from vector_store import get_vector_store
        import config

        cfg = config.load_config()
        vs = get_vector_store(cfg)
        self.assertTrue(vs.persist_directory.endswith(os.path.basename(cfg.chroma_path)))

    def test_only_vector_store_imports_chroma(self):
        # Implementation lives in exactly one module (no architectural change).
        import vector_store
        import inspect
        src = inspect.getsource(vector_store)
        self.assertIn("from langchain_community.vectorstores import Chroma", src)


class TestVectorStoreStatus(unittest.TestCase):
    def test_missing_db_status_has_document_count(self):
        from vector_store import ChromaVectorStore
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "nope")
            vs = ChromaVectorStore(persist_directory=missing)
            st = vs.get_status()
            self.assertFalse(st["exists"])
            self.assertFalse(st["ready"])
            self.assertIsNone(st["chunk_count"])
            self.assertIn("document_count", st)
            self.assertIsNone(st["document_count"])
            self.assertFalse(vs.is_ready())

    def test_real_chroma_counts_chunks_and_documents(self):
        from vector_store import ChromaVectorStore
        # NOTE: Chroma on Windows keeps data_level0.bin open, so
        # TemporaryDirectory cleanup raises PermissionError. Use mkdtemp +
        # best-effort cleanup instead (test-only, no app change).
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        vs = ChromaVectorStore(persist_directory=os.path.join(tmp, "db"),
                               embedding_function=DummyEmbeddings())
        added = vs.build_index(make_chunks())
        self.assertEqual(added, 3)
        st = vs.get_status()
        self.assertTrue(st["exists"])
        self.assertTrue(st["ready"])
        self.assertEqual(st["chunk_count"], 3)
        self.assertEqual(st["document_count"], 2)
        self.assertTrue(vs.is_ready())
        # Append-only: second build grows, never deletes.
        vs.build_index(make_chunks()[:1])
        st2 = vs.get_status()
        self.assertEqual(st2["chunk_count"], 4)
        self.assertEqual(st2["document_count"], 2)

    def test_search_returns_scored_tuples(self):
        from vector_store import ChromaVectorStore
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        vs = ChromaVectorStore(persist_directory=os.path.join(tmp, "db"),
                               embedding_function=DummyEmbeddings())
        vs.build_index(make_chunks())
        res = vs.search("lease", k=2)
        self.assertEqual(len(res), 2)
        for doc, score in res:
            self.assertTrue(hasattr(doc, "page_content"))
            self.assertIsInstance(float(score), float)

    def test_backend_status_forwards_document_count(self):
        import backend
        import config
        cfg = config.load_config()

        class FakeVS:
            def get_status(self):
                return {"ready": True, "exists": True, "chunk_count": 10,
                        "document_count": 3, "persist_directory": "x"}
        st = backend.get_knowledge_base_status(config=cfg, vector_store=FakeVS())
        self.assertEqual(st["document_count"], 3)
        self.assertEqual(st["chunk_count"], 10)


class TestProviderWiring(unittest.TestCase):
    def test_embedding_factory_respects_config_without_download(self):
        from embeddings import HuggingFaceEmbeddingProvider, get_embedding_provider
        import config
        cfg = config.load_config()
        ep = get_embedding_provider(cfg)
        self.assertIsInstance(ep, HuggingFaceEmbeddingProvider)
        self.assertEqual(ep.model_name, "all-MiniLM-L6-v2")

    def test_llm_factory_respects_config(self):
        from llm_provider import LMStudioProvider, get_llm_provider
        import config
        cfg = config.load_config()
        p = get_llm_provider(cfg)
        self.assertIsInstance(p, LMStudioProvider)
        self.assertEqual(p.base_url, cfg.llm_base_url)
        self.assertIn(p.model, p.describe)

    def test_llm_unreachable_host(self):
        from llm_provider import LMStudioProvider
        p = LMStudioProvider(base_url="http://127.0.0.1:1/v1")
        self.assertFalse(p.is_reachable(timeout=1.0))


if __name__ == "__main__":
    unittest.main()
