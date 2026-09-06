"""Provider abstraction + retrieval behaviour tests (fakes, no servers)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})


class FakeVectorStore:
    """In-memory stand-in implementing build_index/search/get_status."""

    def __init__(self):
        self.chunks = []
        self.search_k = None

    def build_index(self, chunks):
        # Must NOT delete existing data (append-only).
        before = len(self.chunks)
        self.chunks.extend(list(chunks))
        return len(chunks)

    def search(self, query, k=5):
        self.search_k = k
        # Return (doc, score) with descending fake scores.
        out = []
        for i, c in enumerate(self.chunks[:k]):
            out.append((c, 0.9 - i * 0.1))
        return out

    def get_status(self):
        return {"ready": bool(self.chunks), "exists": bool(self.chunks),
                "chunk_count": len(self.chunks),
                "persist_directory": "fake_db"}

    def is_ready(self):
        return bool(self.chunks)


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, context, question, prompt_template):
        self.calls.append((context, question))
        assert "{context}" in prompt_template and "{question}" in prompt_template
        return f"ANSWER:{question}:{context[:10]}"

    def is_reachable(self, timeout=3.0):
        return True

    @property
    def describe(self):
        return "Fake"


class TestVectorAbstraction(unittest.TestCase):
    def test_no_chroma_import_outside_vector_store(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        offenders = []
        for fname in os.listdir(root):
            if not fname.endswith(".py") or fname == "vector_store.py":
                continue
            with open(os.path.join(root, fname), encoding="utf-8") as f:
                src = f.read()
            # Catch real Chroma implementation imports (not wrapper names/comments).
            for line in src.splitlines():
                s = line.strip()
                if s.startswith("#"):
                    continue
                if "langchain_community.vectorstores" in s and "Chroma" in s:
                    offenders.append(fname + ":" + s)
                    break
                # Direct 'from ... import Chroma' / 'import Chroma' of impl
                # (exclude 'from vector_store import ...' wrapper usage).
                if s.startswith("from ") and " import " in s and "vector_store" not in s:
                    imported = s.split(" import ", 1)[1]
                    # split comma names, strip aliases
                    names = [p.strip().split(" ")[0] for p in imported.split(",")]
                    if "Chroma" in names:
                        offenders.append(fname + ":" + s)
                        break
                if s.startswith("import ") and "vector_store" not in s:
                    names = [p.strip().split(" ")[0] for p in s[len("import "):].split(",")]
                    if "Chroma" in names:
                        offenders.append(fname + ":" + s)
                        break
        self.assertEqual(offenders, [], f"Chroma imported outside vector_store.py: {offenders}")

    def test_backend_does_not_instantiate_forbidden_classes(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "backend.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("Chroma(", src)
        self.assertNotIn("HuggingFaceEmbeddings(", src)
        self.assertNotIn("ChatOpenAI(", src)
        with open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8") as f:
            app_src = f.read()
        self.assertNotIn("Chroma(", app_src)
        self.assertNotIn("HuggingFaceEmbeddings(", app_src)
        self.assertNotIn("ChatOpenAI(", app_src)

    def test_build_index_appends_no_delete(self):
        store = FakeVectorStore()
        store.build_index([FakeDoc("a"), FakeDoc("b")])
        self.assertEqual(len(store.chunks), 2)
        store.build_index([FakeDoc("c")])
        self.assertEqual(len(store.chunks), 3)

    def test_vector_store_source_has_no_rmtree(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "vector_store.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("shutil.rmtree", src)
        self.assertNotIn("rmtree(", src)
        with open(os.path.join(os.path.dirname(__file__), "..", "backend.py"), encoding="utf-8") as f:
            bsrc = f.read()
        self.assertNotIn("shutil.rmtree", bsrc)
        self.assertNotIn("rmtree(", bsrc)


class TestRetrievalGeneration(unittest.TestCase):
    def test_retrieve_returns_structured_sources(self):
        import backend
        store = FakeVectorStore()
        store.chunks = [
            FakeDoc("lease text", {"source_path": "/docs/lease.pdf",
                                   "source": "/docs/lease.pdf",
                                   "file_name": "lease.pdf", "page": 14,
                                   "document_id": "d1", "chunk_id": "c1"}),
            FakeDoc("other", {"source_path": "/docs/other.pdf",
                              "source": "/docs/other.pdf",
                              "file_name": "other.pdf", "page": None,
                              "document_id": "d2", "chunk_id": "c2"}),
        ]
        import config
        cfg = config.load_config()
        res = backend.retrieve_documents("lock-in?", config=cfg, vector_store=store)
        self.assertEqual(store.search_k, 5)  # top-k preserved
        self.assertTrue(len(res) >= 1)
        for s in res:
            for key in ["source", "file_name", "page", "score"]:
                self.assertIn(key, s)
        self.assertEqual(res[0]["file_name"], "lease.pdf")
        self.assertEqual(res[0]["page"], 14)
        self.assertIsNotNone(res[0]["score"])

    def test_threshold_filters_low_scores(self):
        import backend
        import config
        cfg = config.load_config()

        class LowStore(FakeVectorStore):
            def search(self, query, k=5):
                return [(FakeDoc("x", {"source_path": "/a.pdf", "source": "/a.pdf",
                                               "file_name": "a.pdf", "page": 0,
                                               "document_id": "d", "chunk_id": "c"}), 0.1)]
        res = backend.retrieve_documents("q", config=cfg, vector_store=LowStore())
        self.assertEqual(res, [])

    def test_generate_separated_from_retrieve(self):
        import backend
        import config
        cfg = config.load_config()
        fake_llm = FakeLLM()
        retrieved = [{"content": "CTX1234567890", "source": "/a.pdf",
                      "file_name": "a.pdf", "page": 1, "score": 0.9}]
        ans = backend.generate_answer("What?", retrieved, config=cfg,
                                      llm_provider=fake_llm)
        self.assertTrue(ans.startswith("ANSWER:"))
        self.assertEqual(len(fake_llm.calls), 1)
        # Empty retrieval -> low-relevance message, no LLM call
        fake2 = FakeLLM()
        ans2 = backend.generate_answer("What?", [], config=cfg, llm_provider=fake2)
        self.assertIn("could not find", ans2.lower())
        self.assertEqual(len(fake2.calls), 0)

    def test_query_uses_threshold_and_returns_structured(self):
        import backend
        import config
        cfg = config.load_config()
        store = FakeVectorStore()
        store.chunks = [FakeDoc("t", {"source_path": "/d/a.pdf", "source": "/d/a.pdf",
                                              "file_name": "a.pdf", "page": 2,
                                              "document_id": "d", "chunk_id": "c"})]
        ans, sources = backend.query_documents("q", config=cfg,
                                               vector_store=store,
                                               llm_provider=FakeLLM())
        self.assertTrue(ans.startswith("ANSWER:"))
        self.assertEqual(sources[0]["file_name"], "a.pdf")

    def test_prompt_template_unchanged(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "backend.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn("You are an expert legal assistant for a Chartered Accountant firm.", src)
        self.assertIn('Answer the question based ONLY on the following context.', src)
        self.assertIn('If the answer is not in the context, strictly say "I cannot find this information in the provided documents."', src)
        self.assertIn("{context}", src)
        self.assertIn("{question}", src)


class TestLLMProvider(unittest.TestCase):
    def test_llm_uses_config_values(self):
        from llm_provider import LMStudioProvider, get_llm_provider
        import config
        cfg = config.load_config()
        p = get_llm_provider(cfg)
        self.assertIsInstance(p, LMStudioProvider)
        self.assertEqual(p.base_url, cfg.llm_base_url)
        self.assertEqual(p.model, cfg.llm_model)
        self.assertAlmostEqual(p.temperature, 0.0)

    def test_embedding_provider_model(self):
        from embeddings import get_embedding_provider
        import config
        cfg = config.load_config()
        ep = get_embedding_provider(cfg)
        self.assertEqual(ep.model_name, "all-MiniLM-L6-v2")


if __name__ == "__main__":
    unittest.main()
