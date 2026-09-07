"""Phase 4.3 tests: memory, think cleanup, retrieval robustness."""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LOCKIN_CTX = [
    {"role": "user", "content": "What is the lock-in period in the lease deed?"},
    {"role": "assistant",
     "content": "The lock-in period in the lease deed is 36 months. "
                "Early termination requires notice."},
]


class TestMemory(unittest.TestCase):
    def test_window_bounded(self):
        from conversation_memory import ConversationMemory
        m = ConversationMemory(max_turns=2)
        for i in range(6):
            m.add("user", f"substantive question number {i} here")
            m.add("assistant", "x" * 50)
        self.assertEqual(len(m), 4)
        self.assertEqual(len(m.as_context()), 4)

    def test_focus_helpers(self):
        from conversation_memory import ConversationMemory
        m = ConversationMemory()
        m.add("user", "Hi")
        m.add("assistant", "Hi!")
        self.assertFalse(m.had_document_exchange())
        self.assertEqual(m.last_user_document_question(), "")
        m.add("user", "What is the lock-in period in the lease deed?")
        m.add("assistant", "The lock-in period is 36 months. " * 5)
        self.assertTrue(m.had_document_exchange())
        self.assertIn("lock-in", m.last_user_document_question())
        m.reset()
        self.assertEqual(len(m), 0)

    def test_observer_accepts_memory_object(self):
        from conversation_memory import ConversationMemory
        from query_router import DOCUMENT_FOLLOWUP, observe_query
        m = ConversationMemory()
        for msg in LOCKIN_CTX:
            m.add(msg["role"], msg["content"])
        obs = observe_query("Why?", m)
        self.assertEqual(obs.intent, DOCUMENT_FOLLOWUP)

    def test_unrelated_question_no_contamination(self):
        from query_router import (DOCUMENT_QUERY, OUT_OF_SCOPE,
                                  expand_followup_query, observe_query)
        self.assertEqual(observe_query("What is the capital of France?",
                                       LOCKIN_CTX).intent, OUT_OF_SCOPE)
        # Expansion still anchors only to the prior substantive question.
        q = expand_followup_query("Why?", LOCKIN_CTX)
        self.assertIn("lock-in", q.lower())
        self.assertNotIn("36 months", q)  # assistant text never evidence

    def test_greetings_not_anchors(self):
        from conversation_memory import ConversationMemory
        m = ConversationMemory()
        m.add("user", "What is the lock-in period in the lease deed?")
        m.add("assistant", "Hi!" * 1)  # short reply: no exchange to build on
        m.add("user", "thanks")
        self.assertEqual(m.last_user_document_question(),
                         "What is the lock-in period in the lease deed?")


class TestThinkCleanup(unittest.TestCase):
    def test_one_block_removed(self):
        from output_safety import strip_think_blocks
        out = strip_think_blocks("<think>reasoning here</think>\n\nFinal answer.")
        self.assertEqual(out, "Final answer.")

    def test_multiple_blocks_removed(self):
        from output_safety import strip_think_blocks
        out = strip_think_blocks("<think>a</think> Mid <think>b</think> End.")
        self.assertEqual(out, "Mid  End.")

    def test_think_only_is_empty(self):
        from output_safety import is_empty_response, strip_think_blocks
        self.assertEqual(strip_think_blocks("<think>only this</think>"), "")
        self.assertTrue(is_empty_response("<think>only this</think>"))
        self.assertTrue(is_empty_response("   "))

    def test_normal_answer_unchanged(self):
        from output_safety import strip_think_blocks
        ans = "The lock-in period is 36 months. I think this is right."
        self.assertEqual(strip_think_blocks(ans), ans)

    def test_backend_treats_think_only_gracefully(self):
        import backend
        import config

        class ThinkOnlyLLM:
            def generate(self, context, question, prompt_template):
                return "<think>hidden</think>"

        ans = backend.generate_answer(
            "q", [{"content": "ctx"}], config=config.load_config(),
            llm_provider=ThinkOnlyLLM())
        self.assertIn("empty response", ans.lower())
        self.assertNotIn("hidden", ans)

    def test_backend_strips_think_from_answer(self):
        import backend
        import config

        class ThinkLLM:
            def generate(self, context, question, prompt_template):
                return "<think>hmm</think>\n\nThe answer is 36 months."

        ans = backend.generate_answer(
            "q", [{"content": "ctx"}], config=config.load_config(),
            llm_provider=ThinkLLM())
        self.assertNotIn("<think>", ans)
        self.assertIn("36 months", ans)


class TestRetrievalRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import config
        from embeddings import get_embedding_provider
        from vector_store import get_vector_store
        # NOTE: Chroma on Windows keeps index files open, so
        # TemporaryDirectory cleanup raises; use mkdtemp + best-effort.
        cls._tmp = tempfile.mkdtemp(
            prefix="rob_", dir=r"C:\Users\91973\AppData\Local\Temp\opencode")
        config.reset_config_cache()
        os.environ["CHROMA_PATH"] = os.path.join(cls._tmp, "chroma")
        cls.cfg = config.load_config()
        ep = get_embedding_provider(cls.cfg)
        cls.vs = get_vector_store(cls.cfg, ep)

        class Doc:
            def __init__(self, content, meta):
                self.page_content = content
                self.metadata = dict(meta)
                self.id = None

        chunks = [
            Doc("The lock-in period in the lease deed is 36 months. Early "
                "termination requires 3 months notice.",
                {"source_path": "/t/lease.pdf", "source": "/t/lease.pdf",
                 "file_name": "lease.pdf", "page": 0,
                 "document_id": "d1", "chunk_id": "c1"}),
            Doc("The service contract renews annually with 30 days notice.",
                {"source_path": "/t/contract.pdf", "source": "/t/contract.pdf",
                 "file_name": "contract.pdf", "page": 0,
                 "document_id": "d2", "chunk_id": "c2"}),
        ]
        cls.vs.build_index(chunks)

    @classmethod
    def tearDownClass(cls):
        import config
        os.environ.pop("CHROMA_PATH", None)
        config.reset_config_cache()
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def raw_best(self, query):
        res = self.vs.search(query, k=2)
        return max(float(s) for _, s in res)

    def test_query_forms_exist(self):
        import backend
        forms = backend.build_query_forms("What is the lock-in period?")
        self.assertIn("What is the lock-in period", forms)  # normalized
        self.assertTrue(len(forms) >= 2)
        self.assertIn("lock-in", forms[1])

    def test_how_long_now_retrieves(self):
        import backend
        before = self.raw_best("How long is the lock-in?")
        print(f"\nBEFORE how-long: {before:.3f}")
        res = backend.retrieve_documents("How long is the lock-in?",
                                         config=self.cfg, vector_store=self.vs)
        after = res[0]["score"] if res else None
        print(f"AFTER how-long: {after}")
        self.assertIsNotNone(after)
        self.assertGreaterEqual(after, 0.3)
        self.assertEqual(res[0]["file_name"], "lease.pdf")

    def test_existing_query_still_succeeds(self):
        import backend
        res = backend.retrieve_documents("lock-in period in the lease deed",
                                         config=self.cfg, vector_store=self.vs)
        self.assertTrue(res)
        self.assertEqual(res[0]["file_name"], "lease.pdf")

    def test_irrelevant_stays_out(self):
        import backend
        res = backend.retrieve_documents("What is the capital of France?",
                                         config=self.cfg, vector_store=self.vs)
        self.assertEqual(res, [])

    def test_threshold_unchanged(self):
        self.assertAlmostEqual(self.cfg.relevance_threshold, 0.3)
        self.assertEqual(self.cfg.retrieval_k, 5)


if __name__ == "__main__":
    unittest.main()
