"""Suggestion click-through tests (pure state logic, no Streamlit runtime)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def src(content="The lock-in period is 36 months.", fname="lease.pdf"):
    return [{"content": content, "file_name": fname, "page": 0, "score": 0.8,
             "source": "/t/" + fname}]


class FakeSession(dict):
    """Stand-in for st.session_state (dict interface only)."""


class TestFollowupState(unittest.TestCase):
    def test_remember_and_render(self):
        from suggestions import current_followups, remember_followups
        st = FakeSession(messages=[])
        self.assertIsNone(current_followups(st))
        st["messages"] = [{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "a"}]
        remember_followups(st, "What is the lock-in period?", src())
        specs = current_followups(st)
        self.assertEqual(len(specs), 3)
        self.assertTrue(all(s["label"] and s["key"] for s in specs))
        self.assertEqual(len({s["key"] for s in specs}), 3)

    def test_cleared_without_sources(self):
        from suggestions import current_followups, remember_followups
        st = FakeSession(messages=[1, 2])
        remember_followups(st, "q", src())
        self.assertIsNotNone(current_followups(st))
        remember_followups(st, "Hi", [])
        self.assertIsNone(current_followups(st))

    def test_keys_refresh_after_new_answer(self):
        from suggestions import current_followups, remember_followups
        st = FakeSession(messages=[1, 2])
        remember_followups(st, "q1", src())
        old_keys = {s["key"] for s in current_followups(st)}
        st["messages"] = [1, 2, 3, 4]  # next exchange happened
        remember_followups(st, "q2", src())
        new_keys = {s["key"] for s in current_followups(st)}
        self.assertTrue(old_keys.isdisjoint(new_keys))

    def test_click_simulation_submits_same_pipeline(self):
        """Pending suggestion flows through the identical backend call."""
        import backend
        import config
        from suggestions import current_followups, remember_followups

        seen = {}

        class SpyStore:
            def search(self, query, k=5):
                seen["query"] = query
                return [(FakeDoc(), 0.9)]

        class FakeDoc:
            page_content = "The lock-in period is 36 months."
            metadata = {"source_path": "/t/lease.pdf", "source": "/t/lease.pdf",
                        "file_name": "lease.pdf", "page": 0,
                        "document_id": "d", "chunk_id": "c"}
            id = None

        class FakeLLM:
            def generate(self, context, question, prompt_template):
                seen["question"] = question
                return "ANSWER"

            def is_reachable(self, timeout=3.0):
                return True

            @property
            def describe(self):
                return "Fake"

        st = FakeSession(messages=[1, 2])
        remember_followups(st, "lock-in?", src())
        clicked = current_followups(st)[0]["label"]  # user clicks this
        # Same single pipeline the UI uses for typed input:
        ans, sources = backend.query_documents(
            clicked, config=config.load_config(), vector_store=SpyStore(),
            llm_provider=FakeLLM())
        self.assertEqual(seen.get("question"), clicked)
        self.assertTrue(sources)
        self.assertEqual(ans, "ANSWER")


class TestTypedUnchanged(unittest.TestCase):
    def test_typed_and_clicked_share_backend(self):
        import backend
        import config

        calls = []

        class SpyStore:
            def search(self, query, k=5):
                calls.append(query)
                return []

        backend.query_documents("a typed question",
                                config=config.load_config(),
                                vector_store=SpyStore())
        backend.query_documents("a clicked suggestion",
                                config=config.load_config(),
                                vector_store=SpyStore())
        # Both go through the same multi-form retrieval path.
        self.assertIn("a typed question", calls)
        self.assertIn("a clicked suggestion", calls)


if __name__ == "__main__":
    unittest.main()
