"""Frontend-state tests: warmup, labels, copy, progress hook, retry."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def src(content="The lock-in period is 36 months.", fname="lease.pdf"):
    return [{"content": content, "file_name": fname, "page": 0, "score": 0.8,
             "source": "/t/" + fname, "source_path": "/t/" + fname,
             "document_id": "d", "chunk_id": "c"}]


class TestWarmup(unittest.TestCase):
    def test_ready_when_model_and_llm_ok(self):
        import model_warmup
        fake_fn = mock.MagicMock()
        fake_provider = mock.MagicMock()
        fake_provider.get_embedding_function.return_value = fake_fn
        fake_provider.model_name = "all-MiniLM-L6-v2"
        with mock.patch("embeddings.get_embedding_provider",
                        return_value=fake_provider), \
             mock.patch("backend.check_llm_status",
                        return_value={"reachable": True}):
            import config
            res = model_warmup.warmup(config.load_config())
        self.assertEqual(res["state"], model_warmup.READY)
        fake_fn.embed_query.assert_called_once()

    def test_unavailable_when_llm_down(self):
        import model_warmup
        fake_fn = mock.MagicMock()
        fake_provider = mock.MagicMock()
        fake_provider.get_embedding_function.return_value = fake_fn
        fake_provider.model_name = "m"
        with mock.patch("embeddings.get_embedding_provider",
                        return_value=fake_provider), \
             mock.patch("backend.check_llm_status",
                        return_value={"reachable": False}):
            import config
            res = model_warmup.warmup(config.load_config())
        self.assertEqual(res["state"], model_warmup.UNAVAILABLE)

    def test_unavailable_when_embeddings_fail(self):
        import model_warmup
        with mock.patch("embeddings.get_embedding_provider",
                        side_effect=RuntimeError("no model")):
            res = model_warmup.warmup()
        self.assertEqual(res["state"], model_warmup.UNAVAILABLE)

    def test_shared_provider_cache(self):
        import embeddings
        import config
        cfg = config.load_config()
        self.assertIs(embeddings.get_embedding_provider(cfg),
                      embeddings.get_embedding_provider(cfg))


class TestAnswerLabels(unittest.TestCase):
    def test_grounded_partial_none(self):
        from ui.components import response_label
        lease = src()[0]
        self.assertEqual(
            response_label("How often does the contract renew?",
                           [dict(lease, content="The service contract "
                                                "renews annually.")], True),
            "Grounded answer")
        self.assertEqual(
            response_label("Are there exceptions?", [lease], True),
            "Partial answer")
        self.assertEqual(response_label("q?", [], True), "Not enough context")
        self.assertIsNone(response_label("Hi", [], False))

    def test_copy_payload_is_plain_text(self):
        from ui.components import copy_button_html
        html = copy_button_html("Answer **bold**", "copy_1")
        self.assertIn("Copy answer", html)
        self.assertIn("Answer **bold**", html)
        self.assertNotIn("<script>alert", html)


class TestProgressHook(unittest.TestCase):
    def test_phases_fire_in_order(self):
        import backend
        import config

        phases = []

        class Store:
            def search(self, query, k=5):
                return []

            def chunks_for_source(self, file_name, limit=8):
                return []

            def get_status(self):
                return {"ready": False}

        backend.query_documents(
            "What is the lock-in period in the lease deed?",
            config=config.load_config(), vector_store=Store(),
            on_phase=phases.append)
        self.assertEqual(phases, ["retrieving"])

    def test_no_hook_by_default(self):
        import backend
        import config

        class Store:
            def search(self, query, k=5):
                return []

            def chunks_for_source(self, file_name, limit=8):
                return []

            def get_status(self):
                return {"ready": False}

        ans, _ = backend.query_documents(
            "What is the lock-in period in the lease deed?",
            config=config.load_config(), vector_store=Store())
        self.assertIn("could not find", ans.lower())


if __name__ == "__main__":
    unittest.main()
