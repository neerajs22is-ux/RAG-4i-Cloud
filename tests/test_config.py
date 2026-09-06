"""Config loading tests (stdlib only, no model downloads)."""

import os
import unittest


class TestConfig(unittest.TestCase):
    def setUp(self):
        # Ensure fresh config per test.
        import config
        config.reset_config_cache()
        self._old = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old)
        import config
        config.reset_config_cache()

    def test_defaults_match_original_local_behaviour(self):
        for k in ["APP_ENV", "VECTOR_STORE", "CHROMA_PATH", "EMBEDDING_MODEL",
                  "LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
                  "LLM_TEMPERATURE"]:
            os.environ.pop(k, None)
        import config
        config.reset_config_cache()
        cfg = config.load_config()
        self.assertEqual(cfg.app_env, "local")
        self.assertEqual(cfg.vector_store, "chroma")
        self.assertEqual(cfg.chroma_path, "chroma_db")
        self.assertEqual(cfg.embedding_model, "all-MiniLM-L6-v2")
        self.assertEqual(cfg.llm_provider, "lmstudio")
        self.assertEqual(cfg.llm_base_url, "http://localhost:1234/v1")
        self.assertEqual(cfg.llm_api_key, "lm-studio")
        self.assertEqual(cfg.llm_model, "local-model")
        self.assertEqual(cfg.llm_temperature, 0.0)

    def test_preserved_values(self):
        import config
        cfg = config.load_config()
        self.assertEqual(cfg.chunk_size, 1000)
        self.assertEqual(cfg.chunk_overlap, 200)
        self.assertEqual(cfg.retrieval_k, 5)
        self.assertAlmostEqual(cfg.relevance_threshold, 0.3)

    def test_env_override(self):
        os.environ["EMBEDDING_MODEL"] = "custom-model"
        os.environ["LLM_TEMPERATURE"] = "0.7"
        os.environ["CHROMA_PATH"] = "custom_db"
        import config
        config.reset_config_cache()
        cfg = config.load_config()
        self.assertEqual(cfg.embedding_model, "custom-model")
        self.assertAlmostEqual(cfg.llm_temperature, 0.7)
        self.assertEqual(cfg.chroma_path, "custom_db")

    def test_no_secrets_in_source(self):
        # Config source must not hardcode real secrets; defaults are local dummy values.
        with open(os.path.join(os.path.dirname(__file__), "..", "config.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("sk-", src)


if __name__ == "__main__":
    unittest.main()
