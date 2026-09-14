"""Bedrock Mantle provider tests (mocked; no AWS/network needed)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content="mantle answer", fail=None):
        self.content = content
        self.fail = fail
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail is not None:
            raise self.fail
        return FakeResponse(self.content)


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, completions):
        self.chat = FakeChat(completions)
        self._endpoint = None
        self._token = None


def _provider(comp=None, **kw):
    from bedrock_mantle_provider import BedrockMantleProvider
    comp = comp or FakeCompletions()
    maker = {"client": None}

    def factory(endpoint, token):
        cli = FakeClient(comp)
        cli._endpoint = endpoint
        cli._token = token
        maker["client"] = cli
        return cli

    kw.setdefault("model_id", "qwen.qwen3-235b-a22b-2507-v1:0")
    kw.setdefault("token_fn", lambda region=None: "tok")
    prov = BedrockMantleProvider(client_factory=factory, **kw)
    return prov, comp, maker


class TestMantleProvider(unittest.TestCase):
    def test_describe_and_endpoint(self):
        prov, _, _ = _provider()
        self.assertIn("Mantle", prov.describe)
        self.assertIn("qwen", prov.describe)
        self.assertEqual(
            prov.endpoint,
            "https://bedrock-mantle.ap-south-1.api.aws/v1")

    def test_model_id_required(self):
        from bedrock_mantle_provider import BedrockMantleProvider
        with self.assertRaises(ValueError):
            BedrockMantleProvider(model_id="",
                                  token_fn=lambda region=None: "t")

    def test_generate_uses_mantle_endpoint_and_model(self):
        prov, comp, maker = _provider()
        out = prov.generate("ctx", "q?", "Answer: {question}\n{context}")
        self.assertEqual(out, "mantle answer")
        self.assertEqual(comp.calls[0]["model"],
                         "qwen.qwen3-235b-a22b-2507-v1:0")
        self.assertEqual(maker["client"]._endpoint, prov.endpoint)
        self.assertEqual(maker["client"]._token, "tok")

    def test_generate_failure_is_loud_not_silent(self):
        prov, _, _ = _provider(comp=FakeCompletions(
            fail=RuntimeError("denied")))
        with self.assertRaises(RuntimeError):
            prov.generate("ctx", "q?", "Answer: {question}\n{context}")

    def test_is_reachable_without_inference(self):
        prov, comp, _ = _provider()
        self.assertTrue(prov.is_reachable())
        self.assertEqual(comp.calls, [])

    def test_factory_selects_mantle_explicitly(self):
        import config
        from bedrock_mantle_provider import BedrockMantleProvider
        from llm_provider import get_llm_provider
        config.reset_config_cache()
        os.environ["LLM_PROVIDER"] = "bedrock-mantle"
        os.environ["ANSWER_MODEL_ID"] = "qwen.qwen3-235b-a22b-2507-v1:0"
        os.environ["BEDROCK_REGION"] = "ap-south-1"
        try:
            prov = get_llm_provider(config.load_config())
            self.assertIsInstance(prov, BedrockMantleProvider)
        finally:
            for k in ("LLM_PROVIDER", "ANSWER_MODEL_ID", "BEDROCK_REGION"):
                os.environ.pop(k, None)
            config.reset_config_cache()

    def test_factory_bedrock_still_converse(self):
        import config
        from bedrock_provider import BedrockConverseProvider
        from llm_provider import get_llm_provider
        config.reset_config_cache()
        os.environ["LLM_PROVIDER"] = "bedrock"
        os.environ["ANSWER_MODEL_ID"] = "qwen.qwen3-235b-a22b-2507-v1:0"
        try:
            prov = get_llm_provider(config.load_config())
            self.assertIsInstance(prov, BedrockConverseProvider)
        finally:
            for k in ("LLM_PROVIDER", "ANSWER_MODEL_ID"):
                os.environ.pop(k, None)
            config.reset_config_cache()


if __name__ == "__main__":
    unittest.main()
