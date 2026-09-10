"""Phase 5A tests: Bedrock provider behind LLMProvider (fakes only).

No live AWS calls, no credentials required. langchain_aws / boto3 are
never imported here: provider seams are injected or sys.modules-gated.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.runnables import RunnableLambda


def bedrock_config(**over):
    import config
    base = dict(llm_provider="bedrock", answer_model_id="us.test-model-1",
                bedrock_region="us-east-1", llm_temperature=0.0)
    base.update(over)
    return config.AppConfig(**base)


class TestFactory(unittest.TestCase):
    def test_default_is_lmstudio(self):
        from llm_provider import LMStudioProvider, get_llm_provider
        self.assertIsInstance(get_llm_provider(), LMStudioProvider)
        import config
        self.assertIsInstance(
            get_llm_provider(config.AppConfig()), LMStudioProvider)

    def test_explicit_lmstudio_unchanged(self):
        from llm_provider import LMStudioProvider, get_llm_provider
        import config
        cfg = config.AppConfig(llm_provider="lmstudio")
        prov = get_llm_provider(cfg)
        self.assertIsInstance(prov, LMStudioProvider)
        self.assertEqual(prov.temperature, 0.0)

    def test_bedrock_dispatch(self):
        from bedrock_provider import BedrockConverseProvider
        from llm_provider import get_llm_provider
        prov = get_llm_provider(bedrock_config())
        self.assertIsInstance(prov, BedrockConverseProvider)
        self.assertEqual(prov.model_id, "us.test-model-1")
        self.assertEqual(prov.region, "us-east-1")
        self.assertEqual(prov.temperature, 0.0)

    def test_unknown_provider_raises_loudly(self):
        from llm_provider import get_llm_provider
        import config
        with self.assertRaises(ValueError):
            get_llm_provider(config.AppConfig(llm_provider="openai"))

    def test_bedrock_requires_model_id(self):
        from llm_provider import get_llm_provider
        with self.assertRaises(ValueError):
            get_llm_provider(bedrock_config(answer_model_id="  "))


class TestBedrockProvider(unittest.TestCase):
    def test_constructor_requires_model_id(self):
        from bedrock_provider import BedrockConverseProvider
        with self.assertRaises(ValueError):
            BedrockConverseProvider(model_id="")
        prov = BedrockConverseProvider(model_id="m", region="us-east-1")
        self.assertIn("m", prov.describe)
        self.assertIn("Bedrock", prov.describe)

    def test_generate_kwargs_and_prompt(self):
        from bedrock_provider import BedrockConverseProvider
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        seen = {}

        def factory(**kwargs):
            seen["kwargs"] = kwargs

            def gen(x):
                seen["input"] = str(x)
                yield "CANNED-ANSWER"

            return RunnableLambda(gen)

        prov = BedrockConverseProvider(model_id="us.test-model-1",
                                       region="us-east-1", temperature=0.0)
        prov._converse_class = lambda: factory
        model = prov._converse_class()(
            model=prov.model_id, region_name=prov.region,
            temperature=prov.temperature)
        chain = ChatPromptTemplate.from_template(
            "Context:\n{context}\nQuestion:\n{question}") | model | StrOutputParser()
        out = chain.invoke({"context": "CTX-EVIDENCE", "question": "Q-QUESTION"})
        self.assertEqual(out, "CANNED-ANSWER")
        self.assertEqual(seen["kwargs"]["model"], "us.test-model-1")
        self.assertEqual(seen["kwargs"]["region_name"], "us-east-1")
        self.assertEqual(seen["kwargs"]["temperature"], 0.0)
        self.assertIn("CTX-EVIDENCE", seen["input"])
        self.assertIn("Q-QUESTION", seen["input"])

    def test_provider_generate_passthrough(self):
        from bedrock_provider import BedrockConverseProvider
        prov = BedrockConverseProvider(model_id="m")
        prov._converse_class = lambda: (lambda **kw: RunnableLambda(
            lambda x: "<think>hmm</think>REAL"))
        # Raw output passes through untouched: think-stripping stays with
        # the backend sanitizer, exactly as for LM Studio.
        self.assertEqual(
            prov.generate("c", "q", "Q:{question}"), "<think>hmm</think>REAL")

    def test_provider_stream_verbatim(self):
        from bedrock_provider import BedrockConverseProvider
        prov = BedrockConverseProvider(model_id="m")

        def gen(_x):
            yield "<think>hmm</think>"
            yield "REAL"

        prov._converse_class = lambda: (lambda **kw: RunnableLambda(gen))
        self.assertEqual(list(prov.generate_stream("c", "q", "Q:{question}")),
                         ["<think>hmm</think>", "REAL"])

    def test_provider_stream_prefailure_falls_back(self):
        from langchain_core.runnables import Runnable
        from bedrock_provider import BedrockConverseProvider

        class FailStreamChat(Runnable):
            def invoke(self, input, config=None, **kwargs):
                return "WHOLE"

            def stream(self, input, config=None, **kwargs):
                raise RuntimeError("pre-yield boom")

        prov = BedrockConverseProvider(model_id="m")
        prov._converse_class = lambda: (lambda **kw: FailStreamChat())
        self.assertEqual(
            list(prov.generate_stream("c", "q", "Q:{question}")), ["WHOLE"])

    def test_provider_stream_midfailure_raises(self):
        from bedrock_provider import BedrockConverseProvider
        prov = BedrockConverseProvider(model_id="m")

        def gen(_x):
            yield "partial"
            raise RuntimeError("mid-stream boom")

        prov._converse_class = lambda: (lambda **kw: RunnableLambda(gen))
        it = prov.generate_stream("c", "q", "Q:{question}")
        self.assertEqual(next(it), "partial")
        with self.assertRaises(RuntimeError):
            list(it)

    def test_missing_langchain_aws_is_clean_error(self):
        from bedrock_provider import BedrockConverseProvider
        prov = BedrockConverseProvider(model_id="m")
        old = sys.modules.get("langchain_aws", "--absent--")
        sys.modules["langchain_aws"] = None
        try:
            with self.assertRaises(RuntimeError) as cm:
                prov._converse_class()
            self.assertIn("langchain-aws", str(cm.exception))
        finally:
            if old == "--absent--":
                sys.modules.pop("langchain_aws", None)
            else:
                sys.modules["langchain_aws"] = old

    def test_is_reachable_true_false(self):
        from bedrock_provider import BedrockConverseProvider

        class Ctl:
            def __init__(self, fail=False):
                self.fail = fail

            def list_foundation_models(self, **kw):
                if self.fail:
                    raise RuntimeError("denied")
                return {"modelSummaries": []}

        prov = BedrockConverseProvider(model_id="m")
        prov._control = lambda: Ctl(fail=False)
        self.assertTrue(prov.is_reachable())
        prov._control = lambda: Ctl(fail=True)
        self.assertFalse(prov.is_reachable())

    def test_is_reachable_without_boto3(self):
        from bedrock_provider import BedrockConverseProvider
        prov = BedrockConverseProvider(model_id="m")
        old = sys.modules.get("boto3", "--absent--")
        sys.modules["boto3"] = None
        try:
            self.assertFalse(prov.is_reachable())
        finally:
            if old == "--absent--":
                sys.modules.pop("boto3", None)
            else:
                sys.modules["boto3"] = old


class NoCredentialsError(Exception):
    pass


class AccessDeniedException(Exception):
    pass


class ThrottlingException(Exception):
    pass


class ValidationException(Exception):
    pass


class TestOfflineMessage(unittest.TestCase):
    def test_credentials_guidance(self):
        from bedrock_provider import BedrockConverseProvider
        msg = BedrockConverseProvider(
            model_id="mid", region="us-east-1").offline_message(
                None, NoCredentialsError("nope"))
        self.assertIn("credentials", msg.lower())
        self.assertIn("instance role", msg)
        self.assertIn("mid", msg)

    def test_denied_guidance(self):
        from bedrock_provider import BedrockConverseProvider
        msg = BedrockConverseProvider(
            model_id="mid", region="r").offline_message(
                None, AccessDeniedException("AccessDenied: x"))
        self.assertIn("Access denied", msg)
        self.assertIn("InvokeModel", msg)

    def test_throttled_and_model(self):
        from bedrock_provider import BedrockConverseProvider
        prov = BedrockConverseProvider(model_id="mid", region="r")
        self.assertIn("retry", prov.offline_message(
            None, ThrottlingException("throttled")).lower())
        self.assertIn("ANSWER_MODEL_ID", prov.offline_message(
            None, ValidationException("bad id")))

    def test_unknown_carries_context(self):
        from bedrock_provider import BedrockConverseProvider
        msg = BedrockConverseProvider(
            model_id="mid", region="r").offline_message(
                None, RuntimeError("weird"))
        self.assertIn("mid", msg)
        self.assertIn("r", msg)

    def test_real_botocore_client_error(self):
        try:
            from botocore.exceptions import ClientError
        except ImportError:
            self.skipTest("botocore not installed")
        from bedrock_provider import BedrockConverseProvider
        err = ClientError({"Error": {"Code": "AccessDeniedException",
                                     "Message": "nope"}}, "Converse")
        msg = BedrockConverseProvider(
            model_id="mid", region="r").offline_message(None, err)
        self.assertIn("Access denied", msg)


class FakeLLM:
    def generate(self, context, question, prompt_template):
        raise RuntimeError("boom")

    def is_reachable(self, timeout=3.0):
        return True

    @property
    def describe(self):
        return "Fake"


class TestBackendWiring(unittest.TestCase):
    def test_bedrock_error_names_bedrock(self):
        import backend
        import config

        class Bedrockish(FakeLLM):
            def offline_message(self, cfg, err):
                return "Bedrock inference failed (model m, region r)."

        cfg = config.load_config()
        with self.assertRaises(ConnectionError) as cm:
            backend.generate_answer("q?", [{"content": "c", "file_name": "a",
                                            "page": 0, "score": 0.9}],
                                    config=cfg, llm_provider=Bedrockish())
        self.assertIn("Bedrock", str(cm.exception))
        self.assertNotIn("LM Studio", str(cm.exception))

    def test_lmstudio_literal_unchanged(self):
        import backend
        import config
        cfg = config.load_config()
        with self.assertRaises(ConnectionError) as cm:
            backend.generate_answer("q?", [{"content": "c", "file_name": "a",
                                            "page": 0, "score": 0.9}],
                                    config=cfg, llm_provider=FakeLLM())
        self.assertEqual(
            str(cm.exception),
            "LLM endpoint is unreachable. Make sure LM Studio Server is running "
            f"at {cfg.llm_base_url}.")


class TestBenchmarkSupport(unittest.TestCase):
    def test_live_helper_off_by_default(self):
        from tests.benchmarks.harness import live_bedrock_llm
        self.assertIsNone(live_bedrock_llm())

    def test_live_helper_requires_bedrock_config(self):
        import os
        from unittest import mock
        from tests.benchmarks.harness import live_bedrock_llm
        with mock.patch.dict(os.environ, {"BENCHMARK_LIVE_BEDROCK": "1"}):
            with self.assertRaises(ValueError):
                live_bedrock_llm()

    def test_cost_path_for_bedrock_keys(self):
        from tests.benchmarks.harness import run_all
        out = run_all(model_key="sonnet-4.6")
        costs = [r["metrics"]["cost_usd"] for r in out["results"]
                 if r["status"] != "skipped"]
        self.assertTrue(costs)
        self.assertTrue(all(c is not None and c >= 0 for c in costs))


if __name__ == "__main__":
    unittest.main()
