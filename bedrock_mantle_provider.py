"""Bedrock Mantle answer provider.

OpenAI-compatible chat through the Bedrock Mantle endpoint
(https://bedrock-mantle.{region}.api.aws/v1) using SigV4 short-term
bearer tokens from the ambient AWS credential chain (EC2 instance
role in production). Same LLMProvider interface (generate /
generate_stream / is_reachable / describe) as the other providers, so
frozen retrieval, prompts, sanitizer, and citation paths behave
identically.

Rules (mirror BedrockConverseProvider):
- Credentials come ONLY from the AWS default chain via
  aws-bedrock-token-generator. Never .env secrets: there is
  deliberately no key setting.
- No silent defaults: ANSWER_MODEL_ID is required (a wrong or missing
  model ID fails loudly, never substitutes another model).
- Temperature passes through unchanged (default 0.0, same policy).
- openai / aws-bedrock-token-generator are imported lazily so other
  provider paths never pay for (or break on) their absence.
- No silent fallback: failures raise with Mantle-specific guidance.
  Callers must surface the error, never quietly switch providers.
"""

import logging

from llm_provider import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_REGION = "ap-south-1"
MANTLE_ENDPOINT = "https://bedrock-mantle.{region}.api.aws/v1"


class BedrockMantleProvider(LLMProvider):
    """AWS Bedrock via the Mantle OpenAI-compatible endpoint."""

    def __init__(self, model_id="", region=DEFAULT_REGION,
                 temperature=0.0, client_factory=None, token_fn=None):
        mid = (model_id or "").strip()
        if not mid:
            raise ValueError(
                "ANSWER_MODEL_ID is required when "
                "LLM_PROVIDER=bedrock-mantle. Set it to a Bedrock model "
                "ID served by Mantle in the configured region; no model "
                "is ever chosen silently.")
        self.model_id = mid
        self.region = (region or DEFAULT_REGION).strip() or DEFAULT_REGION
        self.temperature = float(temperature)
        self._client_factory = client_factory
        self._token_fn = token_fn

    @property
    def describe(self) -> str:
        return f"Bedrock Mantle ({self.model_id})"

    @property
    def endpoint(self) -> str:
        return MANTLE_ENDPOINT.format(region=self.region)

    def _token(self) -> str:
        fn = self._token_fn
        if fn is None:
            try:
                from aws_bedrock_token_generator import provide_token
            except ImportError as e:
                raise RuntimeError(
                    "aws-bedrock-token-generator is not installed; "
                    "Mantle answers need it "
                    "(pip install aws-bedrock-token-generator)."
                ) from e
            fn = provide_token
        return fn(region=self.region)

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory(self.endpoint, self._token())
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai is not installed; Mantle answers need it "
                "(pip install openai)."
            ) from e
        return OpenAI(api_key=self._token(), base_url=self.endpoint)

    def generate(self, context: str, question: str,
                 prompt_template: str) -> str:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        # Same chain shape as the other providers: template | model.
        # The OpenAI client speaks directly; langchain_core only
        # formats the prompt and parses the string output.
        prompt = ChatPromptTemplate.from_template(prompt_template)
        rendered = prompt.format(context=context, question=question)
        try:
            resp = self._client().chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": rendered}],
                temperature=self.temperature,
            )
        except Exception as e:
            raise RuntimeError(self.offline_message(err=e)) from e
        try:
            text = (resp.choices[0].message.content or "")
        except Exception as e:
            raise RuntimeError(self.offline_message(err=e)) from e
        return text

    def generate_stream(self, context: str, question: str,
                        prompt_template: str):
        """Stream via Mantle; fall back to whole on pre-first-token failure.

        Raw chunks pass through untouched: think-block removal stays with
        the existing backend sanitizer, exactly as for other providers.
        """
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(prompt_template)
        rendered = prompt.format(context=context, question=question)
        yielded_any = False
        try:
            stream = self._client().chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": rendered}],
                temperature=self.temperature,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta
                    token = getattr(delta, "content", None)
                except Exception:
                    token = None
                if token:
                    yielded_any = True
                    yield token
            if not yielded_any:
                yield self.generate(context, question, prompt_template)
        except Exception:
            if yielded_any:
                raise
            yield self.generate(context, question, prompt_template)

    def is_reachable(self, timeout: float = 3.0) -> bool:
        # No inference cost: a SigV4 bearer token proves ambient
        # credentials + region + signing path. Per-model entitlement
        # failures surface at generation time with mapped guidance.
        try:
            token = self._token()
        except Exception:
            return False
        return bool(token)

    def offline_message(self, config=None, err=None) -> str:
        """Clean failure guidance (no secrets, no stack traces)."""
        base = (f"Bedrock Mantle inference failed (model {self.model_id}, "
                f"region {self.region}).")
        from bedrock_provider import _classify
        code, text = _classify(err)
        if code == "credentials":
            return base + (" AWS credentials were not found; Mantle uses "
                           "the default credential chain (EC2 instance "
                           "role in production) — never .env secrets.")
        if code == "denied":
            return base + (" Access denied; confirm the EC2 role may call "
                           "bedrock-mantle:CreateInference on the region's "
                           "default project, then retry.")
        if code == "throttled":
            return base + " Request throttled; retry shortly."
        if code == "model":
            return base + (" The model ID may be wrong or unavailable via "
                           "Mantle in this region; verify ANSWER_MODEL_ID, "
                           "then retry.")
        if text:
            return base + f" {text}"
        return base + " Check region, model ID, and network, then retry."
