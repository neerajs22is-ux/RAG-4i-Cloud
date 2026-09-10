"""Bedrock answer provider (Phase 5A).

Converse / ConverseStream through the existing LLMProvider interface
(generate / generate_stream / is_reachable / describe), so the frozen
retrieval, prompt, sanitizer, and citation paths behave identically.

Rules:
- Credentials come ONLY from the boto3 default chain (environment,
  shared config, or EC2 instance role). Never .env secrets: there is
  deliberately no BEDROCK_API_KEY setting.
- No silent defaults: ANSWER_MODEL_ID is required (a wrong or missing
  model ID fails loudly, never substitutes another model).
- Temperature passes through unchanged (default 0.0, same policy).
- langchain_aws / boto3 are imported lazily so the default LM Studio
  path never pays for (or breaks on) their absence.
"""

import logging

from llm_provider import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_REGION = "us-east-1"


class BedrockConverseProvider(LLMProvider):
    """AWS Bedrock via the Converse API (langchain_aws, same chain shape
    as LMStudioProvider: prompt template | chat model | string output)."""

    def __init__(self, model_id="", region=DEFAULT_REGION, temperature=0.0):
        mid = (model_id or "").strip()
        if not mid:
            raise ValueError(
                "ANSWER_MODEL_ID is required when LLM_PROVIDER=bedrock. "
                "Set it to a Bedrock model ID or inference-profile ID "
                "(e.g. a us.* profile); no model is ever chosen silently.")
        self.model_id = mid
        self.region = (region or DEFAULT_REGION).strip() or DEFAULT_REGION
        self.temperature = float(temperature)

    @property
    def describe(self) -> str:
        return f"Bedrock ({self.model_id})"

    def _converse_class(self):
        try:
            from langchain_aws import ChatBedrockConverse
        except ImportError as e:
            raise RuntimeError(
                "langchain-aws is not installed; Bedrock answers need it "
                "(pip install langchain-aws). LM Studio remains the default."
            ) from e
        return ChatBedrockConverse

    def _chat_model(self):
        return self._converse_class()(
            model=self.model_id,
            region_name=self.region,
            temperature=self.temperature,
        )

    def generate(self, context: str, question: str, prompt_template: str) -> str:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        model = self._chat_model()
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | model | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def generate_stream(self, context: str, question: str, prompt_template: str):
        """Stream via the same chain/config; fall back to whole on failure.

        Raw chunks pass through untouched: think-block removal stays with
        the existing backend sanitizer, exactly as for LM Studio.
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        model = self._chat_model()
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | model | StrOutputParser()
        yielded_any = False
        try:
            for token in chain.stream({"context": context, "question": question}):
                if token:
                    yielded_any = True
                    yield token
        except Exception:
            if yielded_any:
                raise
            yield self.generate(context, question, prompt_template)

    def _control(self):
        """Bedrock control-plane client (reachability/entitlement probe)."""
        try:
            import boto3
        except ImportError as e:
            raise RuntimeError(
                "boto3 is not installed; Bedrock answers need it."
            ) from e
        return boto3.client("bedrock", region_name=self.region)

    def is_reachable(self, timeout: float = 3.0) -> bool:
        # Control-plane only (no inference cost): endpoint + credentials
        # + basic API access. Per-model entitlement failures surface at
        # generation time with mapped guidance (see offline_message).
        try:
            self._control().list_foundation_models(maxResults=1)
        except Exception:
            return False
        return True

    def offline_message(self, config=None, err=None) -> str:
        """Clean failure guidance (no secrets, no stack traces)."""
        base = (f"Bedrock inference failed (model {self.model_id}, "
                f"region {self.region}).")
        code, text = _classify(err)
        if code == "credentials":
            return base + (" AWS credentials were not found; Bedrock uses "
                           "the default credential chain (environment or EC2 "
                           "instance role) — never .env secrets.")
        if code == "denied":
            return base + (" Access denied; confirm model access (EULA) for "
                           "this model and bedrock:InvokeModel permission, "
                           "then retry.")
        if code == "throttled":
            return base + " Request throttled; retry shortly."
        if code == "model":
            return base + (" The model ID may be wrong or unavailable in "
                           "this region; verify ANSWER_MODEL_ID, then retry.")
        if text:
            return base + f" {text}"
        return base + " Check region, model ID, and network, then retry."


def _classify(err):
    """(code, short_detail): credentials|denied|throttled|model|unknown.

    Prefers botocore error codes when importable; falls back to
    exception-name matching so classification never depends on boto3.
    Raw detail is truncated and never carries key material (callers only
    pass exception text, and ARNs at most).
    """
    if err is None:
        return "unknown", ""
    try:
        from botocore.exceptions import ClientError, NoCredentialsError
    except Exception:
        ClientError, NoCredentialsError = (), ()
    if NoCredentialsError and isinstance(err, NoCredentialsError):
        return "credentials", ""
    if ClientError and isinstance(err, ClientError):
        code = ""
        try:
            code = (err.response.get("Error", {}) or {}).get("Code", "") or ""
        except Exception:
            code = ""
        up = code.upper()
        if "DENIED" in up or "UNAUTHORIZED" in up or "FORBIDDEN" in up:
            return "denied", ""
        if "THROTTL" in up or "TOO_MANY" in up:
            return "throttled", ""
        if "VALIDATION" in up or "NOT_FOUND" in up or "MODEL" in up:
            return "model", ""
        return "unknown", code[:80]
    name = type(err).__name__
    if "NoCredentials" in name or "PartialCredentials" in name:
        return "credentials", ""
    detail = str(err)[:160]
    low = (name + " " + detail).lower()
    if "accessdenied" in name.lower() or "access denied" in low \
            or "not authorized" in low or "forbidden" in low:
        return "denied", ""
    if "throttl" in low:
        return "throttled", ""
    if "validation" in name.lower() or "unknown model" in low \
            or "not found" in low:
        return "model", ""
    return "unknown", detail
