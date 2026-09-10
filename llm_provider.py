"""LLM provider abstraction (Phase 1: LM Studio only, OpenAI-compatible)."""

import urllib.request
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Minimal LLM interface (swappable for future cloud providers)."""

    @abstractmethod
    def generate(self, context: str, question: str, prompt_template: str) -> str:
        raise NotImplementedError

    def generate_stream(self, context: str, question: str, prompt_template: str):
        """Yield response text incrementally; default falls back to whole."""
        yield self.generate(context, question, prompt_template)

    @abstractmethod
    def is_reachable(self, timeout: float = 3.0) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError


class LMStudioProvider(LLMProvider):
    """LM Studio via its OpenAI-compatible API (ChatOpenAI under the hood)."""

    def __init__(self, base_url="http://localhost:1234/v1",
                 api_key="lm-studio", model="local-model", temperature=0.0):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature)

    @property
    def describe(self) -> str:
        return f"LM Studio ({self.model})"

    def _chat_model(self):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
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
        """Stream via the same chain/config; fall back to whole on failure."""
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
                # Partial output already emitted; let the caller decide
                # (backend discards it and falls back cleanly).
                raise
            # Nothing emitted yet: fall back to non-streaming generation.
            yield self.generate(context, question, prompt_template)

    def is_reachable(self, timeout: float = 3.0) -> bool:
        # LM Studio exposes OpenAI-compatible /models; fall back to base URL.
        candidates = [
            self.base_url.rstrip("/") + "/models",
            self.base_url.rstrip("/"),
        ]
        for url in candidates:
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if 200 <= resp.status < 500:
                        return True
            except Exception:
                continue
        return False

    def offline_message(self, config=None, err=None) -> str:
        """User-safe failure guidance. Kept byte-identical to the historic
        backend wording so the lmstudio path behaviour never changes."""
        base = "http://localhost:1234/v1"
        if config is not None:
            base = getattr(config, "llm_base_url", base)
        return ("LLM endpoint is unreachable. Make sure LM Studio Server "
                f"is running at {base}.")


def get_llm_provider(config=None) -> LLMProvider:
    """Factory dispatching on config (lmstudio default; bedrock opt-in).

    Unknown LLM_PROVIDER values raise loudly: providers are never
    substituted silently.
    """
    name = "lmstudio"
    if config is not None:
        name = (getattr(config, "llm_provider", name) or name).lower()
    if name == "bedrock":
        from bedrock_provider import BedrockConverseProvider

        return BedrockConverseProvider(
            model_id=getattr(config, "answer_model_id", "") if config is not None else "",
            region=getattr(config, "bedrock_region", "us-east-1") if config is not None else "us-east-1",
            temperature=getattr(config, "llm_temperature", 0.0) if config is not None else 0.0,
        )
    if name == "lmstudio":
        return _lmstudio_provider(config)
    raise ValueError(
        f"Unknown LLM_PROVIDER: {name!r} (expected 'lmstudio' or 'bedrock').")


def _lmstudio_provider(config=None) -> LLMProvider:
    """Original LM Studio construction (unchanged defaults)."""
    base_url = "http://localhost:1234/v1"
    api_key = "lm-studio"
    model = "local-model"
    temperature = 0.0
    if config is not None:
        base_url = getattr(config, "llm_base_url", base_url)
        api_key = getattr(config, "llm_api_key", api_key)
        model = getattr(config, "llm_model", model)
        temperature = getattr(config, "llm_temperature", temperature)
    return LMStudioProvider(
        base_url=base_url, api_key=api_key, model=model, temperature=temperature
    )
