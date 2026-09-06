"""Embedding provider abstraction (Phase 1: Hugging Face MiniLM only)."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Minimal embedding interface (swappable for future providers)."""

    @abstractmethod
    def get_embedding_function(self):
        """Return a LangChain-compatible embedding function."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Local HuggingFace embeddings (default: all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._fn = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_embedding_function(self):
        if self._fn is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._fn = HuggingFaceEmbeddings(model_name=self._model_name)
        return self._fn


def get_embedding_provider(config=None) -> EmbeddingProvider:
    """Factory returning the configured provider (MiniLM in Phase 1)."""
    model = "all-MiniLM-L6-v2"
    if config is not None:
        model = getattr(config, "embedding_model", model)
    return HuggingFaceEmbeddingProvider(model_name=model)
