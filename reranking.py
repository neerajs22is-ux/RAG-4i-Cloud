"""Pluggable reranker for recall-then-rerank retrieval (Step 5).

 recall (broad candidates) -> rerank (query + text scores) -> top-k.

- Provider-agnostic: any scorer implementing `score(query, texts)`.
- Default: local cross-encoder (no paid cloud model needed to validate).
- Explicit offline fallback: RAG_RERANK_MODEL=overlap (token overlap).
- Disabled by default: RAG_RERANKING_ENABLED=0 preserves baseline paths.
"""

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OVERLAP_MODEL_NAME = "overlap"


class Reranker:
    """Scorer protocol: query + candidate texts -> relevance floats."""

    name = "base"

    def score(self, query, texts):
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """Local cross-encoder via sentence-transformers (lazy model load)."""

    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name
        self.name = f"cross-encoder:{model_name}"
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder "
                    "reranking (already in requirements.txt). "
                    "Install it or set RAG_RERANK_MODEL=overlap for the "
                    "offline fallback.") from e
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query, texts):
        texts = list(texts or [])
        if not texts:
            return []
        model = self._load()
        pairs = [(query, t) for t in texts]
        try:
            out = model.predict(pairs)
        except Exception:
            out = model.predict(pairs, convert_to_numpy=True)
        return [float(s) for s in list(out)]


class TokenOverlapReranker(Reranker):
    """Deterministic offline fallback: query-token coverage of each text.

    Explicit opt-in only (RAG_RERANK_MODEL=overlap). Never silent.
    """

    name = "overlap"

    @staticmethod
    def _tokens(s):
        import re
        return {t for t in re.findall(r"[a-z0-9]+", str(s or "").lower())
                if len(t) > 2}

    def score(self, query, texts):
        qt = self._tokens(query)
        if not qt:
            return [0.0 for _ in (texts or [])]
        return [len(qt & self._tokens(t)) / len(qt)
                for t in (texts or [])]


def get_reranker(config=None, model_name=None):
    """Build the configured reranker, or None when reranking is disabled.

    Raises ImportError (loud, at first use) only when a cross-encoder is
    requested but sentence-transformers is unavailable.
    """
    enabled = ""
    name = model_name
    if config is not None:
        enabled = str(getattr(config, "reranking_enabled", "0") or "0")
        if name is None:
            name = str(getattr(config, "rerank_model", DEFAULT_MODEL)
                       or DEFAULT_MODEL)
    if name is None:
        name = DEFAULT_MODEL
    if enabled not in ("1", "true", "yes", "on"):
        return None
    if name == OVERLAP_MODEL_NAME:
        return TokenOverlapReranker()
    return CrossEncoderReranker(name)


_SHARED_RERANKERS = {}


def shared_reranker(model_name=None):
    """Process-shared scorer instance (one loaded model per process).

    Separate PostgresVectorStore objects share the warmed model instead
    of each loading its own copy (matters on small hosts like t3.micro).
    Explicitly injected rerankers (tests) bypass this cache.
    """
    key = model_name or DEFAULT_MODEL
    rr = _SHARED_RERANKERS.get(key)
    if rr is None:
        if key == OVERLAP_MODEL_NAME:
            rr = TokenOverlapReranker()
        else:
            rr = CrossEncoderReranker(key)
        _SHARED_RERANKERS[key] = rr
    return rr


def shared_reranker_name(config=None):
    if config is None:
        return DEFAULT_MODEL
    return str(getattr(config, "rerank_model", DEFAULT_MODEL)
               or DEFAULT_MODEL)
