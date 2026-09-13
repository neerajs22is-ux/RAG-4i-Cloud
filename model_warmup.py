"""Model warmup: real readiness, never simulated.

Loads the shared embedding model (exactly once per process) and checks
LLM reachability, reporting honest states:

    loading -> ready | unavailable

No timers, no fake progress. Callers cache the result per session and
offer retry on unavailable. Retrieval reuses the warmed model through
the shared embedding-provider cache.
"""

from typing import Dict
import logging
import time

logger = logging.getLogger(__name__)

LOADING = "loading"
READY = "ready"
UNAVAILABLE = "unavailable"


def warmup(config=None) -> Dict:
    """Preload embeddings and probe the LLM. Real checks only."""
    from backend import check_llm_status
    from embeddings import get_embedding_provider

    cfg = config
    if cfg is None:
        from config import get_config
        cfg = get_config()
    _t0 = time.monotonic()
    try:
        provider = get_embedding_provider(cfg)
        fn = provider.get_embedding_function()
        fn.embed_query("readiness probe")
        embeddings = {"state": READY, "model": provider.model_name}
    except Exception as e:
        logger.warning("Embedding warmup failed: %s", e)
        return {"state": UNAVAILABLE, "embeddings": {"state": UNAVAILABLE},
                "llm": {"reachable": False},
                "detail": "Embedding model could not be loaded."}
    llm = check_llm_status(config=cfg)
    logger.info("warmup finished in %.2fs (llm reachable=%s)",
                time.monotonic() - _t0, llm.get("reachable"))
    if not llm.get("reachable"):
        return {"state": UNAVAILABLE, "embeddings": embeddings, "llm": llm,
                "detail": "Model loaded, but the assistant endpoint "
                          "is unreachable."}
    return {"state": READY, "embeddings": embeddings, "llm": llm,
            "detail": "Model ready."}


def warmup_reranker(config=None) -> Dict:
    """Preload the reranker once per process (cold-load is ~18s on CPU).

    Returns a disabled state WITHOUT loading any model when reranking
    is off. Failures report unavailable without touching the
    embeddings/LLM states. Call once at startup when
    RAG_RERANKING_ENABLED=1 so the first user query never pays the
    cold-load cost.
    """
    from reranking import get_reranker, shared_reranker
    from reranking import shared_reranker_name

    cfg = config
    if cfg is None:
        from config import get_config
        cfg = get_config()
    if get_reranker(cfg) is None:
        return {"state": "disabled", "model": None, "latency_ms": 0}
    # Warm the process-shared instance queries actually use (singleton).
    rr = shared_reranker(shared_reranker_name(cfg))
    _t0 = time.monotonic()
    try:
        rr.score("readiness probe", ["readiness probe"])
        latency = max(0, int((time.monotonic() - _t0) * 1000))
    except Exception as e:
        logger.warning("Reranker warmup failed: %s", e)
        return {"state": UNAVAILABLE,
                "model": getattr(rr, "name", "?"), "latency_ms": None}
    return {"state": READY, "model": getattr(rr, "name", "?"),
            "latency_ms": latency}
