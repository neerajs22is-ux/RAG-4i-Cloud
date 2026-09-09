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
