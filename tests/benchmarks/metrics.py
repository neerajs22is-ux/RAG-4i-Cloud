"""Pure benchmark metrics (no I/O, no LLM)."""
import re


def estimate_tokens(text):
    """(count, method). tiktoken when available, else chars/4 heuristic."""
    text = text or ""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text)), "tiktoken-cl100k"
    except Exception:
        return max(0, len(text) // 4), "chars-div-4"


# USD per 1M tokens (input, output). Dated 2026-09-10; volatile.
# Local/deterministic runners cost 0 by definition.
PRICING = {
    "echo": (0.0, 0.0),
    "qwen-local": (0.0, 0.0),
    "nova-micro": (0.035, 0.14),
    "nova-lite": (0.06, 0.24),
    "nova-pro": (0.80, 3.20),
    "haiku-4.5": (1.0, 5.0),
    "sonnet-4.6": (3.0, 15.0),
}


def estimate_cost_usd(model_key, in_tokens, out_tokens):
    per_in, per_out = PRICING.get(model_key, (None, None))
    if per_in is None:
        return None, "unknown-model"
    return (in_tokens * per_in + out_tokens * per_out) / 1_000_000, "estimate"


def compute(answer, sources, info, latencies, evidence_text="",
            model_key="echo"):
    """Deterministic metric bundle. Never stores document/answer text."""
    from answer_support import assess_support, citation_guard

    sources = sources or []
    timings = (info or {}).get("timings", {}) or {}
    guard = citation_guard(answer or "", sources)
    top = None
    for s in sources:
        v = (s or {}).get("score")
        if isinstance(v, (int, float)) and (top is None or v > top):
            top = float(v)
    in_tok, in_how = estimate_tokens(evidence_text)
    out_tok, out_how = estimate_tokens(answer or "")
    cost, cost_how = estimate_cost_usd(model_key, in_tok, out_tok)
    return {
        "answer_chars": len(answer or ""),
        "source_count": len(sources),
        "source_files": sorted({(s or {}).get("file_name") for s in sources
                                if (s or {}).get("file_name")}),
        "support": assess_support("", sources)["level"] if sources
                   else "unsupported",
        "guard_flagged": guard["count"],
        "guard_total": guard["total"],
        "top_strength": top,
        "retrieval_ms": timings.get("retrieval_ms"),
        "support_ms": timings.get("support_ms"),
        "generation_ms": timings.get("generation_ms"),
        "total_ms": timings.get("total_ms"),
        "wall_ms": (latencies or {}).get("wall_ms"),
        "tokens_in": in_tok,
        "tokens_out": out_tok,
        "token_method": in_how if in_how == out_how else in_how + "+" + out_how,
        "cost_usd": cost,
        "cost_method": cost_how,
        "model_key": model_key,
        "workflow": (info or {}).get("workflow"),
        "planner_invoked": False,          # 5.0 has no planner
        "planner_valid": None,
    }


def words(text):
    return set(re.findall(r"[a-z']+", (text or "").lower()))
