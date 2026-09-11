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
# sonnet-5/luna re-verified 2026-09-10 (us.* profile rate for Sonnet 5
# carries the ~10% regional premium over the $2.00/$10.00 global rate;
# Luna $0.22/$1.32 post July-2026 Bedrock cut). Re-verify at live-run
# time; do NOT silently reuse these if the pricing page differs.
# Local/deterministic runners cost 0 by definition.
PRICING = {
    "echo": (0.0, 0.0),
    "qwen-local": (0.0, 0.0),
    "nova-micro": (0.035, 0.14),
    "nova-lite": (0.06, 0.24),
    "nova-pro": (0.80, 3.20),
    "haiku-4.5": (1.0, 5.0),
    "sonnet-4.6": (3.0, 15.0),
    "sonnet-5": (2.20, 11.00),
    "luna": (0.22, 1.32),
}


def estimate_cost_usd(model_key, in_tokens, out_tokens):
    per_in, per_out = PRICING.get(model_key, (None, None))
    if per_in is None:
        return None, "unknown-model"
    return (in_tokens * per_in + out_tokens * per_out) / 1_000_000, "estimate"


def compute(answer, sources, info, latencies, evidence_text="",
            model_key="echo", reasoning=None, review=None):
    """Deterministic metric bundle. Never stores document/answer text.

    reasoning is the metadata-only record from query_planner (or None
    for the frozen baseline); review is the metadata-only record from
    answer_reviewer (or None when review is off). Only whitelisted
    scalar fields land here.
    """
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
    reasoning = reasoning or {}
    planner_latency = reasoning.get("planner_latency_ms")
    try:
        planner_latency = None if planner_latency is None \
            else max(0, int(planner_latency))
    except (TypeError, ValueError):
        planner_latency = None
    review = review or {}
    reviewer_latency = review.get("review_latency_ms")
    try:
        reviewer_latency = None if reviewer_latency is None \
            else max(0, int(reviewer_latency))
    except (TypeError, ValueError):
        reviewer_latency = None
    # Reviewer cost: estimated from reviewer token metadata when both
    # the model key is priced and token counts are present; otherwise
    # None (unknown-model or unavailable). Never blocks the suite.
    review_cost = review.get("review_cost_usd")
    if review_cost is None:
        try:
            rin = review.get("review_tokens_in")
            rout = review.get("review_tokens_out")
            rmodel = review.get("review_model") or ""
            rkey = None
            low = str(rmodel).lower()
            for candidate in PRICING:
                if candidate != "echo" and candidate != "qwen-local" \
                        and candidate in low:
                    rkey = candidate
                    break
            if rkey is not None and isinstance(rin, int) \
                    and isinstance(rout, int):
                review_cost, _how = estimate_cost_usd(rkey, rin, rout)
            else:
                review_cost = None
        except (TypeError, ValueError):
            review_cost = None
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
        "planner_invoked": bool(reasoning.get("reasoning_used", False)),
        "planner_valid": (False if reasoning.get("planner_failure_category")
                          else (True if reasoning.get("reasoning_used")
                                else None)),
        "planner_failure_category": reasoning.get("planner_failure_category"),
        "escalation_reason": reasoning.get("escalation_reason"),
        "planner_latency_ms": planner_latency,
        "planner_cached": bool(reasoning.get("planner_cached", False)),
        # Phase 5E reviewer dimensions (metadata only; None when off).
        # guard_before/after come from the review record (counts only).
        "review_used": bool(review.get("review_used", False)),
        "review_verdict": review.get("review_verdict"),
        "review_trigger": review.get("review_trigger"),
        "review_failure_category": review.get("review_failure_category"),
        "repair_attempted": bool(review.get("repair_attempted", False)),
        "repair_succeeded": bool(review.get("repair_succeeded", False)),
        "reviewer_latency_ms": reviewer_latency,
        "review_tokens_in": review.get("review_tokens_in"),
        "review_tokens_out": review.get("review_tokens_out"),
        "review_cost_usd": review_cost,
        "guard_before_flagged": review.get("guard_before_flagged"),
        "guard_before_total": review.get("guard_before_total"),
        "guard_after_flagged": guard["count"],
        "guard_after_total": guard["total"],
        "review_valid": (False if review.get("review_failure_category")
                         else (True if review.get("review_used")
                               else None)),
    }


def words(text):
    return set(re.findall(r"[a-z']+", (text or "").lower()))
