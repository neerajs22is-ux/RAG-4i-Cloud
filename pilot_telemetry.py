"""Pilot observability: latency, stage timings, feedback, telemetry (Phase 4.11).

Pure, deterministic, no LLM, no secrets, no document text in telemetry.
All helpers are unit-testable without Streamlit.

Telemetry privacy contract (see docs/adr-chat-history.md):
  - Events carry metadata ONLY (counts, labels, timings, feedback).
  - Never store questions, answers, or document text server-side.
  - The in-memory buffer is ephemeral (per session) and replaceable:
    future production telemetry can swap PilotTelemetryStore without
    touching callers.
"""

import time
from datetime import datetime, timezone

# Deterministic negative-feedback categories (no LLM, fixed set).
FEEDBACK_CATEGORIES = (
    "Didn't answer my question",
    "Information was missing",
    "Sources seemed wrong",
    "Answer wasn't clear",
    "Other",
)

_MAX_EVENTS = 200


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_latency(total_ms) -> str:
    """'Answered in 4.2s'. Real elapsed only; never called 'confidence'."""
    try:
        seconds = float(total_ms) / 1000.0
    except (TypeError, ValueError):
        return "Answered"
    if seconds < 0:
        seconds = 0.0
    return f"Answered in {seconds:.1f}s"


def format_answer_meta(total_ms, source_count) -> str:
    """'Answered in 4.2s · 3 sources' (singular-aware)."""
    base = format_latency(total_ms)
    try:
        n = int(source_count)
    except (TypeError, ValueError):
        return base
    noun = "source" if n == 1 else "sources"
    return f"{base} · {n} {noun}"


def validate_feedback_category(category) -> str | None:
    """Return the category if it is in the fixed set, else None."""
    if category in FEEDBACK_CATEGORIES:
        return category
    return None


def top_retrieval_score(sources) -> float | None:
    """Max numeric score, or None (reuses stored scores; no recompute)."""
    best = None
    for s in sources or []:
        v = (s or {}).get("score")
        if isinstance(v, (int, float)):
            f = float(v)
            if best is None or f > best:
                best = f
    return best


# Reasoning telemetry keys (Phase 5D). Only ever attached when a
# reasoning record is explicitly supplied; existing events keep their
# exact shape.
REASONING_KEYS = ("reasoning_used", "reasoning_provider",
                  "reasoning_model", "planner_latency_ms",
                  "planner_failure_category", "escalation_reason",
                  "planner_cached", "planner_tokens_in",
                  "planner_tokens_out", "reasoning_cost_usd", "workflow")

# Review telemetry keys (Phase 5E). Only ever attached when a review
# record is explicitly supplied; existing events keep their exact
# shape. Metadata only: counts, labels, timings, never questions,
# answers, chunks, raw model output, or secrets.
REVIEW_KEYS = ("review_used", "review_provider", "review_model",
               "review_latency_ms", "review_verdict",
               "review_failure_category", "review_trigger",
               "repair_attempted", "repair_succeeded",
               "review_tokens_in", "review_tokens_out",
               "review_cost_usd", "guard_before_flagged",
               "guard_before_total", "guard_after_flagged",
               "guard_after_total")


def build_telemetry_event(*, message_id=None, answer_label=None,
                          source_count=0, retrieval_strength=None,
                          total_ms=None, timings=None,
                          feedback=None, feedback_category=None,
                          timestamp=None, reasoning=None,
                          review=None) -> dict:
    """Metadata-only event. Never includes question/answer/document text.

    reasoning is an optional metadata-only record (see query_planner):
    whitelisted keys are merged when supplied, otherwise omitted so
    existing events keep their exact shape. review is the same for
    the Phase 5E reviewer record (see answer_reviewer).
    """
    event = {
        "timestamp": timestamp or utc_now_iso(),
        "message_id": message_id,
        "answer_label": answer_label,
        "source_count": int(source_count or 0),
        "retrieval_strength": retrieval_strength,
        "total_ms": int(total_ms) if total_ms is not None else None,
        "retrieval_ms": (timings or {}).get("retrieval_ms"),
        "support_ms": (timings or {}).get("support_ms"),
        "preparation_ms": (timings or {}).get("preparation_ms"),
        "generation_ms": (timings or {}).get("generation_ms"),
        "feedback": feedback,
        "feedback_category": validate_feedback_category(feedback_category),
    }
    if reasoning:
        for key in REASONING_KEYS:
            if key in reasoning:
                event[key] = reasoning[key]
    if review:
        for key in REVIEW_KEYS:
            if key in review:
                event[key] = review[key]
    return event


class PilotTelemetryStore:
    """Ephemeral in-memory buffer (no disk, no server DB). Replaceable."""

    def __init__(self, max_events: int = _MAX_EVENTS):
        self._events = []
        self._max = max(1, int(max_events))

    def record(self, event: dict) -> dict:
        """Append a metadata-only event; drop oldest past capacity."""
        clean = {k: event.get(k) for k in (
            "timestamp", "message_id", "answer_label", "source_count",
            "retrieval_strength", "total_ms", "retrieval_ms", "support_ms",
            "preparation_ms", "generation_ms", "feedback", "feedback_category")}
        for k in REASONING_KEYS:
            if k in event:
                clean[k] = event[k]
        for k in REVIEW_KEYS:
            if k in event:
                clean[k] = event[k]
        if not clean.get("timestamp"):
            clean["timestamp"] = utc_now_iso()
        self._events.append(clean)
        if len(self._events) > self._max:
            self._events = self._events[-self._max:]
        return clean

    def record_feedback(self, message_id, feedback: str,
                        category: str | None = None) -> dict | None:
        """Attach feedback to the matching event (idempotent, no dupes)."""
        for ev in reversed(self._events):
            if ev.get("message_id") == message_id:
                ev["feedback"] = feedback
                ev["feedback_category"] = validate_feedback_category(category)
                return ev
        return None

    def events(self):
        return list(self._events)

    def __len__(self):
        return len(self._events)


# --- Feedback state (rerun-safe, dict-only so session_state compatible) --- #

def feedback_key(seq) -> str:
    return f"feedback_{seq}"


def get_feedback(state, seq):
    """Stored feedback for a message seq: {"value": +/-1|None, "category": str|None}."""
    fb = (state or {}).get("feedback_by_seq", {})
    return fb.get(str(seq))


def record_feedback_state(state, seq, value, category=None) -> dict:
    """Idempotent write: same value twice is a no-op; never touches answer data.

    value: 1 (up), -1 (down), or None (clear). Category only kept for -1
    and only when in the fixed set.
    """
    if value not in (1, -1, None):
        raise ValueError(f"Unknown feedback value: {value!r}")
    if "feedback_by_seq" not in state or not isinstance(state["feedback_by_seq"], dict):
        state["feedback_by_seq"] = {}
    key = str(seq)
    if value is None:
        state["feedback_by_seq"].pop(key, None)
        return state
    entry = {"value": value,
             "category": validate_feedback_category(category) if value == -1 else None}
    # Idempotent: writing the identical entry changes nothing observable.
    state["feedback_by_seq"][key] = entry
    return state


def needs_feedback_category(state, seq) -> bool:
    """True when the user down-voted but has not picked a category yet."""
    entry = get_feedback(state, seq)
    return bool(entry and entry.get("value") == -1 and not entry.get("category"))


# --- Stage-timing helpers --- #

def ms_between(start: float, end: float) -> int:
    """Monotonic-clock delta in whole ms (never negative)."""
    try:
        return max(0, int((float(end) - float(start)) * 1000))
    except (TypeError, ValueError):
        return 0


def now() -> float:
    return time.monotonic()
