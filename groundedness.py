"""Bounded groundedness verification (Step 6).

Pipeline position (additive, flag-gated, default off):

    retrieve -> rerank -> answer -> CHECK -> [ONE repair] -> RE-CHECK -> final

- At most 1 groundedness check + 1 repair + 1 re-check, then STOP.
- The checker is DETERMINISTIC (no model call): claim/number/qualifier
  support against the actual retrieved chunks. Cheap, provider-neutral,
  offline-testable. Mirrors the NVIDIA RAG Blueprint verification gate
  and Controllable-RAG-Agent grounded-check + bounded-retry ideas,
  reusing this repo's 5E conventions (validated findings, metadata-only
  telemetry, original preserved on failure).
- Repair reuses the EXISTING answer path (same provider abstraction,
  frozen grounding prompt untouched): a constrained repair call over
  CURRENT ANSWER + RETRIEVED EVIDENCE + CHECKER FINDINGS. Never a fresh
  unrestricted generation, never new retrieval, never new evidence.
- Verifier/reviewer text NEVER becomes document evidence: repair context
  is the original sources only; findings are instructions, not facts.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Hard bounds (structural: the orchestrator has no loop).
MAX_CHECKS = 2
MAX_REPAIRS = 1

# Deterministic checker tuning (conservative: avoid false repairs).
MIN_SENTENCE_TOKENS = 4
SUPPORT_OVERLAP = 0.25
HIGH_OVERLAP = 0.5
MAX_FINDINGS_EACH = 4
MAX_SENTENCE_CHARS = 300
MAX_EVIDENCE_CHARS = 6000
MAX_ANSWER_CHARS = 2000

QUALIFIER_CUES = frozenset({
    "not", "no", "never", "none", "only", "except", "unless", "cannot",
    "can't", "cannot", "must", "required", "always", "without",
})

# Negation cues for dropped-negation detection (answer states as fact
# what evidence negates). Checked at evidence-sentence level with a
# shared-phrase requirement so unrelated "not"s elsewhere don't fire.
NEGATION_CUES = frozenset({"not", "no", "never", "cannot", "can't"})

_UNIT = (r"%|percent|months?|years?|days?|weeks?|hours?|minutes?|"
         r"seconds?|dollars?|usd|inr|rs\.?|kg|km|mm|cm|gb|mb|tbp?")
_NUMBER_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"                       # 2024-03-15
    r"|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\s+\d{2,4}\b"                          # 15 March 2024
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"\s+\d{4}\b"                                  # March 2024
    r"|\b\d+(?:\.\d+)?\s*(?:" + _UNIT + r")", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

REPAIR_TEMPLATE = (
    "Fix ONLY the listed grounding problems in the draft answer, using "
    "ONLY the evidence below. Keep every correct sentence byte-identical. "
    "Do not add facts, do not use outside knowledge, do not change "
    "citations or formatting beyond the fix.\n"
    "Findings:\n{findings}\n"
    "Draft answer:\n{answer}")


def _tokens(text):
    return _WORD_RE.findall(str(text or "").lower())


def _content_tokens(text):
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "is",
            "are", "was", "were", "it", "its", "this", "that", "for",
            "with", "as", "by", "at", "be", "from", "which", "you"}
    return [t for t in _tokens(text) if t not in stop and len(t) > 2]


def _normalize_number(tok):
    t = str(tok or "").lower().replace(",", "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def extract_numbers(text):
    """Normalized numeric/date/unit tokens (deduped, order kept)."""
    out = []
    for m in _NUMBER_RE.findall(text or ""):
        n = _normalize_number(m)
        if n and n not in out:
            out.append(n)
    return out


def split_sentences(text):
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return []
    return [s.strip() for s in _SENT_RE.split(clean) if s.strip()]


def _overlap_ratio(sent_tokens, chunk_tokens):
    if not sent_tokens:
        return 0.0
    cset = set(chunk_tokens)
    return sum(1 for t in sent_tokens if t in cset) / len(sent_tokens)


def check_groundedness(question, answer, evidence_contents):
    """Deterministic groundedness check.

    evidence_contents: iterable of chunk strings (actual retrieved text).
    Returns findings dict with grounded bool + categorized lists. Never
    raises on odd input (empty findings = grounded).
    """
    findings = {"grounded": True, "unsupported_claims": [],
                "contradictions": [], "missing_qualifiers": [],
                "numeric_mismatches": []}

    def _add(key, text):
        items = findings[key]
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        text = text[:MAX_SENTENCE_CHARS]
        if text and text not in items and \
                len(items) < MAX_FINDINGS_EACH:
            items.append(text)
            findings["grounded"] = False

    chunks = [re.sub(r"\s+", " ", str(c or "")).strip()
              for c in (evidence_contents or [])]
    chunks = [c for c in chunks if c]
    answer = re.sub(r"\s+", " ", str(answer or "")).strip()
    if not answer or not chunks:
        return findings
    chunk_toks = [_content_tokens(c) for c in chunks]
    evidence_blob = " ".join(chunks).lower()
    ev_sentences = []
    for c in chunks:
        for s in split_sentences(c):
            ev_sentences.append((s, set(_content_tokens(s)),
                                 s.lower()))

    def _has_cue(text_low, cue):
        return re.search(r"\b" + re.escape(cue) + r"\b", text_low) \
            is not None

    for sent in split_sentences(answer):
        stoks = _content_tokens(sent)
        # Length gate uses raw words (numbers count): short numeric claims
        # such as "The escalation is 7 percent." must still be checked.
        if len(_tokens(sent)) < MIN_SENTENCE_TOKENS:
            continue
        scored = sorted(((_overlap_ratio(stoks, ct), i)
                         for i, ct in enumerate(chunk_toks)),
                        reverse=True)
        best_ratio, best_i = scored[0]
        best_chunk = chunks[best_i].lower()
        sent_numbers = extract_numbers(sent)
        sent_low = sent.lower()
        quals = sorted({q for q in QUALIFIER_CUES
                        if _has_cue(sent_low, q)})
        # Dropped negation: an evidence sentence sharing a phrase with
        # this sentence negates it, but the answer states it plainly.
        sent_set = set(stoks)
        for ev_sent, ev_set, ev_low in ev_sentences:
            if len(sent_set & ev_set) >= 2:
                for q in sorted(NEGATION_CUES):
                    if _has_cue(ev_low, q) and not _has_cue(sent_low, q):
                        _add("missing_qualifiers",
                             f"{sent} [evidence negates shared phrase "
                             f"({q}): {ev_sent[:120]}]")
                        break
                else:
                    continue
                break
        if best_ratio >= HIGH_OVERLAP:
            # Strongly tied to one chunk: numbers/qualifiers must match it.
            for num in sent_numbers:
                if num not in best_chunk and num not in evidence_blob:
                    _add("contradictions",
                         f"{sent} [number not in supporting chunk: {num}]")
            for q in quals:
                if not _has_cue(best_chunk, q):
                    _add("contradictions",
                         f"{sent} [qualifier not in supporting chunk: {q}]")
            continue
        if best_ratio < SUPPORT_OVERLAP:
            _add("unsupported_claims", sent)
            continue
        # Weakly supported: numbers must still hold somewhere; dropped
        # qualifiers are flagged rather than trusted.
        for num in sent_numbers:
            if num not in evidence_blob:
                _add("numeric_mismatches",
                     f"{sent} [number not in evidence: {num}]")
        for q in quals:
            if not _has_cue(best_chunk, q):
                _add("missing_qualifiers",
                     f"{sent} [qualifier not in supporting chunk: {q}]")
    return findings


def build_evidence_block(sources, max_chars=MAX_EVIDENCE_CHARS):
    """Render sources as labeled excerpts (provenance preserved)."""
    lines = []
    used = 0
    for s in (sources or []):
        if not isinstance(s, dict):
            continue
        content = re.sub(r"\s+", " ", str(s.get("content") or "")).strip()
        if not content:
            continue
        label = str(s.get("file_name") or "unknown")[:80]
        if s.get("page") is not None:
            label += " p. " + str(s.get("page"))
        line = "[" + label + "] " + content
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n\n".join(lines)


def build_repair_prompt(question, answer, findings, evidence_block):
    items = []
    for key in ("contradictions", "numeric_mismatches",
                "missing_qualifiers", "unsupported_claims"):
        for f in (findings or {}).get(key) or []:
            items.append("- " + str(f)[:MAX_SENTENCE_CHARS])
    prompt = REPAIR_TEMPLATE.format(
        findings="\n".join(items) or "(no details)",
        answer=str(answer or "")[:MAX_ANSWER_CHARS])
    return prompt, evidence_block


def groundedness_enabled(config=None):
    if config is None:
        return False
    return str(getattr(config, "groundedness_enabled", "0") or "0"
               ).lower() in ("1", "true", "yes", "on")


def maybe_verify_response(question, answer, sources, config=None):
    """App-level seam: verified answer text, never raises.

    Returns the input answer unchanged when the flag is off, when there
    is no retrieved evidence to ground against, or when any stage fails
    (fail open, like the 5E insufficient path).
    """
    try:
        if not groundedness_enabled(config):
            return answer
        if not [s for s in (sources or [])
                if isinstance(s, dict) and s.get("content")]:
            return answer
        final, _meta = verify_response(question, answer, sources,
                                       config=config)
        return final if isinstance(final, str) and final else answer
    except Exception:
        logger.warning("Groundedness gate failed open.", exc_info=True)
        return answer


def verify_response(question, answer, sources, config=None,
                    llm_provider=None):
    """Check -> at most ONE repair -> re-check -> STOP.

    Returns (final_answer, meta). Disabled (default) returns the input
    answer untouched with groundedness_used False. Failures preserve
    the original answer (existing safe behavior); meta records counts
    so callers can assert MAX_REPAIRS is never exceeded.
    """
    meta = {"groundedness_used": False, "grounded": None,
            "checks": 0, "repairs": 0, "repaired": False,
            "failure": None, "latency_ms": None}
    if not groundedness_enabled(config):
        return answer, meta
    import time
    t0 = time.monotonic()

    def _evidence():
        return [s.get("content") for s in (sources or [])
                if isinstance(s, dict) and s.get("content")]

    try:
        meta["groundedness_used"] = True
        findings = check_groundedness(question, answer, _evidence())
        meta["checks"] = 1
        if findings.get("grounded"):
            meta["grounded"] = True
            return answer, meta
        # ONE targeted repair via the existing provider abstraction.
        provider = llm_provider
        if provider is None:
            try:
                from llm_provider import get_llm_provider
                cfg = config
                if cfg is None:
                    from config import get_config
                    cfg = get_config()
                provider = get_llm_provider(cfg)
            except Exception as e:
                logger.warning("Groundedness repair unavailable: %s", e)
                meta.update(grounded=False, failure="unavailable")
                return answer, meta
        block = build_evidence_block(sources)
        if not block:
            meta.update(grounded=False, failure="empty-evidence")
            return answer, meta
        prompt, context = build_repair_prompt(question, answer, findings,
                                              block)
        meta["repairs"] = 1
        try:
            repaired = provider.generate(context, question or "", prompt)
        except Exception as e:
            logger.warning("Groundedness repair failed: %s", e)
            meta.update(grounded=False, failure="unavailable")
            return answer, meta
        if not isinstance(repaired, str) or not repaired.strip():
            meta.update(grounded=False, failure="empty-repair")
            return answer, meta
        # Exactly ONE re-check, then stop regardless of outcome.
        recheck = check_groundedness(question, repaired, _evidence())
        meta["checks"] = 2
        if recheck.get("grounded"):
            meta.update(grounded=True, repaired=True)
            return repaired, meta
        meta.update(grounded=False, repaired=False, failure="still-ungrounded")
        return answer, meta
    finally:
        try:
            meta["latency_ms"] = max(
                0, int((time.monotonic() - t0) * 1000))
        except Exception:
            pass
