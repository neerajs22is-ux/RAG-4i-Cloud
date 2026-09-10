# Phase 5 Acceptance Gates (methodology first — read before 5A)

No "feels better" anywhere below. Every numeric gate is either decided
from a hard property (isolation, parity, round-trips) or marked
**pending-calibration** with the exact calibration method. Machine
form lives in `tests/benchmarks/thresholds.json`.

## Calibration method (all pending gates)

1. Freeze `tests/benchmarks/cases.json` version.
2. Run `run_all` for Baseline (4.12/Qwen or Echo) on the pilot box;
   record per-case metrics (support, guard counts, latencies, tokens).
3. Run the staged config on the same version; compare paired per-case.
4. Set the numeric gate from the baseline distribution (e.g. p95) plus
   an explicit headroom note, and flip status to decided. Until then
   the gate blocks on missing calibration, not on vibes.

## 5A (cloud answer model)

- Answer quality: benchmark pass rate == 1.0 (decided) — no frozen
  behaviour may regress.
- Citation/grounding: guard-flagged share on grounded answers ≤ 4.12
  baseline share on the same set (pending-calibration).
- Latency: p95 total ≤ calibrated budget (pending-calibration).
- Cost: mean $/answer ≤ calibrated budget (pending-calibration).
- Failure rate: errors/empties ≤ calibrated floor (pending-calibration).

## 5B (scope/identity primitives)

- Plan validity: schema-valid first-try rate ≥ calibrated bar.
- Escalation rate (validator fallback): ≤ calibrated bar.
- Validation failure rate: tracked per failure class (schema, target,
  budget); any single class dominating triggers planner-prompt work.
- Latency overhead p95: ≤ calibrated ms vs planner-off on same cases.
- Improvement: paired benchmark wins on retrieval/answer dimensions
  (more cited answers, fewer partials) with zero regressions.

## 5C (session uploads)

- Zero cross-session leakage: 0 retrieved session chunks across the
  isolation probe set (decided, hard gate).
- Upload success, indexing correctness (chunk counts = deterministic
  expectation from the fixture), deletion/detach correctness (post-
  delete retrieval returns zero session rows): all == 1.0 (decided).

## 5D (reasoning planner)

- `REASONING_ENABLED=0` reproduces the 4.12 benchmark exactly
  (decided, diff-zero gate).
- Ships only with measured wins on predefined dimensions (comparison
  completeness, extraction recall, ambiguous-escalation precision)
  large enough to justify added latency + cost (pending-calibration,
  reported as a benchmark report, not a feeling).

## 5E (answer reviewer / bounded repair)

- `REVIEW_ENABLED=0` reproduces the pre-5E benchmark exactly
  (decided, diff-zero gate; proven by `reviewer="deterministic"`
  parity in `tests/test_phase5e.py` + harness).
- Zero cross-session leakage including gap-fill retrieval: 0 session
  chunks across the isolation probe set (decided, hard gate).
- Exactly one repair cycle, never loops (decided, structural).
- Reviewer never generates the final answer (decided, structural:
  no `generate`/`generate_stream` on reviewer classes).
- Citation coverage before/after, material-gap correction rate,
  false repair rate, reviewer schema reliability, reviewer latency,
  reviewer token/cost overhead, net answer-quality delta: all
  measured per case (harness `reviewer=` + metrics `review_*` +
  `guard_before/after`; synthetic data only) and reported as a
  benchmark report (pending-calibration). The gate question is: does
  the measurable grounding improvement justify the reviewer cost?
- `thresholds.json` unchanged by 5E (no silent gate changes).
