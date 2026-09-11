# Live Bedrock arm benchmark — runbook (preparation only, NOT YET RUN)

Frozen baseline `cases.json` (v1, 14 cases, 2 docs) is untouched.
M-corpus `cases_m.json` (v1, 26 cases, 25 synthetic docs) is the
realistic-complexity arm file. All content is synthetic.

## Arms (Answer / Reasoning / Reviewer; IDs are explicit, never silent)

- ARM A (Haiku stack): Haiku 4.5 / Haiku 4.5 / Haiku 4.5
  `us.anthropic.claude-haiku-4-5-20251001-v1:0` in us-east-1.
- ARM B (Sonnet stack): Sonnet 5 / Haiku 4.5 / Haiku 4.5
  answer `us.anthropic.claude-sonnet-5` (us-east-1), helpers Haiku 4.5.
- ARM C (India Luna stack): Luna / Luna / Luna
  `in.openai.gpt-5.6-luna` in ap-south-1 (India-geo routing).
- Control: deterministic baseline (`planner="deterministic"`,
  `reviewer="deterministic"`) and frozen Echo path, $0.

A BLOCKED arm (denied/missing access/quota) is reported as blocked;
no substitution is ever made.

## Pre-run verification (all must pass, else STOP)

1. `aws sts get-caller-identity --profile rag4i-dev` resolves.
2. Bedrock console: model access granted for Haiku 4.5, Sonnet 5,
   GPT-5.6 Luna (manual console check; no API exposes this).
3. Quotas: on-demand + cross-region inference quotas
   non-zero for the three models (currently observed 0.0 defaults).
4. `pip show langchain-aws` succeeds in the run venv (installed
   2026-09-10: langchain-aws 1.7.5; requires langchain-core 1.6.2 /
   langchain-protocol 0.0.19 alongside — resolver-mandated, nothing
   else changed. Full suite re-verified green after install).
5. Re-verify pricing below against the Bedrock pricing page; abort on
   silent drift (update `metrics.PRICING` loudly with a new date).

## Pricing assumptions (re-verified 2026-09-10; $/1M in/out)

haiku-4.5 1.00/5.00 · sonnet-5 2.20/11.00 (us.* profile rate, ~10%
over the $2.00/$10.00 global rate) · luna 0.22/1.32 (post July-2026
cut). Tokens are tiktoken-cl100k estimates recorded per case
(`token_method`); Converse usage metadata is NOT yet surfaced
through the provider chain (limitation — see below).

## Commands (paid; each arm × each corpus)

```bash
BENCHMARK_LIVE_BEDROCK=1 venv/bin/python -c "
from tests.benchmarks.harness import run_all, run_live_arm
import json
for arm in ['A','B','C']:
    for corpus in [None, 'tests/benchmarks/cases_m.json']:
        out = run_live_arm(arm, cases_path=corpus)
        tag = arm + ('-S' if corpus is None else '-M')
        json.dump(out, open('bench-%s.json' % tag, 'w'))
        print(tag, out['summary'])"
```

TTFT (not captured by the non-streaming harness path) is measured
separately per arm on 3 fixed queries via `stream_workflow_answer`
first-chunk timing; snippet kept with the run artifacts.

## Failsafe / retries

Per-case exceptions are recorded by `run_all` (status error), never
retried through another model. At most ONE same-model retry on
throttling responses only. Access-denied/quota/validation failures
mark the arm BLOCKED.

## Acceptance gates (frozen BEFORE execution; changes need explicit note)

- Baseline: 100% execution reliability; frozen behavior unchanged.
- Planner: schema-valid ≥ 95%; fallback rate recorded; zero invented
  documents; zero cross-session leakage.
- Reviewer: false-repair ≤ 10%; zero scope violations; zero
  reviewer-generated answer text.
- Answer: no material degradation vs frozen baseline; citation
  coverage + unsupported claims tracked per case.
- Sonnet promotion: grounding/citation improvement ≥ 3 points vs
  Haiku, or a clearly justified quality threshold.
- Luna promotion: within pre-declared tolerance of Haiku grounding,
  planner/reviewer gates pass, India residency stays valid.

## Spend cap

Call accounting (measured gate rates on Echo/deterministic doubles,
identical gating logic live: S = 14 questions → 1 planner + 4 reviewer
calls; M = 26 questions → 1 planner + 6 reviewer calls):

- Paid QUESTIONS per full run: (14 + 26) × 3 arms = **120**.
- Paid MODEL CALLS per full run: 120 answers + 6 planner + 30
  reviewer + 0–30 repair regenerations (bounded by reviewer count,
  at most one each) = **156–186**, plus 9 TTFT streaming probes
  (answer-only, no helpers).
- Retries: at most ONE same-model retry on throttling responses
  only. Absolute worst case (every call throttled once AND every
  reviewed question repairs): ≈ $4.29 for one full pass —
  inside the cap. Full-run repeats are NOT permitted under this
  cap; only error-status cases may be rerun, bounded by the actual
  error count (expected ~0; access/validation failures mark the arm
  BLOCKED instead).

Blended ≈ $0.010 (A) / $0.019 (B) / $0.0024 (C) per question:
expected ≈ **$1.20 total + ~$0.05 TTFT probes**. Hard ceiling
**$5.00**. No XL-scale live spend.

## Governance record per arm

A/B: inference in US-geo (us-east-1/2, us-west-2); requires the
pending US-geo amendment; Global never used. C: inference inside
India only (ap-south-1↔ap-south-2); clean under strict policy.
No arm is approved here — record, don't decide.

## Structured output (installed langchain-aws 1.7.5, verified 2026-09-10)

`with_structured_output` supports `function_calling` (default),
`json_schema` (native Converse outputConfig), and `prompt_prefill`.
The harness uses the default forced-tool path — unchanged behavior.
Native `json_schema` requires server-side per-model support
(documented e.g. for Claude 4.5+; Luna support is UNVERIFIED until
the live run exercises it — a failed arm is recorded BLOCKED, never
worked around by switching methods mid-run). Our provider constructs
cleanly against 1.7.5 (`model` alias resolves to `model_id`;
verified without inference, no app code change needed).

## Known limitations

1. Token/cost figures are tiktoken estimates, not Converse `usage`
   metadata (provider chain drops usage today).
2. TTFT measured outside the harness (runbook snippet).
3. DictStore is word-overlap, not MiniLM: retrieval-side behavior is
   controlled, so arms differ ONLY by model — intended, but
   retrieval-quality-at-scale is not what this benchmark measures.
4. S-corpus run from the operator box measures US arms fairly; arm C
   additionally carries us-east-1→ap-south-1 network leg — reported
   separately, not conflated with model latency.
