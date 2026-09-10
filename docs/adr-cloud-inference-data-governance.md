# ADR: Cloud Inference Data Governance (Phase 5.0 — decision foundation)

Status: **analysis + recommendation, no provider chosen yet** (final call
is PENDING BENCHMARK per `docs/PHASE_5_DECISIONS.md`). Researched
2026-09-10; prices/regions drift — re-verify at 5A kickoff.

## 1. What legal text leaves the current environment

Today nothing leaves: LM Studio runs on the operator PC; EC2↔PC traffic
is the reverse tunnel. Under 5A, each answer call would send to the
inference provider:
- the user question,
- retrieved chunk text (bounded: ~5 chunks × ~1000 chars normal path;
  ≤24 capped chunks for summaries),
- filenames/page labels (citation context).

Never sent: `.env`/DB credentials (provider is invoked with IAM role
credentials server-side; secrets stay out of prompts — same rule as
logs today), S3 bucket contents beyond retrieved chunks, telemetry
(which stays metadata-only).

## 2. Where inference occurs (primary candidate: AWS Bedrock)

Bedrock runs inference inside AWS-owned model deployment accounts;
model vendors have no access to those accounts, logs, prompts, or
completions. Our call path would be EC2 (us-east-1) → Bedrock
bedrock-runtime endpoint (us-east-1, in-region profile) over TLS, IAM-role
authenticated (no long-lived keys; boto3 already a dependency).

## 3. Region / data-residency implications (our split estate)

- EC2 + RDS: **us-east-1**. S3 documents at rest: **ap-south-2**.
- In-region Bedrock inference in us-east-1 keeps prompts/outputs inside
  us-east-1 (strict single-region profile; deny Global/Geo profiles by
  SCP). No cross-region routing, no new residency surface beyond what
  the pilot already accepts (operator PC in India reads us-east-1 app).
- Residency question for POLICY (not engineering): retrieved text of
  documents stored in ap-south-2 would be processed in us-east-1.
  APAC geographic profiles exist but may not cover every candidate
  model; Mumbai (ap-south-2) Bedrock availability must be confirmed at
  5A kickoff if policy requires India-region inference.
- Cross-region (Geo/Global) inference profiles: REJECTED for the pilot
  (data may land in any commercial region; CloudTrail won't name the
  processing region). Enforce with SCP deny on `global.*` profiles.

## 4. Retention / training policy (the decisive filter)

- Bedrock baseline: inputs/outputs are NOT shared with model vendors
  and are NOT used to train foundation models (dedicated deployment
  accounts). Abuse-detection retention may apply depending on mode.
- Zero-data-retention (ZDR, `data_retention_mode: none`) is enforceable
  per account/project, including via SCP — REQUIRED for 5A.
- ⚠️ Newest Anthropic models (Claude Fable 5/5.1) REQUIRE 30-day
  AWS-side retention for human review as a condition of access.
  **Excluded from the pilot shortlist for this reason.**
- Default (non-ZDR) mode may retain for abuse detection — another
  reason ZDR + audit is the acceptance gate, not an afterthought.

## 5. Candidate comparison (bedrock-runtime, Converse API)

| Provider path | Training on our data | Retention control | Streaming | Structured output | Notes |
|---|---|---|---|---|---|
| Bedrock (Anthropic Claude) | No (dedicated accounts) | ZDR enforceable; avoid Fable-class | ConverseStream ✓ | Forced-tool `with_structured_output` ✓ today; native `outputConfig` schema on Haiku 4.5/Sonnet 4.5/Opus 4.5+ (not yet in langchain-aws — open issue) | Primary candidate |
| Bedrock (Amazon Nova) | No (Amazon FM privacy terms) | Same ZDR story | ✓ | Model-dependent; verify per-model | Cost control arm |
| Bedrock (Meta Llama) | No | Same | ✓ | NO native structured outputs; forced-tool unreliable on open weights | Planner-ineligible; answer-only fallback at most |
| Direct vendor API (Anthropic/OpenAI) | Contract-dependent; data leaves AWS boundary | Weaker enforceability | ✓ | ✓ | REJECTED for pilot: new trust boundary + key management |

## 6. Model availability / cold starts / ops

- Serverless on-demand: no cold starts in the provisioned sense; first-token latency model-dependent (Haiku/Nova Micro fastest; Sonnet higher). Throughput quotas per region; Geo profiles raise limits — we accept in-region quotas instead (pilot volume is tiny).
- Availability risk: single-region pinning means a us-east-1 Bedrock outage halts cloud answers → the LM-Studio fallback flag (5A) exists for exactly this, defaulting to honest-error unless enabled.
- Ops: model access EULA per model in console; IAM `bedrock:InvokeModel*` on the EC2 role; new `langchain-aws` dependency (NOT in requirements today — 5A adds it, pinned in the regenerated lock file).

## 7. Rollback

Provider switch is config-only (`LLM_PROVIDER=lmstudio`), same as every
existing provider seam. No schema or data migration depends on the
answer model (vectors/embeddings untouched), so rollback is a restart,
not a migration. The 4.12 Qwen path stays runnable via the tunnel.

## 8. Cost model (Aug–Sep 2026 figures, volatile — re-verify at 5A)

On-demand $/1M tokens (input/output, us-east-1): Nova Micro 0.035/0.14;
Nova Lite 0.06/0.24; Nova Pro 0.80/3.20; Haiku 4.5 1/5; Sonnet 4.6 3/15;
Opus 4.8 5/25. Batch −50%; prompt caching up to −90% (stable system
prompt + tools are cacheable — our frozen templates help here).
Illustrative (NOT a gate): a 4k-in/0.5k-out answer on Sonnet 4.6 ≈
$0.02; on Haiku 4.5 ≈ $0.0065; 1k pilot answers ≈ $20 / $6.50.
Acceptance gates stay methodology-based (see `PHASE_5_ACCEPTANCE.md`).

## 9. Recommendation + 5A decision criteria

**Recommend Bedrock, us-east-1, in-region profile, ZDR enforced** as the
5A implementation target — best combination of no-training boundary,
enforceable retention, IAM auth inside our existing account, and a thin
LangChain seam (`ChatBedrockConverse`). Do NOT finalize the answer
model until the benchmark runs: Sonnet 4.6 vs Haiku 4.5 (+ Nova Pro
cost control). Go/no-go criteria: acceptance gates in
`docs/PHASE_5_ACCEPTANCE.md`; residency sign-off if policy requires
India-region processing; confirmed ap-south-2/us-east-1 availability of
the chosen model ID at kickoff.
