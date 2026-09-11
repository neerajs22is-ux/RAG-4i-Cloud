# Session handoff — 2026-09-11 (model evaluation + short benchmark)

Authoritative continuation point for the next chat. No secrets in this file.

## A. Project state

- Latest committed production baseline: `741da54` — Phase 5E (5.0→5A→5B→5C→5D→5E complete; 502 tests + 2 skipped at that commit).
- Current branch: `main` (verify with `git status` on resume).
- Uncommitted benchmark work (this session, to be committed as docs/config/scaffolding only): `tests/benchmarks/{harness.py,metrics.py}` (modified), plus new `LIVE_ARMS.md`, `cases_m.json`, `isolations.py`, `mantle_live.py`, `run_final.py`, `run_short.py`, `scenarios_v1.json`, `short_run.json`, `tests/test_live_arms.py`, `tests/test_final_benchmark.py`, `tests/test_mantle_live.py`, and this handoff doc.
- Production code untouched: `backend.py`, `workflows.py`, `query_planner.py`, `answer_reviewer.py`, `app.py`, `config.py`, providers, `deploy/`, `.env` files. Only benchmark scaffolding + this doc change.
- Benchmark result artifacts under `tests/benchmarks/results/20260911-short01/` remain **uncommitted** (15 files: traces, control, calibration packet, spend ledger, summary, teardown, config snapshot).

## B. AWS environment

- Account `305740358559`. Benchmark region **ap-south-1** (Mumbai); S3 corpus stays ap-south-2.
- EC2 `rag4i-admin` (`i-07dea54f4144b8fd0`, t3.micro, us-east-1c; last public IP `3.90.188.164`, drifts on stop/start). Role `arn:aws:iam::305740358559:role/rag4i-ec2-s3-readonly` (assumed-role session suffix `/i-07dea54f4144b8fd0`).
- RDS stays private us-east-1 (untouched; benchmark uses in-memory DictStore, never touches RDS).
- Role policies now on the EC2 role: `s3-read-documents` (original, untouched) + `bedrock-benchmark-converse` (InvokeModel/Streaming on six us-east-1 FM ARNs — unused after the Mantle pivot) + `bedrock-mantle-smoke-default-project` (CreateInference on us-east-1 default project) + `bedrock-mantle-short-term-bearer-only` (CallWithBearerToken SHORT_TERM) + ap-south-1 project grant for `bedrock-mantle:CreateInference` on `arn:aws:bedrock-mantle:ap-south-1:305740358559:project/default` (added manually to unblock Mumbai; simulation + live calls confirm `allowed`).
- Mantle default project (ap-south-1): `arn:aws:bedrock-mantle:ap-south-1:305740358559:project/default`.
- Access pattern that works: on-box SSH via EIC-pushed throwaway keys (60s validity; auto-refresh implemented in `mantle_live.SSHTransport`), SigV4 instance-role creds from IMDSv2. Root cannot AssumeRole the EC2 role; no SSH key persists (removed at teardown); no SSM channel (account lacks default host management).

## C. Working inference path

- **Bedrock Runtime Converse is dead for this account**: `ValidationException: Operation not allowed` in us-east-1 AND ap-south-1 across all vendors (account-level entitlement wall, not IAM — role simulates allowed on FM ARNs; quotas read 0.0).
- **Bedrock Mantle (SigV4, EC2 role) is the proven path**: `https://bedrock-mantle.ap-south-1.api.aws/v1`, default project, no API keys (bearer-token route abandoned after proving it needs a second action; SigV4 needs only CreateInference).
- Proven smokes (each exact-text, HTTP 200): Qwen3-32B `RAG4I_MUMBAI_MANTLE_OK` (28/13/41 tok); Qwen3-235B, Mistral Large 3, Devstral 2 `RAG4I_MODEL_SMOKE_OK`; GPT-OSS-120B via Mantle (us-east-1, earlier turn). Each sub-cent; total program ≈ **$0.032** (ledger $0.0199/124 calls + ~$0.012 pre-ledger).
- Prior notes calling the old short_run.json/LIVE_ARMS.md Haiku/Sonnet/Luna matrix current are **superseded** — those files describe an abandoned track (Converse path dead, Luna denied). Left in place for history; do not use without revision.

## D. Candidate conclusions (exploratory, 1 rep, no judge — not production verdicts)

- Working in Mumbai: Qwen3-32B, Qwen3-235B (true text, ap-south-1 only), GPT-OSS-120B, Mistral Large 3, Devstral 2.
- Excluded: Gemma 4 26B (absent in-region AND Mantle-only vs Converse planner — double block), Luna (permission_denied entitlement), Nova Micro (judge path unverified).
- Qwen3-235B true text model exists in ap-south-1 (`qwen.qwen3-235b-a22b-2507-v1:0`, ON_DEMAND, ACTIVE) and served successfully. Never substitute the VL variant.

## E. Short-benchmark matrix (as run)

- ARM1 QWEN FRONTIER: 235B / 32B / 235B. ARM2 OPEN VALUE: OSS-120B / 32B / 235B. ARM3 MISTRAL BALANCED: Large3 / 32B / 235B. ARM4 SPECIALIST: Devstral / 32B / 235B. ARM5 BUDGET: OSS-120B / 32B / 32B. Control: local Qwen3-1.7B + deterministic. Judge: not run / pending. No Gemma4/Luna/Anthropic/external-OpenAI/VL.

## F. Short results (`20260911-short01`, 12 scenarios ×1, traces in results dir)

- Control green (deterministic + live Qwen, 12/12). Hard gates passed (no leakage/scope/substitution/prose/evidence-divergence; secret scan clean). Zero retries/substitutions. Calibration packet 11/60, PENDING_HUMAN, no fabricated scores.
- Adjudicated: ARM1≈ARM3≈ARM4 6/8 answers + 3/3 clarifications correct; ARM2/ARM5 dragged by OSS empties. S28 extraction byte-identical all arms (deterministic harvester artifact, zero model signal). S04/S11/S16 are correct clarifications on all arms (not answer failures). S06/S18 misses are grader number/hyphen-format artifacts on correct answers.
- OSS-120B: empty on 6/12 generations (reasoning-trace vs truncation undetermined — needs raw-output follow-up).
- Planner 0/12 + reviewer 0/24 schema-valid: all structured calls exceed the frozen 5s timeout (observed 5.5–9s). E2E planner never escalated (gate: ordinary ×60); 0 repairs from ~20 triggered reviews with observed rubber-stamps.
- Latency: answers p50 ~6.1–6.7s, p95 ~6.8–16.8s; TTFT ≈ total (no incremental streaming observed — measurement limitation).
- Cost: ledgered $0.0199; $/grounded success ≈ $0.0003–0.0004/arm; monthly projections in traces (100u×20Q ≈ $2–10/mo).

## G. Interpretation (do NOT promote to production verdict)

Leaders: (1) Devstral 2 / Qwen32 / Qwen235, (2) Qwen235 / Qwen32 / Qwen235, (3) Large 3 / Qwen32 / Qwen235. OSS is a wildcard pending empty-output diagnosis.

## H. Next work (in order)

1. Diagnose OSS empties (raw-output capture; think-block vs truncation).
2. Address helper timeout (5s frozen vs 5.5–9s observed) as a separate benchmark revision with revalidation — never silently.
3. Small targeted validation (planner × few, reviewer × few, OSS × few), then FULL 25-doc benchmark on finalists.
4. Frozen: corpus/scenarios/gold, retrieval (pgvector-first; HNSW ~15–25k; hybrid only on exact-term evidence), prompts/k/threshold/embeddings. No OpenSearch/HNSW, no deploy yet.

## I. Artifacts

`tests/benchmarks/results/20260911-short01/`: trace-{answer,planner,reviewer,arm1..arm5}.jsonl, control.json, calibration_packet.json, spend.json, summary.json, teardown.txt, config_snapshot.json. No secrets (scanned). Uncommitted by policy.
