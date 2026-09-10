# Phase 5A: Bedrock Answer LLM — setup, failure behavior, rollback

Implementation behind `LLM_PROVIDER=bedrock`. Default (`lmstudio`)
behavior is unchanged. No model/region is chosen here: `ANSWER_MODEL_ID`
is required and has no silent default (evaluation candidates in
`docs/PHASE_5_MODEL_EVALUATION.md`).

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `lmstudio` | `lmstudio` (default, frozen path) \| `bedrock` (5A) \| anything else → loud `ValueError`, never a silent switch |
| `ANSWER_MODEL_ID` | _(empty)_ | Bedrock model ID or inference-profile ID, e.g. a `us.*` profile. **Required** for bedrock; empty fails fast with guidance |
| `BEDROCK_REGION` | `us-east-1` | In-region endpoint. Keep `us-east-1` (EC2/RDS region); do not use Geo/Global profiles for the pilot |
| `LLM_TEMPERATURE` | `0.0` | Passed through to Converse `temperature` unchanged |

There is deliberately **no key/secret setting**: credentials come only
from the boto3 default chain (environment → shared config → EC2
instance role). Nothing credential-like belongs in `.env`.

`LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` keep serving LM Studio only.

## AWS prerequisites (no infra change in 5A; needed before any live call)

1. Model access: Bedrock console → Model access → request access for the
   chosen model (EULA acceptance). Until granted, calls fail with
   access-denied guidance (see below).
2. IAM on the EC2 role (least privilege). `Converse` is governed by
   `bedrock:InvokeModel` and `ConverseStream` by
   `bedrock:InvokeModelWithResponseStream`. When `ANSWER_MODEL_ID` is
   an inference-profile ID, allow those actions on BOTH the profile
   ARN and the underlying foundation-model ARN(s) in each associated
   region (profile-only grants fail); add `bedrock:GetInferenceProfile`
   on the profile (required to run inference with a profile) and
   `bedrock:ListFoundationModels` for the readiness probe.
3. Retention/governance (REQUIRED, see `docs/adr-cloud-inference-data-governance.md`):
   zero-data-retention mode, in-region profile only, SCP deny on
   `global.*` inference profiles.
4. Runtime deps: `langchain-aws` (+ `boto3`, already required) must be
   installed; the provider raises a clean `pip install` message if not.
   Lock-file regeneration per `docs/PHASE_5_BUILD.md` happens pre-deploy.

## Failure behavior (all surfaced, never silent)

| Symptom | Cause | User sees |
|---|---|---|
| No credentials in chain | No env/role credentials | Bedrock guidance naming the chain (never suggests `.env` keys) |
| Access denied | Missing EULA grant or IAM | Model + region named; grant `bedrock:InvokeModel`, retry |
| Throttling | Quota burst | Retry shortly |
| Bad/unknown model ID | Typo or wrong region | `ANSWER_MODEL_ID` verification guidance |
| Network/endpoint down | Region/SG/DNS | Region named; check + retry |
| Unknown `LLM_PROVIDER` | Typo in env | `ValueError` at startup, names the valid values |
| Empty `ANSWER_MODEL_ID` | Not configured | `ValueError` at provider construction |

`is_reachable` (warmup/health) is a cheap control-plane call, never an
inference call; per-model entitlement failures surface at generation
with the mapping above. UI maps transport failures to the existing
retry-oriented copy; details stay in structured logs (no secrets).

## Rollback

Config-only, no migration (vectors/embeddings/prompts untouched):

```bash
LLM_PROVIDER=lmstudio   # in .env (or unset; lmstudio is the default)
sudo systemctl restart rag4i-cloud.service
```

Verify with `deploy/healthcheck.sh` (exit 0) as usual.

## Benchmark comparison procedure (5A vs 4.12 baseline)

1. Freeze `tests/benchmarks/cases.json` version.
2. Baseline: default config → `run_all()` (deterministic doubles;
   `model_key="qwen-local"`; cost 0).
3. Staged: `LLM_PROVIDER=bedrock`, `ANSWER_MODEL_ID=<candidate>`,
   `BEDROCK_REGION=us-east-1`, `BENCHMARK_LIVE_BEDROCK=1` →
   `run_all(llm=live_bedrock_llm(), model_key="<sonnet-4.6|haiku-4.5|…>")`.
   `live_bedrock_llm()` returns `None` unless the flag is set and
   raises loudly on misconfiguration — the normal suite never calls it.
4. Compare paired per-case metrics (pass rate, guard counts, support,
   latencies, tokens, cost) per `docs/PHASE_5_ACCEPTANCE.md`
   calibration method.

## Live smoke test (later, on demand — NOT run in 5.0)

```bash
LLM_PROVIDER=bedrock ANSWER_MODEL_ID=<in-region profile id> \
  BEDROCK_REGION=us-east-1 BENCHMARK_LIVE_BEDROCK=1 \
  venv/bin/python -c "
from tests.benchmarks.harness import run_all, live_bedrock_llm
out = run_all(llm=live_bedrock_llm(), model_key='sonnet-4.6')
print(out['summary'])"
```

Requires: EC2 role (or env) credentials, model access granted, ZDR +
SCP posture from the governance ADR. No fallback to another model on
failure — errors report per the table above.
