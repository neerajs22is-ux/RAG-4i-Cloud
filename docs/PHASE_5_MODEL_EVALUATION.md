# Phase 5 Model Evaluation Shortlist (researched 2026-09-10)

Volatile: re-verify prices/regions/availability at 5A kickoff. Nothing
below is wired into code (no winner hard-coded anywhere).

## Answer-model candidates (Bedrock, us-east-1, in-region)

| Model | Quality signal | Context | $/1M in/out | Streaming | Structured out | Latency | Residency note |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 4.6 | Flagship balance; 1M ctx | 1M | 3 / 15 | ConverseStream ✓ | Forced-tool ✓; native schema ✓ | Mid | In-region us-east-1 |
| Claude Haiku 4.5 | Fast/cheap Claude | 200K | 1 / 5 | ✓ | Native schema ✓ | Low | In-region |
| Amazon Nova Pro | Amazon workhorse | 300K | 0.80 / 3.20 | ✓ | Verify per-model | Low-mid | In-region; Amazon FM privacy terms |
| Amazon Nova Lite | Budget anchor | 300K | 0.06 / 0.24 | ✓ | Verify per-model | Low | Same |
| Meta Llama 4 Scout | Open weights, 10M ctx | 10M | 0.17 / 0.66 | ✓ | NO native; forced-tool unreliable | Low-mid | Answer-fallback only |

Excluded from pilot: Claude Fable 5/5.1 (mandatory 30-day review
retention), Opus 4.8 ($5/$25 — unjustifiable for grounded Q&A),
anything requiring Geo/Global profiles.

Shortlist to benchmark: **Sonnet 4.6 vs Haiku 4.5**, Nova Pro as cost
control. Llama 4 Scout only if a no-Anthropic arm is wanted.

## Planner/reasoning candidates (strict-JSON plan proposals, 5D)

| Model | Reasoning/JSON signal | $/1M | Latency | Tool discipline | Structured out |
|---|---|---|---|---|---|
| Claude Haiku 4.5 | Strong small-model instruction following; native schema | 1 / 5 | Low | Good | Native ✓ |
| Amazon Nova Micro | Cheapest ($0.035/0.14), 128K | 0.035 / 0.14 | Lowest | Adequate for fixed-schema plans (benchmark must prove) | Verify |
| Amazon Nova Lite | Step up, 300K | 0.06 / 0.24 | Low | Better headroom for plan+file lists | Verify |

Context needs are small (question + file list + few exemplars, no
document text — planner never sees evidence). Cost per plan call is
sub-cent on any candidate; the gate is JSON reliability + validation
failure rate, not price. Shortlist to benchmark: **Haiku 4.5 vs Nova
Micro** on schema-valid first-try rate.

## Integration notes (for 5A/5D implementers)

- `langchain-aws` `ChatBedrockConverse` (add pinned dep): same
  `BaseChatModel` shape as today's `ChatOpenAI`, so provider subclasses
  stay thin; `temperature=0`, `region_name` = in-region endpoint.
- Native `outputConfig` JSON-schema is NOT yet in langchain-aws (open
  issue): use `with_structured_output` (forced tool calling) first;
  adopt native output when the library lands.
- Prompt caching: our frozen system templates are ideal cache blocks;
  enable `cachePoint` on the prompt template to cut input cost/latency.
- Token accounting: Converse returns usage metadata; the harness
  records it when live, estimates (tiktoken/chars÷4, marked) otherwise.
