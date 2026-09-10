# Phase 5 Architecture (end-state spec — 5D planner + 5E reviewer implemented, rest as specified)

Baseline: Phase 4.12 at `35674f8`. Phase 5 law: **the models may become
smarter; the evidence system does not become less deterministic.**

## 0. End-state data flow

```text
browser → Streamlit (app.py)
  → deterministic router (query_router.py + workflows.detect_workflow)
        │  (no LLM; unchanged. Decides workflow type only.)
  → optional query planner (NEW, 5D, gated by REASONING_ENABLED)
  → plan validator (NEW, deterministic, untrusted-input boundary)
  → workflows / RAG (workflows.py + backend.py — frozen core)
  → scoped corpora (VectorStore.search + scope filter, 5B/5C)
  → answer LLM (NEW provider, 5A; LM Studio retained for dev/fallback)
  → sanitizer + citation guard (output_safety.py, answer_support.py)
  → conditional answer reviewer (NEW, 5E, gated by REVIEW_ENABLED)
  → validated verdict → optional ONE gap-fill retrieval + regeneration
  → citation guard again
  → session_state message (unchanged shape)
```

The planner sits BEFORE retrieval and never touches evidence, prompts,
or the answer path directly. The validator sits BETWEEN planner output
and workflow execution: planner output is untrusted data.

## A. Model roles

- **ANSWER** (5A): generates user-visible answers from retrieved
  evidence. Replaces Qwen-via-tunnel in production; same provider
  interface (`generate` / `generate_stream`), same temperature policy
  (0.0), same templates (frozen) plus the three 4.12 workflow prompts.
- **REASONING / QUERY PLANNER** (5D): proposes a retrieval plan
  (sub-queries, per-document targets, field lists) as strict JSON. It
  never generates user text, never retrieves, never executes. Output
  schema-fixed; anything else is a validation failure.

## B. Provider interfaces

```text
LLMProvider (existing ABC: generate/generate_stream/is_reachable/describe)
├── LMStudioProvider (kept; local dev + production fallback)
└── BedrockAnswerProvider (NEW, 5A): langchain_aws ChatBedrockConverse
    model=answer model id, temperature=0.0, region=in-region profile

PlannerProvider (NEW, 5D): generate_plan(prompt, schema) -> dict|error
└── BedrockPlannerProvider: ChatBedrockConverse.with_structured_output
    (forced tool calling — the verified path; native outputConfig is
    unverified for the installed langchain-aws, see 5D report)

As built: `LLMQueryPlanner(structured_fn)` takes an injected
`structured_fn(prompt, schema) -> dict`; production wiring
(`build_llm_planner`) adapts a `BedrockConverseProvider` chat model via
forced tool calling. No tools/agents, no answer generation in the
planner class by construction.
```

Factory seam: `get_llm_provider(config)` currently ignores
`config.llm_provider` (always LM Studio) — 5A makes it dispatch on
`LLM_PROVIDER=lmstudio|bedrock` plus `ANSWER_MODEL_ID`,
`BEDROCK_REGION`. LM Studio stays the default until 5A acceptance passes.

## C. Document scopes

- **persistent**: today's corpus. `scope='persistent'`, `session_id=NULL`.
  Ingested via the existing pipeline (S3/local → loader → 1000/200 →
  MiniLM → vector store). Unchanged behaviour for all current users.
- **session** (5C): files uploaded in one browser session, indexed
  through the SAME loader/chunker/embedder into the SAME `chunks`
  table with `scope='session'`, `session_id=<opaque 128-bit id>`.
- **combined retrieval**: every scoped search issues
  `scope='persistent' OR (scope='session' AND session_id=:current)`.
  Session chunks are invisible to every other session by construction
  of the filter, not by application discipline.

## D. Identity hierarchy

```text
identity (future login; today: single pilot operator, section J)
└── session (opaque 128-bit id, browser cookie / URL token, server map)
    ├── document (document_id = sha1(content_hash + file_name)?, see 5B)
    │   └── chunk (chunk_id, existing content-hash IDs, + scope/session)
```

Note: today's `document_id` is `sha1(abs_path)` (path-derived) and
`chunk_id` is content-derived. 5B must move `document_id` to a
content-derived identity (path-independent) so re-uploads and session
copies of the same file deduplicate instead of forking identity.

## E. Retrieval isolation rule

Every scoped vector search MUST carry:

```sql
WHERE scope = 'persistent'
   OR (scope = 'session' AND session_id = :current_session_id)
```

Chroma and pgvector implementations both enforce it inside
`search()`/`chunks_for_source()`/`list_sources()` (provider parity,
as today). Missing/NULL session bindings resolve to persistent-only.

## F. Session-document lifecycle

```text
upload → validate → indexed (scope=session) → available (in-filter)
  → logical detach (New Chat / explicit remove / expiry flips
     availability WITHOUT deleting rows)
  → retention window (see security limits)
  → physical deletion (DELETE WHERE session_id=..; vacuum; backups
     age out per policy)
```

Detach must be instant and total for retrieval even before physical
deletion completes. Delete must cover the vector rows AND any cached
text (no orphaned S3 objects: session files live under a per-session
prefix deleted as a unit).

## G. Planner → validator → workflow boundary (implemented, 5D)

```text
planner output (JSON): {sub_queries[], targets[], fields[], strategy}
        │  UNTRUSTED
validator (deterministic, no LLM):
  1. schema check (required keys, types, length caps)
  2. target check (filenames ⊆ indexed ∪ session files; else drop)
  3. budget check (≤4 sub-queries, ≤24 chunks, char caps — 4.12 budgets)
  4. on ANY failure → fall back to the 4.12 deterministic path for
     this turn and count a validation failure (telemetry)
        │  VALIDATED plan (or fallback marker)
workflows.py executes with the SAME retrieval/generation/verify steps
```

As built (`query_planner.py`): Plan v1 has exactly the eight approved
keys (see prompt §2); the validator returns `(plan, None)` or
`(None, category)` and never repairs. Workflow type stays with
deterministic detection — a plan claiming another workflow is discarded
and counted. Deterministic plans carry empty `retrieval_queries`, so
the executor augments nothing and the 4.12 path runs byte-identically.
LLM plans may add ≤4 validated queries to normal retrieval only
(appended after `build_query_forms`, same max-pool/top-k/threshold).
A required clarification renders a deterministic template, never plan
text. Bounded LRU plan cache keyed by
(schema, normalized question, sorted document IDs, session, provider,
model). Escalation reasons emitted: non-document, ordinary,
ambiguous_target, compound, complex_extraction, multi_form_retrieval
(`comparison`/`other` reserved).

The validator is small, pure, and fully unit-tested. The planner can
be wrong; the system cannot.

## H. Failure / fallback rules

- Cloud answer provider unreachable/slow → LM Studio fallback ONLY if
  explicitly enabled (`CLOUD_FALLBACK=lmstudio`); otherwise an honest
  error. Never silently substitute another model (invariant).
- Planner timeout/invalid JSON → silent-to-user fallback to the 4.12
  deterministic path + telemetry counter (no user-visible error for a
  helper that adds no value that turn). As built, failure categories
  are: credentials, unavailable, timeout, malformed, schema, workflow
  (reroute attempt), unsupported, model — each recorded, each falling
  back to the deterministic path with identical evidence behavior.
- `REASONING_ENABLED=0` → byte-identical 4.12 behaviour (acceptance gate).
  As built, `get_session_planner_bundle` returns None without building
   anything, and the app takes the untouched call path (proven by parity
   tests, not just inspection).
- Reviewer failure (credentials, unavailable, timeout, malformed,
  schema, unsupported, model_error) → preserve the original answer +
  existing citation-guard behavior, record metadata-only failure, never
  repair. `REVIEW_ENABLED=0` → pre-5E behavior exactly (no reviewer
  object, no network; proven by parity tests).
- Session store unavailable → session features disable with a clear
  message; persistent Q&A continues.

## I. Data flow and trust boundaries

```text
[browser] ── question, session token ──▶ [app]
[app] ── retrieved chunk text + question ──▶ [answer LLM @ region R]
[app] ── plan request (NO document text) ──▶ [planner LLM]
[planner] ── JSON plan ──▶ [validator] ──▶ [workflows]
[app] ── bounded review input (question + answer + excerpts) ──▶ [reviewer LLM]
[reviewer] ── JSON verdict ──▶ [validator] ──▶ [one gap-fill retrieval + regen]
```

Trust: browser input untrusted (as today); planner output untrusted;
reviewer output untrusted (treated as untrusted input to the
deterministic validator; suggested queries re-validated before any
retrieval); retrieved evidence trusted-for-grounding (as today, same
pipeline); LLM answer text untrusted until sanitizer + citation guard
(as today). The deterministic evidence executor remains authoritative;
the reviewer is downstream quality control only (never a second
planner, fact checker, answer writer, agent, or tool executor).

## J. Future authentication insertion point

No auth in 5.0. Insertion point (no code yet): `config.identity` plumbed
from a login layer into (1) session issuance/binding, (2) the
`session_id` filter, (3) per-row identity scoping when multi-tenancy
arrives. Until then: single-operator pilot assumptions hold, and
server-side history stays out (per the chat-history ADR).

## K. How Phase 5 preserves the existing RAG core

Unchanged by design: loader, 1000/200 splitter, MiniLM-384, k=5,
threshold 0.3, temperature 0.0, all frozen prompt templates, provider
abstraction shape, Chroma/pgvector parity, streaming + sanitizer,
support/clarification/citation-guard semantics, rerun-safe
session_state, telemetry metadata-only rule. Phase 5 adds providers
beside LM Studio, columns beside existing metadata, and a planner
beside (never inside) the retrieval path. The 4.12 benchmark
(`tests/benchmarks/`) is the regression proof at every stage.

## L. Answer reviewer / bounded repair (implemented, 5E)

Module `answer_reviewer.py` (mirrors the 5D planner structure).

- **Schema (exact, schema_version=1):**
  `{schema_version, verdict, missing_aspects, unsupported_claims,
  suggested_followup_queries}` with `verdict` in
  `[pass, repair, insufficient]`. Bounds: ≤4 missing, ≤4 unsupported
  (≤200/300 chars), ≤3 followups (≤200 chars). Exact key set;
  unknown keys, malformed structures, invented verdicts, answer-text
  fields, path fields, and secret fields are all rejected (never
  silently repaired).
- **Gap definitions:** missing_aspect = explicitly requested aspect
  not addressed using retrieved evidence; unsupported_claim =
  materially factual claim with no supporting chunk. Style, wording,
  verbosity, formatting are never gaps.
- **Input boundary (bounded, deterministic):** effective question
  (≤500), answer (≤2000), ≤5 evidence excerpts (≤1200 chars each,
  file_name + page labels only), validated Plan v1 metadata when
  available (workflow/escalation only), workflow, ≤20 filenames.
  Never transcripts, raw paths, session ids, secrets, unrelated
  documents, hidden state, or corpus contents.
- **Invocation (deterministic):** review runs only when the citation
  guard flags (count>0), workflow==comparison, workflow==summary, or
  REVIEW_ALWAYS_IF_CONFIGURED=1. Priority:
  citation_guard > comparison > summary > always > none. System
  answers (refusals/clarifications/errors) and empty evidence are
  never reviewed. Trigger recorded as metadata.
- **Verdicts:** pass = no material gap; repair = bounded retrieval
  could plausibly improve grounding (only path to repair);
  insufficient = evidence cannot support the request (no search,
  preserve grounded/refusal behavior).
- **One-cycle repair:** validate followups (strip, 3–200 chars, no
  path structure, known .pdf names only when a file list is given),
  run the SAME `backend.retrieve_documents` per query (k=5,
  threshold 0.3, same session_id/scope), merge deterministically
  (union by chunk_id, max score wins, score-desc, stable), regenerate
  once via the EXISTING path (normal: reassessed support template;
  comparison/summary: same workflow templates; extraction:
  deterministic table), rerun the citation guard, stop. Max 1 cycle,
  never loops. No reviewer-specific answer prompt.
- **Providers:** `LLMAnswerReviewer(structured_fn)` + Bedrock forced-
  tool adapter (same verified path as 5D); answer/reasoning/reviewer
  stay independently configurable (`REVIEW_PROVIDER/MODEL_ID/REGION/
  TIMEOUT_S`, default `REVIEW_ENABLED=0`). `langchain-aws` is not
  installed here, so native outputConfig is unverified; every verdict
  passes the deterministic validator regardless.
- **Telemetry (metadata only):** review_used, review_provider/model,
  review_latency_ms, review_verdict, review_failure_category,
  review_trigger, repair_attempted/succeeded, review_tokens_in/out,
  review_cost_usd, guard_before/after counts. Never questions,
  answers, chunks, raw output, or secrets.
- **Benchmarks:** harness `reviewer=` param + metrics review_* +
  guard_before/after; summary aggregates review_used /
  repair_attempted / repair_succeeded / guard_improved (synthetic
  data only; thresholds.json untouched).
- **Known limitations:** repair regeneration for comparison/summary
  rebuilds workflow context from merged evidence (new files outside
  the original pair join as sorted extra blocks; summary stays
  scoped to its target file); staged part-summaries are not re-run
  during repair (single bounded call, ≤24 chunks).
