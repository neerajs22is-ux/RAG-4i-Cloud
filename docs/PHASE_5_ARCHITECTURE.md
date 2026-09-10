# Phase 5 Architecture (end-state spec — design only, nothing implemented)

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
    (forced tool calling today; native outputConfig when langchain-aws
    supports it — see model evaluation doc)
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

## G. Planner → validator → workflow boundary

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

The validator is small, pure, and fully unit-tested. The planner can
be wrong; the system cannot.

## H. Failure / fallback rules

- Cloud answer provider unreachable/slow → LM Studio fallback ONLY if
  explicitly enabled (`CLOUD_FALLBACK=lmstudio`); otherwise an honest
  error. Never silently substitute another model (invariant).
- Planner timeout/invalid JSON → silent-to-user fallback to the 4.12
  deterministic path + telemetry counter (no user-visible error for a
  helper that adds no value that turn).
- `REASONING_ENABLED=0` → byte-identical 4.12 behaviour (acceptance gate).
- Session store unavailable → session features disable with a clear
  message; persistent Q&A continues.

## I. Data flow and trust boundaries

```text
[browser] ── question, session token ──▶ [app]
[app] ── retrieved chunk text + question ──▶ [answer LLM @ region R]
[app] ── plan request (NO document text) ──▶ [planner LLM]
[planner] ── JSON plan ──▶ [validator] ──▶ [workflows]
```

Trust: browser input untrusted (as today); planner output untrusted;
retrieved evidence trusted-for-grounding (as today, same pipeline);
LLM answer text untrusted until sanitizer + citation guard (as today).

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
