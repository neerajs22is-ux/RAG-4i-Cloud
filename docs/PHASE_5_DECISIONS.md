# Phase 5 Open Decisions Register (updated at each stage gate)

## DECIDED (5.0)

- Phase 5 law: models may get smarter; the evidence system does not get
  less deterministic.
- Bedrock is the primary 5A candidate path (not yet the final model).
- ZDR (`data_retention_mode: none`) + in-region (us-east-1) profile +
  SCP deny on Global/Geo profiles are REQUIRED for any cloud pilot call.
- Fable-class models (mandatory review retention) excluded from pilot.
- Isolation = 0 leakage; `REASONING_ENABLED=0` = 4.12 parity; session
  lifecycle round-trips = 1.0.
- No server-side chat history in Phase 5 scope (chat-history ADR stands).
- No embedding replacement, no HNSW migration, no global RAG rewrite.

## DECIDED (5C)

- Uploads: PDF-only (magic bytes authoritative), 10 MB/file,
  5 files/session, 50 MB/session; encrypted/corrupt rejected per-file.
- S3 keys at bucket root `sessions/<sid>/uploads/<stem>-<sha12>.pdf`
  (outside the corpus prefix); local mirror under SESSION_STORAGE_DIR.
- Identity: `sha1("session-doc:" + sid + content_sha256 + safe_name)`;
  same-session re-upload is idempotent; no cross-session or
  persistent collisions by construction.
- Ordering: validate → store object → load/chunk/embed/index →
  scoped verify → ready; build failures clean up best-effort, partial
  states never marked ready, retry safe via deterministic IDs.
- New Chat rotates the binding (detach, vectors kept); restore rebinds
  same-lineage snapshots; starter cache is session-keyed.
- No per-doc remove UI, no retention job, no promotion, no scope toggle.

## DECIDED (5B)

- Scope primitives implemented (5B); session uploads implemented (5C).
- `document_id` stays path-derived in 5B (content-derived move
  deferred: changing identity now would break delete flows and
  existing IDs for zero 5B benefit).
- `content_hash` column deferred to 5C (matters at upload/dedupe time).
- Legacy mapping: pg `DEFAULT 'persistent'` + retry-once; Chroma
  metadata-only backfill; unbound reads degrade to 4.12 behavior
  (logged) instead of hiding data if backfill fails.
- Re-indexing identical content under a session re-scopes those chunk
  IDs via upsert (deterministic, no duplicates) rather than forking.

## DECIDED (5D)

- Planner proposes, validator disposes, executor (existing runners)
  stays authoritative; plan/workflow mismatch is discarded + counted.
- Deterministic plans carry empty `retrieval_queries` (augmentation
  neutral by construction); LLM plans augment normal retrieval only.
- Clarifications render a deterministic template; plan text is never
  rendered, logged, or telemetered.
- Forced-tool structured output is the v1 adapter; native outputConfig
  adoption waits on langchain-aws verification. `comparison`/`other`
  escalation reasons reserved, not emitted.
- Thresholds file untouched by 5D (measurements reported separately).

## DECIDED (5E)

- Reviewer is downstream quality control, never a second planner,
  fact checker, answer writer, agent, or tool executor; the
  deterministic evidence executor stays authoritative.
- Verdict schema fixed at version 1 (`pass`/`repair`/`insufficient`);
  gaps are missing_aspect / unsupported_claim only (style, wording,
  verbosity, formatting are never gaps).
- Invocation is deterministic: citation-guard flag, comparison,
  summary, or REVIEW_ALWAYS_IF_CONFIGURED; ordinary answers are not
  reviewed by default; system answers and empty evidence never are.
- Only `repair` repairs, at most ONE gap-fill retrieval +
  regeneration through the existing path (k=5, threshold 0.3, MiniLM,
  scope `persistent OR (session AND current)`); insufficient never
  searches; failures preserve the original answer.
- Forced-tool structured output is the v1 reviewer adapter (same
  verified path as 5D); native outputConfig waits on langchain-aws
  verification (still not installed here).
- Thresholds file untouched by 5E (measurements reported separately).

## PENDING BENCHMARK

- Cloud provider final confirmation (Bedrock vs fallback).
- Answer model: Sonnet 4.6 vs Haiku 4.5 (Nova Pro cost control).
- Planner model: Haiku 4.5 vs Nova Micro.
- Reviewer model: Haiku 4.5 vs Nova Micro (schema reliability +
  gap-correction per cost).
- Inference region confirmation (us-east-1 in-region availability of
  chosen model IDs at kickoff).
- All pending-calibration numeric gates (method in PHASE_5_ACCEPTANCE).
- Planner escalation threshold (from validation-failure distribution).
- Cloud fallback behaviour (honest-error vs LM-Studio fallback flag).
- HNSW trigger (only if pgvector IVFFlat/sequential scan proves too
  slow at session scale — measure first).

## PENDING POLICY

- India-region processing requirement (ap-south-2 docs → us-east-1
  inference): acceptable or not.
- Session retention window (proposal: 24h).
- Maximum upload size (proposal: 10 MB), files/session (5), session
  storage cap (50 MB).
- Upload rate/abuse constraints.
- Future auth insertion decision (SSO/OIDC when multi-user arrives).
