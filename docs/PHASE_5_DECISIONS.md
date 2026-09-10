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

## PENDING BENCHMARK

- Cloud provider final confirmation (Bedrock vs fallback).
- Answer model: Sonnet 4.6 vs Haiku 4.5 (Nova Pro cost control).
- Planner model: Haiku 4.5 vs Nova Micro.
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
