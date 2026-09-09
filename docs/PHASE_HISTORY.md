# PHASE_HISTORY (one–two lines per phase)

- Phase 1: local provider-agnostic refactor (storage/vector/embed/LLM abstractions, config-driven, Chroma default).
- Phase 2: PostgreSQL + pgvector second provider; live RDS migration verified, Chroma fallback kept.
- Phase 3: S3 document-storage provider; lease/contract test-set migration verified.
- Phase 4: EC2 pilot deployment (systemd Streamlit, private RDS/S3, reverse LM tunnel, IP-restricted 8501).
- Phase 4.1: cloud/local ingestion split, per-file reporting, no masked failures.
- Phase 4.2: deterministic query router (conversation/capability/document/follow-up/out-of-scope).
- Phase 4.3: conversation memory, think cleanup, multi-form retrieval (synonym form measured and rejected).
- Phase 4.4: smart starter/follow-up suggestions (probed, clickable).
- Phase 4.4.1: fixed lost-click rerun bug for follow-up buttons.
- Phase 4.5: Streamlit UI overhaul (tokens, AppTest acceptance, rerun-safe rendering).
- Phase 4.6: answer reasoning (support levels, clarification loop, citation guard, sentence-form anchors).
- Phase 4.7: frontend hardening, idempotent indexing, delete flow, logging, healthcheck, AppTest + S3 suites.
- Phase 4.8: streaming answers, ingest progress, excerpts, strength signal, keyboard polish, history ADR.
- Phase 4.9: sidebar condense, grounded (probe-validated) starters.
- Phase 4.10: starter probe via real prep path, clarification loop, citation guard, structured anchors (this baseline).
