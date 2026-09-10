# CURRENT_STATE (HEAD `23503d7`, Phase 4.10)

Provider-agnostic Streamlit RAG legal assistant. All RAG behavior,
thresholds, prompts, and provider boundaries frozen (see RAG_INVARIANTS.md).

## Live / deployment state (last verified)
- EC2 `rag4i-admin` (`t3.micro`, us-east-1): Streamlit systemd service
  `rag4i-cloud.service`, app at `http://54.196.157.142:8501` (IP drifts on
  stop/start; SG allows 22 + 8501 from operator `/32` only).
- RDS `rag4i-db`: private, PostgreSQL 18.3 + pgvector 0.8.1. Local access
  via SSH tunnel `localhost:15432`; on EC2 direct to private endpoint.
- S3 `rag4i-company-documents-305740358559-ap-south-2-an` (ap-south-2,
  private, versioned, SSE-S3), prefix `documents/`. EC2 reads via
  least-privilege IAM role (GetObject + ListBucket on the prefix).
- LM Studio on operator PC (Qwen 3 1.7B); EC2 reaches it only through the
  reverse tunnel `-R 1234:127.0.0.1:1234` (must be re-established after
  reboot/logout; port 1234 never opened publicly).
- Defaults: `VECTOR_STORE=chroma`, `DOCUMENT_STORAGE=local`. Secrets only
  in git-ignored `.env` (local) and EC2 `/home/ec2-user/RAG-4i-Cloud/.env`.

## Active features
- Intent observer (conversation/capability/document/follow-up/out-of-scope)
  with bounded conversation memory and topic-anchored follow-up expansion.
- Multi-form retrieval + max-pooling; threshold/top-k unchanged.
- Support levels (DIRECT/PARTIAL/UNSUPPORTED), one deterministic
  clarification per question, citation guard (label downgrade + marker).
- Streaming answers with think-block sanitizer + non-streaming fallback.
- Grounded starter/follow-up suggestions (probed, clickable), source
  expanders with excerpts, retrieval-strength note, copy/export/new-chat,
  light/dark toggle, model warmup gate with retry, ingest progress +
  idempotent re-index + delete-by-document flow.

## Known limitations
- `t3.micro` (913 MB RAM) is tight for MiniLM+torch; avoid parallel model
  loads on the box (previously starved SSH).
- Qwen emits `<think>` traces (stripped at display) and sometimes
  editorializes beyond context; prompt frozen, so this is model-side.
- Short paraphrases can score just under the 0.3 threshold on tiny
  indexes (intended conservative behavior, not a bug).
- Synonym query expansion was measured and deliberately NOT shipped.
- No auth/HTTPS (pilot serves test documents only); history is in-memory.

## Important files
- `app.py` (all UI), `backend.py` (orchestration; `query_documents`,
  `stream_answer`, `ingest_with_report`, `preview_answer`), `ui/`,
  `query_router.py`, `answer_support.py`, `conversation_memory.py`,
  `vector_store.py` + `postgres_vector_store.py`, `document_storage.py` +
  `s3_document_storage.py`, `embeddings.py`, `llm_provider.py`,
  `model_warmup.py`, `output_safety.py`, `suggestions.py`, `config.py`,
  `deploy/` (service, setup, healthcheck, SECURITY.md), `docs/`.
- Original `RAG-4i` repo (sibling directory): read-only reference, never touch.

## Immediate roadmap
- Phase 5 in progress: 5.0 foundation, 5A Bedrock answer provider,
  5B scope primitives, 5C session uploads, 5D query planner, 5E answer
  reviewer / bounded repair (see `docs/PHASE_5_*.md`; 5F not started).
  Candidate future work is scoped (not implemented) in
  `docs/adr-retrieval-upgrades.md` and `docs/adr-chat-history.md`.
