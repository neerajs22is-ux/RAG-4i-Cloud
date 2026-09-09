# ARCHITECTURE (HEAD `23503d7`)

## Control flow (question)
```text
browser → Streamlit (app.py)
  → observe_query(message, ConversationMemory)   [query_router.py]
  → route: conversation/capability/out-of-scope → canned reply (no retrieval)
  → DOCUMENT_QUERY/FOLLOWUP
      → expand_followup_query (follow-ups only)  [query_router.py]
      → build_query_forms → per-form search, max-pool, top-k, threshold 0.3
      → broad scope? source top-up (chunks_for_source)   [backend.py]
      → assess_support → DIRECT | PARTIAL | UNSUPPORTED  [answer_support.py]
      → PARTIAL + missing terms → one deterministic clarification (no LLM)
      → generate (streaming w/ ThinkStreamSanitizer, fallback to whole)
      → think-strip guard → citation guard (label downgrade + marker)
      → persist message {label, sources, strength, guard_note} to session_state
```

## Control flow (ingestion)
```text
local folder | S3 bucket → PyPDFLoader → RecursiveCharacterTextSplitter
  → HuggingFaceEmbeddings → Chroma | pgvector  (per-file on_progress)
```

## Provider boundaries (config-only switching)
- Document storage: `DOCUMENT_STORAGE=local|s3`
  (`document_storage.py` ↔ `s3_document_storage.py`; boto3 default chain).
- Vector store: `VECTOR_STORE=chroma|postgres`
  (`vector_store.py` ↔ `postgres_vector_store.py`; only these two modules
  may import Chroma/psycopg internals — enforced by tests).
- Embeddings: `embeddings.py` (shared per-model process cache).
- LLM: `llm_provider.py` (`generate` + `generate_stream`, same chain/config).

## UI layering (`app.py` + `ui/`)
- `ui/tokens.py` + `ui/css/` (variables-first styling, OS dark-mode default).
- `ui/components.py` (pure helpers: labels, sources, copy, export, strength).
- Rerun-safety rule: widgets render from `session_state` metadata every run;
  prompt handling only appends state, then reruns.

## Cross-cutting
- `conversation_memory.py`: bounded 3-turn window; greetings/out-of-scope
  never contaminate retrieval anchors.
- `model_warmup.py`: real readiness gate (embeddings load + LLM probe).
- `output_safety.py`: batch + streaming think removal.
- `suggestions.py`: probed starters (`preview_answer`), contextual follow-ups.
- Structured logging everywhere; user errors via `friendly_error()`.
