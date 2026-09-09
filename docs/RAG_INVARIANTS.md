# RAG_INVARIANTS — do not change without explicit approval

Verified against code at HEAD `23503d7` (`config.load_config()` defaults
unless noted). The full suite asserts most of these.

## Retrieval core (semantic identity of the product)
- Loader: `PyPDFLoader`, page numbers preserved as-is (None if absent).
- Splitter: `RecursiveCharacterTextSplitter`, `chunk_size=1000`,
  `chunk_overlap=200`.
- Embeddings: `HuggingFaceEmbeddings("all-MiniLM-L6-v2")`, 384-dim,
  `pgvector vector(384)`.
- Retrieval: `similarity_search_with_relevance_scores`, `k=5`,
  relevance threshold `0.3` (never lowered to fix recall).
- LLM: LM Studio OpenAI-compatible endpoint, `temperature=0.0`.
- `PROMPT_TEMPLATE` byte-for-byte (DIRECT path); `PARTIAL_PROMPT_TEMPLATE`
  is the only sanctioned variant (PARTIAL/OVERVIEW evidence).

## Grounding contract
- Uncertain classifications default to `DOCUMENT_QUERY` (retrieve, don't guess).
- Empty retrieval → existing low-relevance fallback (unchanged text).
- `UNSUPPORTED` evidence → contextual refusal, no LLM call.
- Assistant text is never retrieval evidence; greetings/out-of-scope never
  contaminate anchors; memory stays bounded.
- Sources (file/page/score) preserved exactly; citations never dropped.
- `<think>` never reaches the user (batch + stream sanitizers).

## Architecture rails
- Provider switching is config-only (`DOCUMENT_STORAGE`, `VECTOR_STORE`);
  app code never imports provider internals (test-enforced for Chroma).
- Migrations additive/non-destructive; re-indexing idempotent (content-hash
  chunk IDs; pgvector `ON CONFLICT DO NOTHING`); original `RAG-4i` untouched.
- Streaming changes presentation/latency only (same evidence/prompts).
- `.env` git-ignored; no secrets in code, logs, or commits.
