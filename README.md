# RAG-4i-Cloud (Phase 2 — PostgreSQL/pgvector second provider)

Provider-agnostic successor of the local `RAG-4i` CA Legal Assistant.
Phase 1 refactored the app into providers (all local, same RAG behaviour).
Phase 2 adds **PostgreSQL + pgvector (AWS RDS)** as a second Vector Store
provider **alongside Chroma** — storage engine only. Embeddings, chunking,
retrieval behaviour, prompt, and LM Studio are unchanged.

## Relationship to original RAG-4i

- Original repo: `RAG-4i` (`app.py`, `backend.py`, `requirements.txt`,
  `requirements.lock.txt`, `start_app.bat`, `.gitignore`) — treated as
  **read-only reference/fallback** during this refactor.
- This folder `RAG-4i-Cloud/` is a separate copy/successor.
- Preserved exactly: `PyPDFLoader`, `RecursiveCharacterTextSplitter`
  (`chunk_size=1000`, `chunk_overlap=200`), `HuggingFaceEmbeddings`
  (`all-MiniLM-L6-v2`), `Chroma(persist_directory="chroma_db")`,
  `similarity_search_with_relevance_scores(k=5, threshold 0.3)`,
  `ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",
  model="local-model", temperature=0.0)`, and the prompt template.

## Architecture

```text
Application (app.py / backend.py)
├── Document Storage Provider (document_storage.py)
│   └── LocalDocumentStorage now, S3-ready interface
├── Ingestion (document_loader.py → chunking.py → embeddings.py → vector_store.py)
│   └── PyPDFLoader / 1000-200 splitter / MiniLM now
├── Vector Store Provider (vector_store.py dispatches by VECTOR_STORE)
│   ├── Chroma (default, local) — sole Chroma owner module
│   └── PostgreSQL/pgvector (postgres_vector_store.py) — AWS RDS
├── Embedding Provider (embeddings.py)
│   └── Hugging Face MiniLM now, swappable
└── LLM Provider (llm_provider.py)
    └── LM Studio now (OpenAI-compatible), swappable
```

`VECTOR_STORE=chroma` (default) or `VECTOR_STORE=postgres` selects the
provider via configuration only — no code changes to switch. Chroma remains
the default until PostgreSQL is explicitly verified.

Why each module exists:

- `config.py` / `.env.example` — centralise all env/config (no secrets in code,
  stdlib `.env` loader, no new dependency).
- `document_storage.py` — `list/get/save/delete` abstraction over local FS.
- `document_loader.py` — locate/load PDFs, standardise
  `source_path/file_name/page/document_id`, per-file error reporting.
- `chunking.py` — single place for 1000/200 splitter + `chunk_id`.
- `embeddings.py` — wraps `HuggingFaceEmbeddings` so app code never imports it.
- `vector_store.py` — `build_index/search/get_status` interface + dispatch;
  Chroma implementation append-only (no `shutil.rmtree`).
- `postgres_vector_store.py` — pgvector implementation of the same interface
  (`chunks` table, `embedding vector(384)`, additive `ON CONFLICT DO NOTHING`).
- `migrate_to_postgres.py` — copy/additive migration
  (PDFs → loader → chunker → embedder → PostgreSQL; Chroma never touched).
- `compare_providers.py` — same query vs both providers (content/source/page/score).
- `llm_provider.py` — wraps `ChatOpenAI`/LM Studio; owns base URL/key/model/temp.
- `backend.py` — orchestrates `retrieve_documents()` vs `generate_answer()`,
  structured sources `{source, file_name, page, score, ...}`, status helpers.
- `app.py` — same Streamlit flow, real Environment/KB/LLM status, no offline claim.
- `tests/` — proves the refactor did not break local behaviour.

## Local setup

Requirements: Python 3.10+, LM Studio running locally.

```bash
# 1. Clone / enter this folder
cd RAG-4i-Cloud

# 2. (Optional) virtualenv
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# 3. Install (ask before installing in shared envs)
pip install -r requirements.txt

# 4. Configure
copy .env.example .env
# or: cp .env.example .env  (edit values if needed)
```

`.env` stays git-ignored (see `.gitignore`). Defaults already match local use:

```text
APP_ENV=local
VECTOR_STORE=chroma
CHROMA_PATH=chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_PROVIDER=lmstudio
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=local-model
LLM_TEMPERATURE=0.0
```

PostgreSQL (only used when `VECTOR_STORE=postgres`; `.env` stays git-ignored,
never commit real credentials):

```text
DB_HOST=localhost
DB_PORT=15432
DB_NAME=rag4i
DB_USER=
DB_PASSWORD=
# Or: DATABASE_URL=postgresql://USER:PASSWORD@localhost:15432/rag4i
```

`requirements.txt` additionally includes `psycopg2-binary` + `pgvector`.

## How to run

Windows:

```bat
start_app.bat
```

Any OS (no batch dependency):

```bash
streamlit run app.py
```

## How to index documents

1. Start the app, open the sidebar **Knowledge Base**.
2. Enter a local folder path (e.g. `D:\Clients\ABC_Ltd\Legal`).
3. Click **Build/Update Database**.
4. The message reports `found/succeeded/failed` + failed filenames.
   Existing index data is **kept** (re-indexing appends; duplicates possible in P1).

## How to query

1. Wait for `Knowledge Base: Ready` + `LLM: Reachable` in the sidebar.
2. Ask in the chat box (e.g. “What is the lock-in period in the lease deed?”).
3. Answers are grounded in retrieved chunks; sources show
   `filename (p. N) [score]`. Low-relevance queries return the
   “could not find enough relevant information” fallback.

## PostgreSQL/pgvector (Phase 2)

RDS stays **private** (no public access). Local development connects via SSH
local port forwarding through the existing EC2 host:

```text
Local machine → ssh -L 15432:<RDS-ENDPOINT>:5432 <SSH-USER>@<EC2-HOST>
→ private RDS rag4i-db:5432
```

The app then connects to `localhost:15432` (`DB_HOST`/`DB_PORT`).
DB credentials live only in local `.env` (git-ignored).

Migrate a PDF folder (additive copy; Chroma is never modified):

```bash
python migrate_to_postgres.py <pdf-folder>
```

Compare both providers on the same query (scores may differ; content should match):

```bash
python compare_providers.py "What is the lock-in period?"
```

Switch providers with configuration only (`VECTOR_STORE=chroma` remains default):

```bash
VECTOR_STORE=postgres streamlit run app.py
```

## Local LM Studio requirements

- Run LM Studio server at `LLM_BASE_URL` (default `http://localhost:1234/v1`).
- Load any local model (`LLM_MODEL` name is opaque to LM Studio).
- Temperature `0.0` for precise, non-creative answers.

## What is NOT implemented (later phases)

S3, Lambda, Bedrock, cloud GPUs, vLLM, Docker, Kubernetes, auth,
multi-user permissions, OCR, hybrid search, BM25, reranking, query rewriting,
conversational memory. (RDS PostgreSQL/pgvector second provider and EC2 SSH
tunnel for local access are covered above in Phase 2; RDS stays private.)

## Tests

```bash
python -m unittest discover -s tests -v
```

Covers config, ingestion/chunking, metadata, retrieval, provider abstractions,
and PostgreSQL (mocked). Heavy deps (Chroma/model/LLM server) are mocked or
skipped so the suite runs without a GPU or server. The real-RDS test is
opt-in only and needs the SSH tunnel up:

```bash
$env:RUN_PG_INTEGRATION="1"
python -m unittest tests.integration.test_postgres_rds -v
```
