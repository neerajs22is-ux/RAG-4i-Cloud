# RAG-4i-Cloud (Phase 5 — cloud answer path + model benchmark)

Provider-agnostic Streamlit RAG assistant for CA legal documents, successor
of the local `RAG-4i`. Phase 1 refactored the app into providers (all
local, same RAG behaviour). Phase 2 adds **PostgreSQL + pgvector (AWS RDS)**
as a second Vector Store provider **alongside Chroma** — storage engine
only. Phase 3 adds **Amazon S3** as a second Document Storage provider
**alongside local filesystem** (private bucket: Block Public Access,
SSE-S3, versioned). Phase 4 deployed a hardened EC2 pilot. Phase 5 adds
the cloud answer path — Bedrock provider, query planner, session scope and
uploads, answer reviewer with bounded repair — plus a controlled model
benchmark. **Benchmark work is ongoing; no production model has been
selected.**

## Status at a glance

- **Production deployment:** none. What exists is a temporary EC2 pilot
  (test documents only, no auth/HTTPS) — see *EC2 pilot deployment (Phase 4)*
  below and `docs/CURRENT_STATE.md`.
- **Working inference test path:** AWS Bedrock Mantle, ap-south-1 (Mumbai),
  SigV4 from the EC2 instance role. Bedrock Converse is blocked for this
  account (entitlement wall, not IAM) — see
  `docs/SESSION_HANDOFF_2026-09-11.md`.
- **Benchmark status:** short exploratory run complete (2026-09-11);
  full 40-scenario run in progress. Results are exploratory, the judge has
  not run, and short results are **not** the final production decision.
- **Local defaults unchanged:** Chroma + MiniLM + LM Studio remain the
  defaults; no model winner is hard-coded anywhere.
- **Details live in `docs/`:** `CURRENT_STATE.md` (live/deployment state),
  `SESSION_HANDOFF_2026-09-11.md` (benchmark continuation point),
  `PHASE_HISTORY.md`, `PHASE_5_*.md`.

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
├── Document Storage Provider (document_storage.py dispatches by DOCUMENT_STORAGE)
│   ├── LocalDocumentStorage (default, local filesystem)
│   └── S3DocumentStorage (s3_document_storage.py) — private S3 bucket
├── Ingestion (document_loader.py → chunking.py → embeddings.py → vector_store.py)
│   └── PyPDFLoader / 1000-200 splitter / MiniLM now
├── Vector Store Provider (vector_store.py dispatches by VECTOR_STORE)
│   ├── Chroma (default, local) — sole Chroma owner module
│   └── PostgreSQL/pgvector (postgres_vector_store.py) — AWS RDS
├── Embedding Provider (embeddings.py)
│   └── Hugging Face MiniLM now, swappable
├── Answer LLM Provider (llm_provider.py dispatches by LLM_PROVIDER)
│   ├── LM Studio (default, OpenAI-compatible, local)
│   └── Bedrock Converse (bedrock_provider.py, opt-in; ANSWER_MODEL_ID required)
├── Session & Scope (document_scope.py, session_uploads.py, session_restore.py)
├── Query Planner (query_planner.py, Phase 5D) — strict-JSON plan, validated
├── Answer Reviewer (answer_reviewer.py, Phase 5E) — verdicts + bounded repair
└── Benchmark harness (tests/benchmarks/, experimental; results uncommitted)
```

`VECTOR_STORE=chroma` (default) or `VECTOR_STORE=postgres` selects the
vector provider via configuration only — no code changes to switch. Chroma remains
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
  Chroma implementation upserts by content hash (no `shutil.rmtree`,
  re-indexing is idempotent).
- `postgres_vector_store.py` — pgvector implementation of the same interface
  (`chunks` table, `embedding vector(384)`, additive `ON CONFLICT DO NOTHING`).
- `migrate_to_postgres.py` — copy/additive migration
  (PDFs → loader → chunker → embedder → PostgreSQL; Chroma never touched).
- `compare_providers.py` — same query vs both providers (content/source/page/score).
- `llm_provider.py` — wraps `ChatOpenAI`/LM Studio; owns base URL/key/model/temp.
- `bedrock_provider.py` — Phase 5A answer provider (Bedrock Converse /
  ConverseStream); opt-in via `LLM_PROVIDER=bedrock`, `ANSWER_MODEL_ID`
  required (a missing ID fails loudly, never substitutes a model),
  `BEDROCK_REGION` defaults to `us-east-1`.
- `query_planner.py` — Phase 5D planner: strict-JSON plan proposals with a
  deterministic validator (no unvalidated plan reaches retrieval).
- `answer_reviewer.py` — Phase 5E reviewer: strict-JSON verdicts and
  bounded repair (single attempt, frozen prompt; never rewrites freely).
- `document_scope.py` / `session_uploads.py` / `session_restore.py` —
  Phase 5B/5C per-session document scoping, uploads, and restore.
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
   A live `N / M documents processed` bar tracks the real ingestion loop.
   Re-indexing is idempotent: same content produces the same chunk IDs,
   so rebuilding does not create duplicates. Use **Manage indexed
   documents** in the sidebar to remove a document's vectors (source
   files are left untouched).

## Managing documents

- The sidebar lists indexed source files with per-file **Remove** actions.
- Removing deletes that document's vectors from the active vector store
  only; the original PDF (local folder or S3) is never deleted.
- Switch appearance between Auto/Light/Dark from the sidebar at any time.

## How to query

1. Wait for `Knowledge Base: Ready` + `LLM: Reachable` in the sidebar.
   A first-run **Get started** checklist shows live Model/KB state and
   disappears once both are ready (dismissible). The composer stays
   disabled (with an explanation) until both are ready.
2. Ask in the chat box (e.g. “What is the lock-in period in the lease deed?”).
   Answers stream in as they are generated.
3. Answers are grounded in retrieved chunks; sources show
   `filename (p. N) [score]` with an expandable chunk excerpt, plus a
   `Retrieval strength · X.XX` similarity note (not a probability).
   Each answer also shows real `Answered in X.Xs · N sources` timing.
   Low-relevance queries return the
   “could not find enough relevant information” fallback.
4. Use **Copy answer** (per message), **Export chat**, or **New chat**
   from the sidebar. Each source preview has **Ask about this document**
   (asks “Tell me more about `<file>`” through the normal pipeline).
   Rate answers with 👍/👎; 👎 offers a fixed reason set (no extra LLM).
   Appearance follows your system setting unless you
   pick Light/Dark in the sidebar.
5. When evidence only partly covers a question, the assistant asks one
   clarifying question first (reply “yes” to proceed); answers that go
   beyond the evidence are marked down to Partial with an
   “unverified against sources” note. Starter questions are validated
   to be answerable before they are shown.
6. Accidental refresh: the app keeps a browser-local snapshot
   (`localStorage`, current conversation only, no server storage). Use
   **Recover previous session?** to paste a backup and **Restore** /
   **Start fresh**. **New chat** clears messages, memory, follow-ups,
   feedback, and restore state only — documents, vectors, and config
   are untouched.

## Advanced legal workflows (Phase 4.12)

Same chat box, same grounding contract (retrieval k=5, threshold 0.3,
temperature 0.0). Ordinary questions use the unchanged Q&A path.

- **Compare documents**: “Compare the lock-in clauses in contract.pdf
  and lease.pdf.” Per-document evidence, a topic table plus a
  similarities/differences/missing conclusion. Ambiguous targets get a
  clarification listing real indexed files (nothing invented); if one
  side has no matching evidence the answer is marked
  **Partial comparison** and says so.
- **Structured extraction**: “List all payment deadlines.” Returns a
  table of verbatim evidence values with document + page. Missing
  values read “Not stated in the retrieved material”. No LLM call.
- **Document summaries**: “Summarize lease.pdf.” Bounded,
  document-scoped evidence (max 24 chunks, staged for longer files);
  topical questions such as “What does lease.pdf say about renewal?”
  stay on normal Q&A.
- **Open source**: each source preview has **Open source** to read the
  full cited page text (from already-retrieved evidence; works for
  local and S3 alike) plus sibling cited pages. PDF page-image
  rendering is not available in this pilot (no renderer on the
  `t3.micro` box, no public URLs); the viewer is text from retrieved
  chunks, never raw filesystem paths.

## Cloud answer path (Phase 5)

Keeps the frozen RAG contract (same retrieval, prompt, citation rules) and
adds, all behind the existing interfaces:

- **5.0** architecture/evaluation foundation (`docs/PHASE_5_ARCHITECTURE.md`).
- **5A** Bedrock answer provider (`LLM_PROVIDER=bedrock`, opt-in; local
  LM Studio remains the default and nothing switches silently).
- **5B/5C** per-session document scope, uploads, and scoped ingestion.
- **5D** query planner: strict-JSON plan proposals, deterministically validated.
- **5E** answer reviewer: strict-JSON verdicts with bounded repair.

Provider/model/region settings for planner and reviewer are independent
configuration (see `.env.example`); none is hard-coded to a model.

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

## S3 document storage (Phase 3)

The bucket stays **private** (Block Public Access, SSE-S3, versioned, owner
enforced). Authentication uses your local AWS credential chain
(`aws configure` profile or env credentials); nothing is created in AWS and
no credentials live in the repo (`.env` is git-ignored).

```text
DOCUMENT_STORAGE=local   # default, current local workflow
DOCUMENT_STORAGE=s3      # S3-backed ingestion, config-only switch
S3_BUCKET=<existing private bucket>
S3_REGION=ap-south-2
S3_PREFIX=documents/
```

Object keys are `<prefix>/<filename>` (e.g. `documents/lease.pdf`).
Migrate only the small test set (copy-only; local originals and existing
S3 objects are never deleted or overwritten):

```bash
python migrate_to_s3.py <local-folder-with-lease-and-contract-pdfs>
```

With `DOCUMENT_STORAGE=s3`, ingestion becomes
S3 → PyPDFLoader → 1000/200 chunking → MiniLM → pgvector (all unchanged).

## EC2 pilot deployment (Phase 4)

Temporary pilot: Streamlit on the existing EC2 instance, browser via
`http://<EC2-PUBLIC-IP>:8501` (restricted SG, no HTTPS/auth yet — test
documents only). S3 + private RDS as above; LM Studio stays on your PC,
reached from EC2 through a reverse SSH tunnel (port 1234 never opened):

```bash
# On your PC (EC2 IP changes on start; LM Studio must be running):
ssh -i <key.pem> -N -R 1234:localhost:1234 ec2-user@<EC2-PUBLIC-IP>
```

Deploy (see `deploy/`):

```bash
scp -i <key.pem> -r . ec2-user@<EC2-PUBLIC-IP>:~/RAG-4i-Cloud
# or: bash deploy/setup_ec2.sh  (on the instance, as ec2-user)
sudo cp deploy/env.ec2.example .env   # then set DB_PASSWORD in .env
sudo systemctl start rag4i-cloud.service
```

`deploy/` holds `setup_ec2.sh`, `rag4i-cloud.service` (systemd,
restart-on-failure), `streamlit_config.toml` (port 8501), and
`env.ec2.example`. The EC2 instance needs an S3 least-privilege IAM role
(read-only on the bucket prefix) attached before first run.

## Local LM Studio requirements

- Run LM Studio server at `LLM_BASE_URL` (default `http://localhost:1234/v1`).
- Load any local model (`LLM_MODEL` name is opaque to LM Studio).
- Temperature `0.0` for precise, non-creative answers.

## Model benchmark (experimental, not a production decision)

`tests/benchmarks/` holds a controlled evaluation harness (frozen corpus,
40 scenarios with gold answers, deterministic grader; continuation point in
`docs/SESSION_HANDOFF_2026-09-11.md`). The current working inference test
path is **AWS Bedrock Mantle, ap-south-1 (Mumbai)** via SigV4 from the EC2
instance role — Bedrock Converse is blocked for this account (entitlement,
not IAM).

- Five arms compare Qwen3-32B/235B, GPT-OSS-120B, Mistral Large 3 and
  Devstral 2 across planner/reviewer/answer roles. Several candidates
  (e.g. Gemma 4 26B, Luna) were excluded for in-region availability or
  entitlement reasons — see the handoff doc. Nothing below is a verdict.
- `DRY_RUN=1 python tests/benchmarks/run_full.py` is the free local
  pre-flight (doubles, writes to a separate `*-DRY` directory, no AWS).
- Live runs are explicitly gated (`BENCHMARK_LIVE_BEDROCK=1`) and capped by
  an $8 spend tracker checked before every call.
- Status (2026-09-11): short exploratory run (`20260911-short01`) complete;
  full run in progress — scenarios S01–S09 recorded, paused before S10.
  Judge not run; **no final production model has been selected**.
- Result artifacts stay uncommitted by policy under
  `tests/benchmarks/results/`. `run_short.py` remains frozen as the
  historical short01 runner.

## What is NOT implemented (later phases)

Lambda, cloud GPUs, vLLM, Docker, Kubernetes, auth,
multi-user permissions, OCR. A production model rollout is also not
implemented: the benchmark is ongoing and no winner has been selected.
Hybrid search/BM25, reranking, LLM query
rewriting, and router coverage hardening are scoped in
`docs/adr-retrieval-upgrades.md` (deferred deliberately, not forgotten).
(RDS PostgreSQL/pgvector second provider and EC2 SSH tunnel for local
access are covered above in Phase 2; RDS stays private.)

## Tests

```bash
python -m unittest discover -s tests -v
```

Covers config, ingestion/chunking, metadata, retrieval, provider
abstractions, the Phase 5 planner/reviewer adapters, and benchmark harness
logic. PostgreSQL is mocked. Heavy deps (Chroma/model/LLM server) are mocked or
skipped so the suite runs without a GPU or server. The real-RDS test is
opt-in only and needs the SSH tunnel up:

```bash
$env:RUN_PG_INTEGRATION="1"
python -m unittest tests.integration.test_postgres_rds -v
```
