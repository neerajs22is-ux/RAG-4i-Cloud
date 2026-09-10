# Phase 5 Session Scope Design (metadata contract + lifecycle)

Status: **scope primitives implemented (5B)** in `document_scope.py` +
both vector providers; **session uploads implemented (5C)** in
`session_uploads.py` + Streamlit UI. Retention cleanup and planner (5D)
still pending.

## 1. Metadata contract (additive columns on `chunks`)

| Column | persistent | session |
|---|---|---|
| `scope` | `'persistent'` | `'session'` |
| `session_id` | `NULL` | opaque 128-bit id (hex, `secrets.token_hex(16)`) |
| `document_id` | path-derived, unchanged in 5B (content-derived move deferred, see §6) | same derivation |
| `chunk_id` | unchanged content-hash IDs (scope excluded from derivation) | unchanged |
| `content_hash` | deferred to 5C (dedupe matters at upload time) | deferred |

Chroma: same keys in metadata dicts. Both providers, same rule
(parity, as today).

## 2. Combined retrieval

Persistent corpus is always visible. Session corpus is visible only
with the current session bound:

```text
scope = 'persistent'
OR (scope = 'session' AND session_id = :current_session_id)
```

Enforced INSIDE `search()` / `chunks_for_source()` / `list_sources()`
of both providers. Unbound/missing session ⇒ persistent-only. There is
no code path that reads session rows without the filter.

## 3. Isolation expectations

- User/session A sees: persistent + A-session documents.
- User/session B sees: persistent + B-session documents.
- B MUST NEVER retrieve A-session chunks (hard acceptance gate,
  probed by benchmark cases `session-upload-*`).

## 4. Lifecycle (conceptual)

```text
upload → validate → indexed → available → detach → retention → delete
```

- **upload/validate**: type/size/count caps (section 5), virus/corrupt
  handling, per-session object prefix.
- **indexed**: same loader → 1000/200 → MiniLM pipeline as persistent
  (invariant: session uploads must use the existing ingestion pipeline).
- **available**: rows carry the session id; in-filter immediately.
- **detach**: New Chat / explicit remove / expiry flips availability
  (fast flag or id-list) WITHOUT waiting for row deletion; retrieval
  sees zero session rows from that instant.
- **retention**: window per policy decision (default proposal: 24h
  pilot; PENDING POLICY).
- **delete**: `DELETE WHERE session_id=..` + storage prefix removal;
  backups age out per policy; no orphaned rows/objects.

## 5. Session upload security limits (implemented, 5C)

| Limit | Pilot value | Status |
|---|---|---|
| Allowed types | PDF only (extension hint + `%PDF-` magic bytes) | DECIDED (5C) |
| Max file size | 10 MB (`MAX_FILE_BYTES`) | DECIDED (5C) |
| Max files / session | 5 (`MAX_FILES_PER_SESSION`) | DECIDED (5C) |
| Max total session storage | 50 MB (`MAX_SESSION_BYTES`) | DECIDED (5C) |
| Corrupt/invalid PDF | reject with per-file reason (existing report shape) | DECIDED (5C) |
| Password-protected PDF | reject ("encrypted PDFs unsupported") | DECIDED (5C) |
| Upload rate/abuse | per-session quotas above + single-operator pilot; no public endpoint | PENDING POLICY |
| Retention window | 24h proposal stands; no cleanup job yet (vectors/objects retained) | PENDING POLICY |
| Delete semantics | detach instant (New Chat rotates the binding); physical delete async ≤ retention end | Delete NOT implemented |

Scanned-image PDFs (no extractable text) already report zero-text
through the existing diagnostics — session uploads inherit that.

## 7. Phase 5C implementation notes

- **S3 key convention**: `sessions/<session-id>/uploads/<stem>-<sha12>.pdf`
  at bucket root — deliberately OUTSIDE the persistent corpus prefix so
  session objects never appear in corpus listings. Stem is a sanitized
  basename (no paths, bounded charset); the 12-hex content suffix makes
  keys collision-resistant. Same layout is mirrored for local staging
  under `SESSION_STORAGE_DIR/<session-id>/uploads/`.
- **Identity/deduplication**: `document_id = sha1("session-doc:" +
  session_id + content_sha256 + safe_name)`. Same (session, bytes)
  re-upload → identical IDs → upsert is a no-op (retry-safe, duplicate
  shortcut returns `already-indexed`). Different sessions or bytes fork
  deterministically; persistent path-derived IDs can never collide.
  No contents in registry/results/telemetry (sizes/hashes/names only).
- **Ordering** (`ingest_session_upload`): cheap validation (type/magic/
  size/quota/session) → duplicate shortcut → S3/object write → load via
  shared `PyPDFLoader` primitive → 1000/200 chunk → MiniLM →
  `build_index(..., session_id)` → scoped verify (own `document_id`
  present) → `ready`. Build failures best-effort delete the staged
  object; verify failures keep it for diagnosis; both report `failed`
  (never ready) with per-file reasons. No LLM in ingestion.
- **Lifecycle implemented**: binding minted per browser session
  (`ensure_session_id`), rebound on snapshot restore (same lineage),
  rotated on New Chat (detach; vectors kept). Retention/physical
  deletion: placeholder only (no cleanup job, no expiry scan).
- **UI**: sidebar “Session documents” (uploader + per-file progress +
  results + registry list). No scope selector, no storage internals,
  no session IDs shown. Every question searches persistent + current
  session automatically; starter cache is session-keyed.
- **Security assumptions**: browser filenames are labels only; S3 creds
  stay server-side (instance role); isolation holds even if
  `session_state`/snapshot JSON is hand-edited (provider-side filter);
  quota reconciliation counts unknown vector rows at max-file weight;
  EC2 IAM needs `PutObject`/`DeleteObject` on `sessions/*` (see
  `deploy/SECURITY.md`; local mode needs nothing new).

## 6. Migration behavior (implemented, 5B)

- PostgreSQL: `ensure_schema()` runs `ADD COLUMN IF NOT EXISTS` for
  `scope` (`DEFAULT 'persistent'`, which backfills existing rows) and
  `session_id`. Fresh tables include both columns. Reads retry once
  after migrating on pre-scope tables (`42703`). Existing rows map
  deterministically to persistent; nothing is deleted or re-embedded.
- Chroma: writes stamp scope since 5B; reads run a one-time,
  metadata-only backfill (missing/invalid scope → `persistent`,
  stray `session_id` dropped, session rows untouched, all logged).
  If the backfill fails, unbound reads degrade to the exact 4.12
  unfiltered behavior (logged) while session-bound reads refuse
  loudly — never silent, never leaking.
- `chunk_id` derivation excludes scope, so re-indexing never
  duplicates; note: re-indexing byte-identical content under a
  session re-scopes those chunk IDs (upsert semantics) instead of
  forking them — deterministic, no duplicates.
- New Chat stays logical DETACH: it clears conversation state only and
  never deletes vectors; dropping the session binding hides session
  rows instantly. Physical retention/deletion is 5C work (NOT
  implemented).
