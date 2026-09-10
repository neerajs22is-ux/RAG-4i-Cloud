# Phase 5 Session Scope Design (metadata contract + lifecycle)

Status: **scope primitives implemented (5B)** in `document_scope.py` +
both vector providers; session uploads (5C) still pending. Anything
below marked “NOT implemented” stays that way.

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

## 5. Session upload security limits (defined, NOT implemented)

| Limit | Proposed pilot value | Status |
|---|---|---|
| Allowed types | PDF only (`application/pdf` + magic-byte check) | DECIDED (design) |
| Max file size | 10 MB | PENDING POLICY |
| Max files / session | 5 | PENDING POLICY |
| Max total session storage | 50 MB | PENDING POLICY |
| Corrupt/invalid PDF | reject with per-file reason (existing report shape) | DECIDED (design) |
| Password-protected PDF | reject ("encrypted PDFs unsupported") | DECIDED (design) |
| Upload rate/abuse | per-session cooldown + size quotas; no public endpoint in pilot | PENDING POLICY |
| Retention window | 24h, then physical delete | PENDING POLICY |
| Delete semantics | detach instant; physical delete async ≤ retention end | DECIDED (design) |

Scanned-image PDFs (no extractable text) already report zero-text
through the existing diagnostics — session uploads inherit that.

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
