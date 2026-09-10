# Phase 5 Session Scope Design (metadata contract + lifecycle)

Design only — no upload code in 5.0.

## 1. Metadata contract (additive columns on `chunks`)

| Column | persistent | session |
|---|---|---|
| `scope` | `'persistent'` | `'session'` |
| `session_id` | `NULL` | opaque 128-bit id (hex, `secrets.token_hex(16)`) |
| `document_id` | content-derived (5B changes derivation, additive migration) | same derivation |
| `chunk_id` | unchanged content-hash IDs | unchanged |
| `content_hash` | NEW: sha256 of normalized chunk text (dedupe + audit) | same |

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
