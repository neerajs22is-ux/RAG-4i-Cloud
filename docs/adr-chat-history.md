# ADR: Chat History Persistence for RAG-4i-Cloud

Status: design only — no persistence implemented.

## 1. Current in-memory session behaviour

Conversation lives in Streamlit `session_state` (`messages` list plus a
bounded `ConversationMemory` window used only for intent/follow-up
reasoning). It vanishes on page reload, server restart, or session expiry.
No conversation content touches disk, S3, or PostgreSQL today.

## 2. Browser-local storage option

Persist transcripts in `localStorage` via a small Streamlit component:
no server changes, works offline, per-browser history. Downsides: tied
to one browser/device, no sharing, user can wipe it unknowingly, and
sensitive legal text sits unencrypted on the endpoint.

## 3. Server-side per-user persistence option

New `conversations`/`messages` tables in the existing RDS database
(or an S3 prefix per user), written on each turn and loaded at login.
Enables cross-device history, retention control, and audit. Costs: schema
migration, storage growth, backup scope, and a hard dependency on user
identity (see 7).

## 4. Privacy implications (legal documents/conversations)

Chat transcripts can quote client confidences verbatim (answers embed
retrieved chunks). Any persistence must treat transcripts with the same
confidentiality as the source documents: encryption at rest, no logging
of message content in application logs, and no third-party analytics.

## 5. Retention/deletion implications

Legal matter work needs defensible deletion: per-conversation delete,
per-matter expiry, and admin purge must remove embeddings-adjacent cached
text as well as rows. "New chat" today only clears memory; a persisted
design must make delete mean delete everywhere (DB rows, backups per
policy, no orphaned S3 objects).

## 6. Multi-user isolation

Server-side history requires strict tenant isolation (user_id on every
row, row-level checks on every read/write). The current single-user
pilot has no identity at all; bolting storage on without it would leak
conversations across users.

## 7. Authentication/authorization dependency

There is no login today. Persisted history therefore depends on a prior
auth decision (SSO/OIDC preferred over passwords). Until auth exists,
only browser-local or no persistence is defensible.

## 8. EC2 pilot assumptions

Single trusted operator, test/non-sensitive documents, no auth, private
RDS/S3, ephemeral sessions acceptable. Persistence buys little here and
adds secrets/backups surface.

## 9. Recommendation for the current pilot

Keep in-memory sessions only. Optionally add browser-local opt-in later
as a pure-frontend convenience. Do NOT build server-side history for the
pilot: without auth, isolation, and a retention policy it creates risk
without user value.

## 10. What must change before company-wide deployment

SSO login, per-user isolation in every query path, encrypted server-side
store with retention/delete workflows, updated security review (threat
model covers transcripts as client data), and user-visible controls for
viewing/exporting/deleting history. This ADR must be revisited then —
it must not automatically become the production design.
