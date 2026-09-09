# ADR: Deferred Retrieval Upgrades (Tier 2 — NOT implemented)

Status: scoped future work. None of the below is implemented; the
deterministic pipeline (multi-form retrieval + max-pooling, threshold 0.3)
remains the retrieval behavior.

## 1. LLM query rewriting (deferred)

Rewriting user queries with a model call before retrieval could help
vague/keyword-heavy legal questions (e.g. bare clause numbers). Deferred
because it adds a model call plus latency to every query and needs its
own prompt-safety review (a rewriting model must not answer, leak, or
narrow the question). Explicitly excluded from `build_query_forms`
unless that review happens. A curated synonym form was measured against
MiniLM and consistently scored lowest, so it was deliberately not
shipped either — evidence beats intuition here.

## 2. Reranking / hybrid BM25 (deferred)

A cross-encoder reranker or BM25 hybrid would help keyword-heavy queries.
Deferred because parity is mandatory: it must be implemented in BOTH
vector providers (Chroma and pgvector) so `VECTOR_STORE` stays a pure
configuration switch. The current max-pooling over deterministic forms
is the approved lightweight substitute.

## 3. Router coverage hardening (deferred)

Broader phrasings and non-English frames for the intent observer.
Deferred because every new frame risks hijacking document questions;
coverage grows only with measured false-route cases and semantic-group
tests, never bare phrase lists. The uncertain→DOCUMENT_QUERY default
stays the safety net.
