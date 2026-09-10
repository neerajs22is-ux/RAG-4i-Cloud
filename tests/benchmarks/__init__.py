"""Deterministic benchmark scaffold for Phase 5 (5.0: framework only).

Compares Baseline (4.12 local Qwen) vs staged 5A/5D configurations using
the SAME cases and the SAME deterministic doubles by default. Live
providers are supported only through explicit injection (never required
by tests). Session-upload cases are fixtures: skipped until 5C exists.
No document text is stored in results (counts/ids only).
"""
