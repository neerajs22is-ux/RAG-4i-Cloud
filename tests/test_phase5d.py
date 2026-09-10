"""Phase 5D tests: planner, validator, gate, cache, fallback, parity.

Mocks/fakes only: no live AWS calls, no model downloads. The reasoning
path is exercised through injected fakes; REASONING_ENABLED=0 parity is
asserted exactly.
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SID_A = "a" * 32
SID_B = "b" * 32
FILES = ["lease.pdf", "contract.pdf"]


def valid_plan(**over):
    plan = {"schema_version": 1, "workflow": "normal",
            "documents": [], "topics": ["lock-in"],
            "retrieval_queries": ["lock-in period lease"],
            "fields": [], "comparison_dimensions": [],
            "needs_clarification": {"required": False, "question": ""}}
    plan.update(over)
    return plan


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})
        self.id = None


class FakeStore:
    """Scope-aware word-overlap store recording searched forms."""

    def __init__(self):
        self.rows = []
        self.forms = []

    def add(self, content, fname, score=0.9, scope="persistent", sid=None,
            cid=None, page=0):
        self.rows.append({"content": content, "file": fname, "score": score,
                          "scope": scope, "sid": sid,
                          "cid": cid or f"c{len(self.rows)}", "page": page})

    def _visible(self, r, session_id):
        if r["scope"] == "session":
            return r["sid"] is not None and r["sid"] == session_id
        return True

    def _doc(self, r):
        return FakeDoc(r["content"],
                       {"source_path": "/t/" + r["file"],
                        "source": "/t/" + r["file"],
                        "file_name": r["file"], "page": r["page"],
                        "document_id": "d-" + r["file"],
                        "chunk_id": r["cid"]})

    def search(self, query, k=5, session_id=None):
        from document_scope import validate_session_id
        if session_id is not None:
            validate_session_id(session_id)
        self.forms.append(query)
        words = [w for w in (query or "").lower().split() if len(w) >= 4]
        out = []
        for r in self.rows:
            if not self._visible(r, session_id):
                continue
            if any(w in r["content"].lower() for w in words):
                out.append((self._doc(r), r["score"]))
        return out[:k]

    def chunks_for_source(self, file_name, limit=8, session_id=None):
        return [(self._doc(r), None) for r in self.rows
                if r["file"] == file_name
                and self._visible(r, session_id)][:limit]

    def list_sources(self, limit=50, session_id=None):
        seen = []
        for r in self.rows:
            if r["file"] not in seen and self._visible(r, session_id):
                seen.append(r["file"])
        return [{"file_name": f, "document_id": "d-" + f}
                for f in seen[:limit]]


class FakeLLM:
    def __init__(self, text="ANSWER"):
        self.text = text

    def generate(self, context, question, prompt_template):
        return self.text

    def generate_stream(self, context, question, prompt_template):
        yield self.text

    def is_reachable(self, timeout=3.0):
        return True

    @property
    def describe(self):
        return "Fake"


def lease_store():
    store = FakeStore()
    store.add("The lock-in period in the lease deed is 36 months.",
              "lease.pdf")
    return store


def _cfg(**over):
    import config
    base = dict()
    base.update(over)
    return config.AppConfig(**base)


class FakePlanner:
    """Scripted planner double: returns queued (raw, meta) pairs."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def plan(self, query_text, *, available_files=None, session_id=None,
             context=None):
        self.calls.append({"query": query_text, "files": available_files,
                           "session": session_id, "context": context})
        if not self.script:
            raise AssertionError("planner called more often than scripted")
        return self.script.pop(0)


def canned_meta(**over):
    meta = {"provider": "fake", "model": "fake-1", "latency_ms": 3,
            "failure": None, "tokens_in": None, "tokens_out": None}
    meta.update(over)
    return meta


def bundle_for(planner):
    return {"planner": planner, "cache": None, "provider": "fake",
            "model": "fake-1", "timeout_s": 5,
            "_meta_base": {"reasoning_used": True,
                           "reasoning_provider": "fake",
                           "reasoning_model": "fake-1"}}


def stable_result(answer, sources, info):
    """Comparable projection: volatile timings/provider identity removed."""
    kept = {k: info.get(k) for k in ("answer", "retrieved",
                                     "needs_retrieval", "support_level",
                                     "failed", "workflow", "label",
                                     "fallback_template",
                                     "fallback_question")}
    kept["timings_keys"] = sorted((info.get("timings") or {}).keys())
    provider = info.get("provider")
    kept["provider"] = provider.describe if hasattr(
        provider, "describe") else provider
    return answer, sources, kept


class TestParityDisabled(unittest.TestCase):
    def test_bundle_none_by_default(self):
        from query_planner import get_planner_bundle
        import config
        self.assertIsNone(get_planner_bundle(config.load_config()))
        self.assertIsNone(get_planner_bundle(None))
        self.assertIsNone(get_planner_bundle(_cfg(reasoning_enabled="0")))

    def test_no_langchain_import_when_disabled(self):
        import sys
        from query_planner import get_planner_bundle
        import config
        get_planner_bundle(config.load_config())
        self.assertNotIn("langchain_aws", sys.modules)

    def test_query_parity(self):
        from query_planner import query_planned
        from workflows import query_workflow
        cfg = _cfg()
        q = "What is the lock-in period in the lease deed?"
        expected = query_workflow(q, config=cfg, vector_store=lease_store(),
                                  llm_provider=FakeLLM("A"))
        got = query_planned(q, config=cfg, vector_store=lease_store(),
                            llm_provider=FakeLLM("A"), bundle=None)[:3]
        self.assertEqual(got, expected)

    def test_stream_parity(self):
        from query_planner import stream_planned_answer
        from workflows import stream_workflow_answer
        cfg = _cfg()
        q = "What is the lock-in period in the lease deed?"
        info1, stream1 = stream_workflow_answer(
            q, config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("A"))
        info2, stream2, meta = stream_planned_answer(
            q, config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("A"), bundle=None)
        self.assertEqual(list(stream1), list(stream2))
        self.assertEqual(info1, info2)
        self.assertFalse(meta["reasoning_used"])

    def test_deterministic_bundle_parity(self):
        from query_planner import DeterministicPlanner, PlanCache, query_planned
        from workflows import query_workflow
        cfg = _cfg()
        for q in ["What is the lock-in period in the lease deed?",
                  "Compare the lock-in clauses in lease.pdf and contract.pdf.",
                  "List all notice periods.",
                  "Summarize lease.pdf."]:
            store = lease_store()
            store.add("The service contract renews annually.", "contract.pdf")
            expected = query_workflow(q, config=cfg, vector_store=store,
                                      llm_provider=FakeLLM("A"))
            bundle = bundle_for(DeterministicPlanner())
            bundle["cache"] = PlanCache()
            got = query_planned(q, config=cfg, vector_store=store,
                                llm_provider=FakeLLM("A"),
                                bundle=bundle)
            with self.subTest(q=q):
                self.assertEqual(stable_result(*got[:3]),
                                 stable_result(*expected))


class TestDeterministicPlanner(unittest.TestCase):
    def test_wraps_detection(self):
        from query_planner import DeterministicPlanner
        from workflows import detect_workflow
        planner = DeterministicPlanner()
        for q in ["What is the lock-in period?",
                  "Compare a.pdf and b.pdf.",
                  "List all notice periods.",
                  "Summarize lease.pdf."]:
            with self.subTest(q=q):
                plan, meta = planner.plan(q, available_files=FILES)
                self.assertEqual(plan["workflow"],
                                 detect_workflow(q)["workflow"])
                self.assertEqual(plan["retrieval_queries"], [])
                self.assertFalse(
                    plan["needs_clarification"]["required"])
                self.assertIsNone(meta["failure"])

    def test_output_always_valid(self):
        from query_planner import DeterministicPlanner, validate_plan
        planner = DeterministicPlanner()
        plan, _meta = planner.plan("List all notice periods and deadlines.",
                                   available_files=FILES)
        valid, failure = validate_plan(plan, available_files=FILES)
        self.assertIsNone(failure)
        self.assertIsNotNone(valid)


class TestEscalationGate(unittest.TestCase):
    def test_positives(self):
        from query_planner import requires_reasoning_plan
        cases = [
            ("Compare the notice periods.", "document_query",
             "ambiguous_target"),
            ("Summarize the document.", "document_query",
             "ambiguous_target"),
            ("List dates, deadlines and notice periods.", "document_query",
             "complex_extraction"),
            ("What is the lock-in period? Who are the parties?",
             "document_query", "compound"),
            ("List all notice periods and extract payment deadlines.",
             "document_query", "complex_extraction"),
        ]
        for query, intent, reason in cases:
            with self.subTest(query=query):
                escalate, got = requires_reasoning_plan(
                    query, intent, available_files=FILES)
                self.assertTrue(escalate, query)
                self.assertEqual(got, reason)

    def test_followup_multi_form(self):
        from query_planner import requires_reasoning_plan
        escalate, reason = requires_reasoning_plan(
            "what about renewal and payment terms?", "document_followup",
            available_files=FILES)
        self.assertTrue(escalate)
        self.assertEqual(reason, "multi_form_retrieval")

    def test_negatives(self):
        from query_planner import requires_reasoning_plan
        cases = [
            ("Hi", "conversation"),
            ("What can you do?", "capability"),
            ("What is the capital of France?", "out_of_scope"),
            ("What is the lock-in period in the lease deed?",
             "document_query"),
            ("Compare the lock-in clauses in lease.pdf and contract.pdf.",
             "document_query"),
            ("List all notice periods.", "document_query"),
            ("Summarize lease.pdf.", "document_query"),
            ("And payment?", "document_followup"),
        ]
        for query, intent in cases:
            with self.subTest(query=query):
                escalate, reason = requires_reasoning_plan(
                    query, intent, available_files=FILES)
                self.assertFalse(escalate, query)
                self.assertIn(reason, ("non-document", "ordinary"))

    def test_reason_categories_closed(self):
        from query_planner import ESCALATION_REASONS, requires_reasoning_plan
        for query, intent in [("hi", "conversation"),
                              ("Compare x and y in a.pdf and b.pdf.",
                               "document_query"),
                              ("Summarize it.", "document_query")]:
            _esc, reason = requires_reasoning_plan(query, intent,
                                                   available_files=FILES)
            self.assertIn(reason, ESCALATION_REASONS)

    def test_reserved_reasons_never_emitted(self):
        from query_planner import (RESERVED_ESCALATION_REASONS,
                                   requires_reasoning_plan)
        self.assertEqual(RESERVED_ESCALATION_REASONS,
                         {"comparison", "other"})
        corpus = ["hi", "What is the lock-in period?",
                  "Compare the lock-in clauses in lease.pdf and contract.pdf.",
                  "Compare the notice periods.",
                  "List dates, deadlines and notice periods.",
                  "List all notice periods.",
                  "Summarize lease.pdf.", "Summarize the document.",
                  "What is the lock-in period? Who are the parties?",
                  "List all notice periods and extract payment deadlines.",
                  "Tell me about this document.",
                  "And payment?", "what about renewal and payment terms?",
                  "What does lease.pdf say about renewal?"]
        for query in corpus:
            for intent in ("conversation", "capability", "document_query",
                           "document_followup", "out_of_scope"):
                _esc, reason = requires_reasoning_plan(
                    query, intent, available_files=FILES)
                with self.subTest(query=query, intent=intent):
                    self.assertNotIn(reason, RESERVED_ESCALATION_REASONS)

    def test_no_answer_generation_from_planner(self):
        from query_planner import LLMQueryPlanner
        calls = []

        class FakeChat:
            def with_structured_output(self, schema):
                calls.append(("structured", schema))
                return self

            def invoke(self, prompt_text):
                calls.append(("invoke", prompt_text))
                return valid_plan()

        self.assertFalse(hasattr(LLMQueryPlanner, "generate"))
        self.assertFalse(hasattr(LLMQueryPlanner, "generate_stream"))
        self.assertFalse(hasattr(LLMQueryPlanner, "generate_answer"))
        from query_planner import bedrock_structured_fn
        fn = bedrock_structured_fn(FakeChat())
        planner = LLMQueryPlanner(fn, timeout_s=5)
        raw, meta = planner.plan("What is the lock-in?",
                                 available_files=FILES)
        self.assertEqual(raw["schema_version"], 1)
        kinds = [k for k, _v in calls]
        self.assertIn("structured", kinds)
        self.assertIn("invoke", kinds)
        # Prompt constrains the model to plans, never answers.
        prompt = calls[1][1]
        self.assertIn("JSON", prompt)
        self.assertIn("Never answer", prompt)


class TestPlanSchema(unittest.TestCase):
    def test_version(self):
        from query_planner import validate_plan
        for bad in [{**valid_plan(), "schema_version": 2},
                    {**valid_plan(), "schema_version": "1"},
                    {k: v for k, v in valid_plan().items()
                     if k != "schema_version"}]:
            _plan, failure = validate_plan(bad, available_files=FILES)
            self.assertEqual(failure, "schema")

    def test_unknown_and_missing_keys(self):
        from query_planner import validate_plan
        bad = valid_plan()
        bad["surprise"] = 1
        _plan, failure = validate_plan(bad, available_files=FILES)
        self.assertEqual(failure, "schema")
        _plan, failure = validate_plan("not-a-dict",
                                       available_files=FILES)
        self.assertEqual(failure, "schema")
        _plan, failure = validate_plan(
            {k: v for k, v in valid_plan().items() if k != "topics"},
            available_files=FILES)
        self.assertEqual(failure, "schema")

    def test_workflow_allowlist(self):
        from query_planner import validate_plan
        for workflow in ["agent", "chat", "", None, 42]:
            _plan, failure = validate_plan(
                valid_plan(workflow=workflow), available_files=FILES)
            self.assertEqual(failure, "workflow")
        for workflow in ["normal", "comparison", "extraction", "summary"]:
            _plan, failure = validate_plan(
                valid_plan(workflow=workflow), available_files=FILES)
            self.assertIsNone(failure)

    def test_query_bounds(self):
        from query_planner import validate_plan
        _plan, failure = validate_plan(
            valid_plan(retrieval_queries=["q%d" % i for i in range(5)]),
            available_files=FILES)
        self.assertEqual(failure, "schema")
        _plan, failure = validate_plan(
            valid_plan(retrieval_queries=["x" * 201]),
            available_files=FILES)
        self.assertEqual(failure, "schema")
        plan, failure = validate_plan(
            valid_plan(retrieval_queries=[]), available_files=FILES)
        self.assertIsNone(failure)
        self.assertEqual(plan["retrieval_queries"], [])

    def test_fields_allowlisted(self):
        from query_planner import validate_plan
        _plan, failure = validate_plan(
            valid_plan(fields=["notice periods", "drop table"]),
            available_files=FILES)
        self.assertEqual(failure, "fields")
        plan, failure = validate_plan(
            valid_plan(fields=["notice periods"]), available_files=FILES)
        self.assertIsNone(failure)

    def test_dimensions_validated(self):
        from query_planner import validate_plan
        _plan, failure = validate_plan(
            valid_plan(comparison_dimensions=["favorable"]),
            available_files=FILES)
        self.assertEqual(failure, "dimensions")
        plan, failure = validate_plan(
            valid_plan(comparison_dimensions=["notice", "dates"]),
            available_files=FILES)
        self.assertIsNone(failure)

    def test_clarification_shape(self):
        from query_planner import validate_plan
        base = {"required": False, "question": ""}
        plan, failure = validate_plan(
            valid_plan(needs_clarification=dict(base)),
            available_files=FILES)
        self.assertIsNone(failure)
        # required=false with text is inconsistent.
        _p, failure = validate_plan(
            valid_plan(needs_clarification={"required": False,
                                            "question": "Which?"}),
            available_files=FILES)
        self.assertEqual(failure, "clarification")
        # required=true needs bounded text anchored in real context.
        _p, failure = validate_plan(
            valid_plan(needs_clarification={"required": True,
                                            "question": ""}),
            available_files=FILES)
        self.assertEqual(failure, "clarification")
        _p, failure = validate_plan(
            valid_plan(needs_clarification={"required": True,
                                            "question": "x" * 301}),
            available_files=FILES)
        self.assertEqual(failure, "clarification")
        _p, failure = validate_plan(
            valid_plan(needs_clarification={
                "required": True,
                "question": "Which ghost document do you mean?"}),
            available_files=FILES)
        self.assertEqual(failure, "clarification")
        plan, failure = validate_plan(
            valid_plan(documents=["lease.pdf"],
                       topics=["lock-in"],
                       needs_clarification={
                           "required": True,
                           "question": "Which lock-in terms in lease.pdf?"}),
            available_files=FILES)
        self.assertIsNone(failure)
        self.assertEqual(plan["needs_clarification"]["required"], True)
        # Extra keys and non-bool required rejected.
        _p, failure = validate_plan(
            valid_plan(needs_clarification={"required": True,
                                            "question": "Which in lease.pdf?",
                                            "extra": 1}),
            available_files=FILES)
        self.assertEqual(failure, "clarification")
        _p, failure = validate_plan(
            valid_plan(needs_clarification={"required": "yes",
                                            "question": ""}),
            available_files=FILES)
        self.assertEqual(failure, "clarification")

    def test_document_references(self):
        from query_planner import validate_plan
        for docs in [["ghost.pdf"], ["lease.pdf", "ghost.pdf"],
                     ["../etc/passwd"], ["a/b.pdf"], [""],
                     ["lease.pdf"] * 5, "not-a-list"]:
            _plan, failure = validate_plan(
                valid_plan(documents=docs), available_files=FILES)
            self.assertIsNotNone(failure, docs)
        plan, failure = validate_plan(
            valid_plan(documents=["lease.pdf", "contract.pdf"]),
            available_files=FILES)
        self.assertIsNone(failure)


class TestBoundedInput(unittest.TestCase):
    def test_prompt_bounds_and_hygiene(self):
        from query_planner import build_planner_prompt
        prompt = build_planner_prompt(
            "Q" * 1000,
            available_files=[f"doc{i}.pdf" for i in range(30)],
            session_bound=True,
            context=[{"role": "user", "content": "U" * 1000},
                     {"role": "assistant", "content": "SECRET-CHUNK-TEXT"},
                     {"role": "user", "content": "second q"}])
        self.assertIn("JSON", prompt)
        self.assertNotIn("SECRET-CHUNK-TEXT", prompt)
        self.assertNotIn(SID_A, prompt)
        self.assertNotIn("/home/", prompt)
        self.assertLessEqual(len(prompt), 4000)
        # Only two user turns survive, truncated.
        self.assertIn("second q", prompt)
        self.assertNotIn("U" * 100, prompt)

    def test_planner_never_sees_evidence(self):
        from query_planner import LLMQueryPlanner
        seen = {}

        def structured(prompt_text, schema):
            seen["prompt"] = prompt_text
            return valid_plan()

        planner = LLMQueryPlanner(structured, timeout_s=5)
        planner.plan("What is the lock-in?",
                     available_files=FILES, session_id=SID_A,
                     context=[{"role": "user", "content": "hi"}])
        self.assertNotIn("lease lock-in period", seen["prompt"])
        self.assertNotIn(SID_A, seen["prompt"])


class TestPlanCache(unittest.TestCase):
    def test_reuse_and_isolation(self):
        from query_planner import PlanCache
        cache = PlanCache()
        key = PlanCache.cache_key("What is lock-in?", FILES, SID_A,
                                  "bedrock", "m")
        self.assertIsNone(cache.get(key))
        plan = valid_plan()
        cache.put(key, plan)
        self.assertEqual(cache.get(key), plan)
        # Any context change misses.
        self.assertIsNone(PlanCache.cache_key("What is lock-in?", FILES,
                                              SID_B, "bedrock", "m")
                          and cache.get(PlanCache.cache_key(
                              "What is lock-in?", FILES, SID_B,
                              "bedrock", "m")))
        self.assertIsNone(cache.get(PlanCache.cache_key(
            "What is renewal?", FILES, SID_A, "bedrock", "m")))
        self.assertIsNone(cache.get(PlanCache.cache_key(
            "What is lock-in?", ["other.pdf"], SID_A, "bedrock", "m")))
        self.assertIsNone(cache.get(PlanCache.cache_key(
            "What is lock-in?", FILES, SID_A, "bedrock", "other")))

    def test_bounded_lru(self):
        from query_planner import PlanCache
        cache = PlanCache(max_entries=4)
        keys = [PlanCache.cache_key(f"q{i}?", FILES, SID_A, "b", "m")
                for i in range(6)]
        for key in keys:
            cache.put(key, valid_plan())
        self.assertEqual(len(cache), 4)
        self.assertIsNone(cache.get(keys[0]))
        self.assertIsNotNone(cache.get(keys[-1]))

    def test_cached_plan_reused_without_recall(self):
        from query_planner import PlanCache, query_planned
        planner = FakePlanner([(valid_plan(
            workflow="normal",
            retrieval_queries=["lock-in period lease deed"]), canned_meta())])
        bundle = bundle_for(planner)
        real_cache = PlanCache()
        bundle["cache"] = real_cache
        cfg = _cfg()
        # Compound query escalates, so the planner is actually consulted.
        q = "What is the lock-in period? Who are the parties?"
        query_planned(q, config=cfg, vector_store=lease_store(),
                      llm_provider=FakeLLM("A"), bundle=bundle)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(len(real_cache), 1)
        query_planned(q, config=cfg, vector_store=lease_store(),
                      llm_provider=FakeLLM("A"), bundle=bundle)
        self.assertEqual(len(planner.calls), 1)  # cache hit: no recall


COMPOUND_Q = "What is the lock-in period? Who are the parties?"


class TestTimeoutsAndFailures(unittest.TestCase):
    def test_timeout_falls_back(self):
        import time
        from query_planner import LLMQueryPlanner, query_planned
        from workflows import query_workflow

        def slow(prompt_text, schema):
            time.sleep(5)
            return valid_plan()

        planner = LLMQueryPlanner(slow, timeout_s=0.05)
        cfg = _cfg()
        expected = query_workflow(COMPOUND_Q, config=cfg,
                                  vector_store=lease_store(),
                                  llm_provider=FakeLLM("A"))
        got = query_planned(COMPOUND_Q, config=cfg, vector_store=lease_store(),
                            llm_provider=FakeLLM("A"),
                            bundle=bundle_for(planner))
        self.assertEqual(got[:3], expected)
        self.assertEqual(got[3]["planner_failure_category"], "timeout")

    def test_failure_categories_fall_back(self):
        from query_planner import LLMQueryPlanner, query_planned
        from workflows import query_workflow

        class NoCredentialsError(Exception):
            pass

        cases = [
            (NoCredentialsError("nope"), "credentials"),
            (ConnectionError("endpoint down"), "unavailable"),
            (ValueError("not a dict plan"), "unavailable"),
        ]
        cfg = _cfg()
        expected = query_workflow(COMPOUND_Q, config=cfg,
                                  vector_store=lease_store(),
                                  llm_provider=FakeLLM("A"))
        for err, category in cases:
            def failing(prompt_text, schema, _e=err):
                raise _e

            planner = LLMQueryPlanner(failing, timeout_s=5)
            got = query_planned(COMPOUND_Q, config=cfg,
                                vector_store=lease_store(),
                                llm_provider=FakeLLM("A"),
                                bundle=bundle_for(planner))
            with self.subTest(err=err):
                self.assertEqual(got[:3], expected)
                self.assertEqual(got[3]["planner_failure_category"],
                                 category)

    def test_malformed_output_falls_back(self):
        from query_planner import query_planned
        from workflows import query_workflow

        def prose(prompt_text, schema):
            return "The lock-in period is 36 months."

        from query_planner import LLMQueryPlanner
        planner = LLMQueryPlanner(prose, timeout_s=5)
        cfg = _cfg()
        expected = query_workflow(COMPOUND_Q, config=cfg,
                                  vector_store=lease_store(),
                                  llm_provider=FakeLLM("A"))
        _a, _s, _i, meta = query_planned(
            COMPOUND_Q, config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("A"), bundle=bundle_for(planner))
        self.assertEqual((_a, _s), (expected[0], expected[1]))
        self.assertEqual(meta["planner_failure_category"], "malformed")

    def test_schema_invalid_falls_back(self):
        from query_planner import query_planned
        from workflows import query_workflow

        def bad(prompt_text, schema):
            return {"workflow": "agent"}  # unsupported + missing keys

        from query_planner import LLMQueryPlanner
        planner = LLMQueryPlanner(bad, timeout_s=5)
        cfg = _cfg()
        expected = query_workflow(COMPOUND_Q, config=cfg,
                                  vector_store=lease_store(),
                                  llm_provider=FakeLLM("A"))
        _a, _s, _i, meta = query_planned(
            COMPOUND_Q, config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("A"), bundle=bundle_for(planner))
        self.assertEqual((_a, _s), (expected[0], expected[1]))
        self.assertEqual(meta["planner_failure_category"], "schema")

    def test_misconfigured_builder_loud(self):
        from query_planner import build_llm_planner
        with self.assertRaises(ValueError):
            build_llm_planner(_cfg(reasoning_provider="openai",
                                   reasoning_model_id="m"))
        with self.assertRaises(ValueError):
            build_llm_planner(_cfg(reasoning_provider="bedrock",
                                   reasoning_model_id=""))
        with self.assertRaises(ValueError):
            build_llm_planner(_cfg(reasoning_provider="bedrock",
                                   reasoning_model_id="m",
                                   reasoning_timeout_s="soon"))

    def test_no_answer_generation_from_planner(self):
        from query_planner import LLMQueryPlanner
        calls = []

        class FakeChat:
            def with_structured_output(self, schema):
                calls.append(("structured", schema))
                return self

            def invoke(self, prompt_text):
                calls.append(("invoke", prompt_text))
                return valid_plan()

            def generate(self, *a, **k):
                raise AssertionError("planner must not generate answers")

            def generate_stream(self, *a, **k):
                raise AssertionError("planner must not stream answers")

        from query_planner import bedrock_structured_fn
        fn = bedrock_structured_fn(FakeChat())
        planner = LLMQueryPlanner(fn, timeout_s=5)
        raw, meta = planner.plan("What is the lock-in?",
                                 available_files=FILES)
        self.assertEqual(raw["schema_version"], 1)
        kinds = [k for k, _v in calls]
        self.assertIn("structured", kinds)
        self.assertIn("invoke", kinds)
        self.assertNotIn("generate", kinds)


class TestRetrievalExecution(unittest.TestCase):
    def test_augmentation_reaches_backend(self):
        import backend
        store = lease_store()
        store.add("Zebra stripe pattern savanna study.", "zebra.pdf")
        got = backend.retrieve_documents(
            "What is the lock-in period in the lease deed?",
            config=_cfg(), vector_store=store,
            extra_forms=["zebra stripe pattern"])
        files = {s["file_name"] for s in got}
        self.assertIn("zebra.pdf", files)
        self.assertIn("lease.pdf", files)

    def test_max_pooling_unchanged(self):
        import backend
        store = FakeStore()
        store.add("The lock-in period in the lease deed is 36 months.",
                  "lease.pdf", score=0.9)
        got = backend.retrieve_documents(
            "What is the lock-in period in the lease deed?",
            config=_cfg(), vector_store=store,
            extra_forms=["lock-in period in the lease deed"])
        cids = [s["chunk_id"] for s in got]
        self.assertEqual(len(cids), len(set(cids)))

    def test_k_threshold_preserved(self):
        import backend
        store = FakeStore()
        for i in range(4):
            store.add(f"payment term number {i} alpha", "p.pdf",
                      score=0.9 if i < 2 else 0.1)
        got = backend.retrieve_documents(
            "payment term alpha", config=_cfg(), vector_store=store,
            k=2, threshold=0.3, extra_forms=["payment term"])
        self.assertLessEqual(len(got), 2)
        self.assertTrue(all(s["score"] >= 0.3 for s in got))

    def test_extra_forms_empty_is_noop(self):
        import backend
        store = lease_store()
        plain = backend.retrieve_documents(
            "What is the lock-in period in the lease deed?",
            config=_cfg(), vector_store=store)
        augmented = backend.retrieve_documents(
            "What is the lock-in period in the lease deed?",
            config=_cfg(), vector_store=store, extra_forms=[])
        self.assertEqual(plain, augmented)

    def test_session_isolation_with_extras(self):
        import backend
        store = FakeStore()
        store.add("lease lock-in alpha", "lease.pdf")
        store.add("session bravo secret", "sess.pdf", scope="session",
                  sid=SID_A)
        got = backend.retrieve_documents(
            "session bravo", config=_cfg(), vector_store=store,
            session_id=SID_B, extra_forms=["session bravo secret"])
        self.assertFalse(any(s["file_name"] == "sess.pdf" for s in got))
        got_a = backend.retrieve_documents(
            "session bravo", config=_cfg(), vector_store=store,
            session_id=SID_A, extra_forms=["session bravo secret"])
        self.assertTrue(any(s["file_name"] == "sess.pdf" for s in got_a))

    def test_normal_augmentation_end_to_end(self):
        from query_planner import query_planned
        store = lease_store()
        store.add("Zebra stripe pattern savanna study.", "zebra.pdf")
        planner = FakePlanner([(valid_plan(
            workflow="normal",
            retrieval_queries=["zebra stripe pattern"]), canned_meta())])
        _a, sources, _i, meta = query_planned(
            COMPOUND_Q,
            config=_cfg(), vector_store=store, llm_provider=FakeLLM("A"),
            bundle=bundle_for(planner))
        self.assertTrue(meta["reasoning_used"])
        self.assertEqual(len(planner.calls), 1)
        self.assertIn("zebra.pdf", {s["file_name"] for s in sources})
        self.assertTrue(any("zebra stripe" in f
                            for f in store.forms))

    def test_workflow_mismatch_discarded(self):
        from query_planner import query_planned
        from workflows import query_workflow
        planner = FakePlanner([(valid_plan(workflow="comparison"),
                                canned_meta())])
        cfg = _cfg()
        expected = query_workflow(COMPOUND_Q, config=cfg,
                                  vector_store=lease_store(),
                                  llm_provider=FakeLLM("A"))
        _a, _s, _i, meta = query_planned(
            COMPOUND_Q, config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("A"), bundle=bundle_for(planner))
        self.assertEqual((_a, _s), (expected[0], expected[1]))
        self.assertEqual(meta["planner_failure_category"], "workflow")

    def test_clarification_short_circuits(self):
        from query_planner import query_planned
        planner = FakePlanner([(valid_plan(
            workflow="normal", documents=["lease.pdf"],
            topics=["lock-in"],
            needs_clarification={
                "required": True,
                "question": "Which lock-in terms in lease.pdf?"}),
            canned_meta())])
        answer, sources, info, meta = query_planned(
            COMPOUND_Q,
            config=_cfg(), vector_store=lease_store(),
            llm_provider=FakeLLM("UNREACHABLE-PROOF"),
            bundle=bundle_for(planner))
        self.assertEqual(sources, [])
        self.assertIn("Quick check before I answer:", answer)
        self.assertIn("lease.pdf", answer)
        self.assertTrue(meta["reasoning_used"])


class TestWorkflowsRegression5D(unittest.TestCase):
    def test_all_workflows_unchanged_with_deterministic_bundle(self):
        from query_planner import DeterministicPlanner, PlanCache
        from query_planner import query_planned
        from workflows import query_workflow
        cfg = _cfg()
        store = lease_store()
        store.add("The service contract renews annually.", "contract.pdf")
        queries = ["What is the lock-in period in the lease deed?",
                   "Compare the lock-in clauses in lease.pdf and contract.pdf.",
                   "List all notice periods.",
                   "Summarize lease.pdf."]
        for q in queries:
            s1 = FakeStore()
            for r in store.rows:
                s1.rows.append(dict(r))
            s2 = FakeStore()
            for r in store.rows:
                s2.rows.append(dict(r))
            expected = query_workflow(q, config=cfg, vector_store=s1,
                                      llm_provider=FakeLLM("A"))
            bundle = bundle_for(DeterministicPlanner())
            bundle["cache"] = PlanCache()
            got = query_planned(q, config=cfg, vector_store=s2,
                                llm_provider=FakeLLM("A"),
                                bundle=bundle)
            with self.subTest(q=q):
                self.assertEqual(stable_result(*got[:3]),
                                 stable_result(*expected))


class TestStreamingSanitizer5D(unittest.TestCase):
    def test_planned_stream_sanitized(self):
        from query_planner import stream_planned_answer
        store = lease_store()

        class ThinkLLM(FakeLLM):
            def generate_stream(self, context, question, prompt_template):
                yield "<think>secret reasoning</think>REAL-ANSWER lease."

        _info, stream, _meta = stream_planned_answer(
            "What is the lock-in period in the lease deed?",
            config=_cfg(), vector_store=store,
            llm_provider=ThinkLLM(), bundle=None)
        text = "".join(stream)
        self.assertNotIn("<think>", text)
        self.assertNotIn("secret reasoning", text)
        self.assertIn("REAL-ANSWER", text)

    def test_planned_fallback_same_evidence(self):
        from query_planner import stream_planned_answer

        class FailLLM(FakeLLM):
            def generate_stream(self, context, question, prompt_template):
                raise RuntimeError("stream down")

            def generate(self, context, question, prompt_template):
                return "FALLBACK lease."

        _info, stream, _meta = stream_planned_answer(
            "What is the lock-in period in the lease deed?",
            config=_cfg(), vector_store=lease_store(),
            llm_provider=FailLLM(), bundle=None)
        # Caller-side fallback assembles from the same evidence; the
        # stream itself surfaces the failure for the caller to catch.
        with self.assertRaises(RuntimeError):
            list(stream)


class TestTelemetry5D(unittest.TestCase):
    def test_meta_shape_metadata_only(self):
        from query_planner import query_planned
        _a, _s, _i, meta = query_planned(
            "What is the lock-in period in the lease deed?",
            config=_cfg(), vector_store=lease_store(),
            llm_provider=FakeLLM("A"), bundle=None)
        blob = str(meta).lower()
        self.assertNotIn("36 months", blob)
        self.assertNotIn("lock-in period in the lease", blob)
        for key in ("reasoning_used", "reasoning_provider",
                    "reasoning_model", "planner_latency_ms",
                    "planner_failure_category", "escalation_reason",
                    "workflow"):
            self.assertIn(key, meta, key)

    def test_event_merge(self):
        from pilot_telemetry import PilotTelemetryStore, build_telemetry_event
        ev = build_telemetry_event(message_id=9, answer_label="Grounded",
                                   source_count=1, total_ms=100,
                                   timings={},
                                   reasoning={"reasoning_used": True,
                                              "reasoning_provider": "fake",
                                              "reasoning_model": "fake-1",
                                              "planner_latency_ms": 4,
                                              "planner_failure_category": None,
                                              "escalation_reason": "ordinary",
                                              "workflow": "normal"})
        self.assertTrue(ev["reasoning_used"])
        self.assertEqual(ev["escalation_reason"], "ordinary")
        self.assertNotIn("question", ev)
        store = PilotTelemetryStore()
        stored = store.record(ev)
        self.assertEqual(stored["planner_latency_ms"], 4)

    def test_event_shape_unchanged_without_reasoning(self):
        from pilot_telemetry import build_telemetry_event
        ev = build_telemetry_event(message_id=1, total_ms=10, timings={})
        self.assertNotIn("reasoning_used", ev)
        self.assertNotIn("escalation_reason", ev)


class TestConfig5D(unittest.TestCase):
    def test_defaults_disabled(self):
        cfg = _cfg()
        self.assertEqual(cfg.reasoning_enabled, "0")
        self.assertEqual(cfg.reasoning_provider, "")
        self.assertEqual(cfg.reasoning_model_id, "")
        self.assertEqual(cfg.reasoning_region, "us-east-1")
        self.assertEqual(cfg.reasoning_timeout_s, 5)

    def test_enabled_parsing(self):
        from query_planner import reasoning_enabled
        for on in ["1", "true", "yes", "on", "TRUE"]:
            self.assertTrue(reasoning_enabled(_cfg(reasoning_enabled=on)))
        for off in ["0", "", "false", "no"]:
            self.assertFalse(reasoning_enabled(_cfg(reasoning_enabled=off)))

    def test_bundle_build_rules(self):
        from query_planner import get_planner_bundle
        self.assertIsNone(get_planner_bundle(_cfg()))
        with self.assertRaises(ValueError):
            get_planner_bundle(_cfg(reasoning_enabled="1",
                                    reasoning_provider="openai",
                                    reasoning_model_id="m"))
        with self.assertRaises(ValueError):
            get_planner_bundle(_cfg(reasoning_enabled="1",
                                    reasoning_provider="bedrock",
                                    reasoning_model_id=""))


class TestBenchmark5D(unittest.TestCase):
    def test_deterministic_parity(self):
        from tests.benchmarks.harness import run_all
        base = run_all()
        planned = run_all(planner="deterministic")
        bstatus = {r["id"]: r["status"] for r in base["results"]}
        pstatus = {r["id"]: r["status"] for r in planned["results"]}
        self.assertEqual(bstatus, pstatus)

        def substance(result):
            m = result["metrics"]
            return (result["status"],
                    m["answer_chars"], m["source_count"],
                    tuple(m["source_files"]), m["support"],
                    m["guard_flagged"], m["guard_total"],
                    m["tokens_in"], m["tokens_out"])

        for rid in bstatus:
            with self.subTest(rid=rid):
                b = next(r for r in base["results"] if r["id"] == rid)
                p = next(r for r in planned["results"] if r["id"] == rid)
                self.assertEqual(substance(p), substance(b))

    def test_planner_metrics_present(self):
        from tests.benchmarks.harness import run_all
        out = run_all(planner="deterministic")
        for r in out["results"]:
            if r["status"] == "pass":
                m = r["metrics"]
                for key in ("planner_invoked", "planner_valid",
                            "escalation_reason", "planner_latency_ms"):
                    self.assertIn(key, m, (r["id"], key))

    def test_fake_llm_planner_end_to_end(self):
        from tests.benchmarks.harness import run_all
        planner = FakePlanner([(valid_plan(
            workflow="normal",
            retrieval_queries=["lock-in period lease deed"]), canned_meta())
            for _ in range(30)])
        bundle = bundle_for(planner)
        out = run_all(planner=bundle)
        by_id = {r["id"]: r for r in out["results"]}
        # Ordinary questions never consult the planner (conservative gate).
        self.assertEqual(by_id["normal-lockin"]["status"], "pass")
        self.assertFalse(
            by_id["normal-lockin"]["metrics"]["planner_invoked"])
        # Ambiguous comparison escalates and stays passing.
        self.assertEqual(
            by_id["ambiguous-compare unnamed"]["status"], "pass")
        self.assertTrue(
            by_id["ambiguous-compare unnamed"]["metrics"]["planner_invoked"])


if __name__ == "__main__":
    unittest.main()
