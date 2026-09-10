"""Phase 5E tests: reviewer, schema, triggers, repair, failures.

Mocks/fakes only: no live AWS calls, no model downloads. The reviewer
path is exercised through injected fakes; REVIEW_ENABLED=0 parity is
asserted exactly.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SID_A = "a" * 32
SID_B = "b" * 32
FILES = ["lease.pdf", "contract.pdf"]


def valid_verdict(**over):
    verdict = {"schema_version": 1, "verdict": "pass",
               "missing_aspects": [], "unsupported_claims": [],
               "suggested_followup_queries": []}
    verdict.update(over)
    return verdict


def repair_verdict(queries=None, **over):
    base = {"schema_version": 1, "verdict": "repair",
            "missing_aspects": ["notice period"],
            "unsupported_claims": ["zebra claim"],
            "suggested_followup_queries": list(queries if queries is not None
                                               else ["notice period lease"])}
    base.update(over)
    return base


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
                          "cid": cid or "c%d" % len(self.rows), "page": page})

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
        self.calls = 0

    def generate(self, context, question, prompt_template):
        self.calls += 1
        return self.text

    def generate_stream(self, context, question, prompt_template):
        self.calls += 1
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
    return config.AppConfig(**over)


class FakeReviewer:
    """Scripted reviewer double: returns queued (raw, meta) pairs."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def review(self, question, answer, evidence, *, workflow=None,
               available_files=None, session_id=None, plan_meta=None):
        self.calls.append({"question": question, "answer": answer,
                           "evidence": evidence, "workflow": workflow,
                           "files": available_files, "session": session_id,
                           "plan": plan_meta})
        if not self.script:
            raise AssertionError("reviewer called more often than scripted")
        return self.script.pop(0)


def canned_review_meta(**over):
    meta = {"provider": "fake", "model": "fake-review-1",
            "latency_ms": 4, "failure": None,
            "tokens_in": None, "tokens_out": None}
    meta.update(over)
    return meta


def review_bundle_for(reviewer):
    return {"reviewer": reviewer, "provider": "fake",
            "model": "fake-review-1", "timeout_s": 5,
            "_meta_base": {"review_used": True,
                           "review_provider": "fake",
                           "review_model": "fake-review-1"}}


def stable_result(answer, sources, info):
    kept = {k: info.get(k) for k in ("answer", "retrieved",
                                     "needs_retrieval", "support_level",
                                     "failed", "workflow", "label",
                                     "fallback_template",
                                     "fallback_question")}
    kept["timings_keys"] = sorted((info.get("timings") or {}).keys())
    return answer, sources, kept


class TestDisabledParity(unittest.TestCase):
    def test_bundle_none_by_default(self):
        from answer_reviewer import get_reviewer_bundle
        import config
        self.assertIsNone(get_reviewer_bundle(config.load_config()))
        self.assertIsNone(get_reviewer_bundle(None))
        self.assertIsNone(get_reviewer_bundle(_cfg(review_enabled="0")))

    def test_no_langchain_import_when_disabled(self):
        import sys
        from answer_reviewer import get_reviewer_bundle
        import config
        get_reviewer_bundle(config.load_config())
        self.assertNotIn("langchain_aws", sys.modules)

    def test_query_parity(self):
        from answer_reviewer import query_reviewed
        from workflows import query_workflow
        cfg = _cfg()
        q = "What is the lock-in period in the lease deed?"
        expected = query_workflow(q, config=cfg, vector_store=lease_store(),
                                  llm_provider=FakeLLM("A"))
        got = query_reviewed(q, config=cfg, vector_store=lease_store(),
                             llm_provider=FakeLLM("A"),
                             planner_bundle=None,
                             reviewer_bundle=None)[:3]
        self.assertEqual(got, expected)

    def test_stream_parity(self):
        from answer_reviewer import stream_reviewed_answer
        from query_planner import stream_planned_answer
        cfg = _cfg()
        q = "What is the lock-in period in the lease deed?"
        info1, stream1, _r1 = stream_planned_answer(
            q, config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("A"), bundle=None)
        info2, stream2, _r2, review = stream_reviewed_answer(
            q, config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("A"), planner_bundle=None,
            reviewer_bundle=None)
        self.assertEqual(list(stream1), list(stream2))
        self.assertEqual(info1, info2)
        self.assertFalse(review["review_used"])
        self.assertEqual(review["review_failure_category"], "disabled")

    def test_deterministic_bundle_parity(self):
        from answer_reviewer import (DeterministicReviewer, query_reviewed)
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
            bundle = review_bundle_for(DeterministicReviewer())
            got = query_reviewed(q, config=cfg, vector_store=store,
                                 llm_provider=FakeLLM("A"),
                                 planner_bundle=None,
                                 reviewer_bundle=bundle)
            with self.subTest(q=q):
                # Deterministic pass-only reviewer never repairs, so the
                # comparison/summary triggers still call it once but the
                # answer is unchanged.
                self.assertEqual(stable_result(*got[:3]),
                                 stable_result(*expected))


class TestNoAnswerGeneration(unittest.TestCase):
    def test_reviewer_has_no_generation_methods(self):
        from answer_reviewer import DeterministicReviewer, LLMAnswerReviewer
        for cls in (DeterministicReviewer, LLMAnswerReviewer):
            self.assertFalse(hasattr(cls, "generate"))
            self.assertFalse(hasattr(cls, "generate_stream"))
            self.assertFalse(hasattr(cls, "generate_answer"))

    def test_structured_path_never_answers(self):
        from answer_reviewer import (LLMAnswerReviewer,
                                     bedrock_reviewer_structured_fn)
        calls = []

        class FakeChat:
            def with_structured_output(self, schema):
                calls.append(("structured", schema))
                return self

            def invoke(self, prompt_text):
                calls.append(("invoke", prompt_text))
                return valid_verdict()

        fn = bedrock_reviewer_structured_fn(FakeChat())
        reviewer = LLMAnswerReviewer(fn, timeout_s=5)
        raw, meta = reviewer.review("Q?", "A.", [], workflow="normal",
                                    available_files=FILES)
        self.assertEqual(raw["schema_version"], 1)
        kinds = [k for k, _v in calls]
        self.assertIn("structured", kinds)
        self.assertIn("invoke", kinds)
        prompt = calls[1][1]
        self.assertIn("JSON", prompt)
        self.assertIn("Never write", prompt)


class TestSchemaValidation(unittest.TestCase):
    def test_exact_schema_pass(self):
        from answer_reviewer import validate_verdict
        plan, failure = validate_verdict(valid_verdict())
        self.assertIsNone(failure)
        self.assertEqual(plan["verdict"], "pass")

    def test_version(self):
        from answer_reviewer import validate_verdict
        for bad in [dict(valid_verdict(), schema_version=2),
                    dict(valid_verdict(), schema_version="1"),
                    {k: v for k, v in valid_verdict().items()
                     if k != "schema_version"}]:
            _p, failure = validate_verdict(bad)
            self.assertEqual(failure, "schema")

    def test_unknown_and_missing_keys(self):
        from answer_reviewer import validate_verdict
        bad = valid_verdict()
        bad["surprise"] = 1
        _p, failure = validate_verdict(bad)
        self.assertEqual(failure, "schema")
        _p, failure = validate_verdict("not-a-dict")
        self.assertEqual(failure, "malformed")
        _p, failure = validate_verdict(
            {k: v for k, v in valid_verdict().items()
             if k != "verdict"})
        self.assertEqual(failure, "schema")

    def test_no_answer_field(self):
        from answer_reviewer import validate_verdict
        for key in ("answer", "repaired_answer", "final_answer",
                    "generated_answer", "text", "response"):
            bad = valid_verdict()
            bad[key] = "new answer text"
            _p, failure = validate_verdict(bad)
            self.assertEqual(failure, "schema", key)

    def test_no_path_fields(self):
        from answer_reviewer import validate_verdict
        for key in ("file_path", "source_path", "path", "secret",
                    "api_key", "password", "token"):
            bad = valid_verdict()
            bad[key] = "x"
            _p, failure = validate_verdict(bad)
            self.assertEqual(failure, "schema", key)

    def test_verdict_enum_closed(self):
        from answer_reviewer import validate_verdict
        for bad_verdict in ["approve", "retry", "", None, 42, "PASS"]:
            _p, failure = validate_verdict(
                valid_verdict(verdict=bad_verdict))
            self.assertEqual(failure, "schema", repr(bad_verdict))
        for good in ["pass", "repair", "insufficient"]:
            _p, failure = validate_verdict(valid_verdict(verdict=good))
            self.assertIsNone(failure, good)

    def test_pass_consistency(self):
        from answer_reviewer import validate_verdict
        _p, failure = validate_verdict(valid_verdict(
            verdict="pass", missing_aspects=["x"]))
        self.assertEqual(failure, "schema")
        _p, failure = validate_verdict(valid_verdict(
            verdict="pass",
            suggested_followup_queries=["notice period lease"]))
        self.assertEqual(failure, "schema")

    def test_bounds(self):
        from answer_reviewer import validate_verdict
        _p, failure = validate_verdict(valid_verdict(
            verdict="repair", missing_aspects=["a"] * 5,
            suggested_followup_queries=["q"]))
        self.assertEqual(failure, "schema")
        _p, failure = validate_verdict(valid_verdict(
            verdict="repair", unsupported_claims=["a"] * 5,
            suggested_followup_queries=["q"]))
        self.assertEqual(failure, "schema")
        _p, failure = validate_verdict(valid_verdict(
            verdict="repair", missing_aspects=["a"],
            suggested_followup_queries=["q1", "q2", "q3", "q4"]))
        self.assertEqual(failure, "schema")
        _p, failure = validate_verdict(valid_verdict(
            verdict="repair", missing_aspects=["x" * 201],
            suggested_followup_queries=["qqq"]))
        self.assertEqual(failure, "schema")
        _p, failure = validate_verdict(valid_verdict(
            verdict="repair", unsupported_claims=["x" * 301],
            suggested_followup_queries=["qqq"]))
        self.assertEqual(failure, "schema")
        _p, failure = validate_verdict(valid_verdict(
            verdict="repair", missing_aspects=["ok"],
            suggested_followup_queries=["x" * 201]))
        self.assertEqual(failure, "schema")

    def test_path_values_rejected(self):
        from answer_reviewer import validate_verdict
        _p, failure = validate_verdict(repair_verdict(
            queries=["/etc/passwd lease"]))
        self.assertEqual(failure, "schema")
        _p, failure = validate_verdict(valid_verdict(
            verdict="repair", missing_aspects=["a/b"],
            suggested_followup_queries=["notice lease"]))
        self.assertEqual(failure, "schema")

    def test_gap_semantics_style_not_gaps(self):
        # Style/wording/verbosity are documented as never-gaps; the
        # schema itself carries no style dimension, and the reviewer
        # prompt names the exclusion explicitly.
        from answer_reviewer import build_reviewer_input
        prompt = build_reviewer_input("Q?", "A.", [], workflow="normal",
                                      available_files=FILES)
        self.assertIn("Style", prompt)
        self.assertIn("never gaps", prompt)


class TestFollowupValidation(unittest.TestCase):
    def test_accepts_clean(self):
        from answer_reviewer import validate_followup_queries
        valid, rejected = validate_followup_queries(
            ["notice period lease", "lock-in terms"], available_files=FILES)
        self.assertEqual(valid, ["notice period lease", "lock-in terms"])
        self.assertEqual(rejected, 0)

    def test_rejects_unsafe(self):
        from answer_reviewer import validate_followup_queries
        valid, rejected = validate_followup_queries(
            ["", "x", "y" * 201, "/etc/passwd", "a\\b",
             "ghost.pdf terms", 42, None], available_files=FILES)
        self.assertEqual(valid, [])
        self.assertEqual(rejected, 8)

    def test_known_pdf_allowed_unknown_rejected(self):
        from answer_reviewer import validate_followup_queries
        valid, _r = validate_followup_queries(
            ["notice terms in lease.pdf"], available_files=FILES)
        self.assertEqual(valid, ["notice terms in lease.pdf"])
        valid, rejected = validate_followup_queries(
            ["terms in ghost.pdf"], available_files=FILES)
        self.assertEqual(valid, [])
        self.assertEqual(rejected, 1)

    def test_dedupes(self):
        from answer_reviewer import validate_followup_queries
        valid, _r = validate_followup_queries(
            ["same query terms", "same query terms"], available_files=FILES)
        self.assertEqual(valid, ["same query terms"])


class TestBoundedInput(unittest.TestCase):
    def test_bounds_and_hygiene(self):
        from answer_reviewer import build_reviewer_input
        evidence = [{"file_name": "lease.pdf", "page": 0,
                     "content": "E" * 5000,
                     "source_path": "/secret/home/lease.pdf",
                     "source": "/secret/home/lease.pdf"}]
        prompt = build_reviewer_input(
            "Q" * 2000, "A" * 5000, evidence * 10,
            workflow="comparison", available_files=FILES * 10,
            plan_meta={"workflow": "normal",
                       "escalation_reason": "compound",
                       "question": "SECRET-QUESTION"},
            session_bound=True)
        self.assertIn("JSON", prompt)
        self.assertNotIn(SID_A, prompt)
        self.assertNotIn("/secret/home", prompt)
        self.assertNotIn("/home/", prompt)
        self.assertNotIn("SECRET-QUESTION", prompt)
        self.assertNotIn("E" * 1300, prompt)
        self.assertNotIn("Q" * 600, prompt)
        self.assertNotIn("A" * 2100, prompt)
        self.assertLessEqual(len(prompt), 15000)

    def test_evidence_only_context(self):
        from answer_reviewer import LLMAnswerReviewer
        seen = {}

        def structured(prompt_text, schema):
            seen["prompt"] = prompt_text
            return valid_verdict()

        reviewer = LLMAnswerReviewer(structured, timeout_s=5)
        reviewer.review("What is the lock-in?",
                        "The lock-in is 36 months.",
                        [{"file_name": "lease.pdf", "page": 0,
                          "content": "lock-in 36 months",
                          "source_path": "/secret/x.pdf"}],
                        workflow="normal", available_files=FILES,
                        session_id=SID_A,
                        plan_meta={"workflow": "normal"})
        prompt = seen["prompt"]
        self.assertNotIn(SID_A, prompt)
        self.assertNotIn("/secret/x.pdf", prompt)
        self.assertIn("lease.pdf", prompt)
        self.assertIn("lock-in 36 months", prompt)

    def test_session_id_rejected(self):
        from answer_reviewer import DeterministicReviewer
        reviewer = DeterministicReviewer()
        _v, meta = reviewer.review("Q?", "A.", [], session_id="bad-id")
        self.assertEqual(meta["failure"], "schema")


class TestInvocationPolicy(unittest.TestCase):
    def test_ordinary_non_invocation(self):
        from answer_reviewer import query_reviewed
        reviewer = FakeReviewer([(valid_verdict(),
                                  canned_review_meta())])
        cfg = _cfg()
        q = "What is the lock-in period in the lease deed?"
        _a, _s, _i, _r, review = query_reviewed(
            q, config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("The lock-in period is 36 months."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(reviewer.calls, [])
        self.assertFalse(review["review_used"])
        self.assertEqual(review["review_trigger"], "none")
        self.assertFalse(review["repair_attempted"])

    def test_citation_guard_trigger(self):
        from answer_reviewer import query_reviewed
        reviewer = FakeReviewer([(valid_verdict(),
                                  canned_review_meta())])
        cfg = _cfg()
        q = "What is the lock-in period in the lease deed?"
        _a, _s, _i, _r, review = query_reviewed(
            q, config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(len(reviewer.calls), 1)
        self.assertTrue(review["review_used"])
        self.assertEqual(review["review_trigger"], "citation_guard")
        self.assertEqual(review["review_verdict"], "pass")

    def test_comparison_trigger(self):
        from answer_reviewer import query_reviewed
        store = lease_store()
        store.add("The service contract renews annually with 30 days notice.",
                  "contract.pdf")
        reviewer = FakeReviewer([(valid_verdict(),
                                  canned_review_meta())])
        cfg = _cfg()
        q = "Compare the lock-in and notice terms in contract.pdf and lease.pdf."
        _a, _s, _i, _r, review = query_reviewed(
            q, config=cfg, vector_store=store,
            llm_provider=FakeLLM("Both mention notice periods."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(len(reviewer.calls), 1)
        self.assertEqual(review["review_trigger"], "comparison")

    def test_summary_trigger(self):
        from answer_reviewer import query_reviewed
        reviewer = FakeReviewer([(valid_verdict(),
                                  canned_review_meta())])
        cfg = _cfg()
        q = "Summarize lease.pdf."
        _a, _s, _i, _r, review = query_reviewed(
            q, config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("The lease deed states the lock-in period."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(len(reviewer.calls), 1)
        self.assertEqual(review["review_trigger"], "summary")

    def test_always_flag(self):
        from answer_reviewer import query_reviewed
        reviewer = FakeReviewer([(valid_verdict(),
                                  canned_review_meta())])
        cfg = _cfg(review_always="1")
        q = "What is the lock-in period in the lease deed?"
        _a, _s, _i, _r, review = query_reviewed(
            q, config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("The lock-in period is 36 months."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(len(reviewer.calls), 1)
        self.assertEqual(review["review_trigger"], "always")

    def test_system_answers_never_reviewed(self):
        from answer_reviewer import query_reviewed
        reviewer = FakeReviewer([(valid_verdict(),
                                  canned_review_meta())])
        cfg = _cfg(review_always="1")
        # Unsupported question -> contextual refusal, no sources.
        _a, _s, _i, _r, review = query_reviewed(
            "What is the capital of France?",
            config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("UNREACHABLE"),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(reviewer.calls, [])
        self.assertFalse(review["review_used"])


class TestVerdictPaths(unittest.TestCase):
    def test_pass_preserves(self):
        from answer_reviewer import query_reviewed
        from workflows import query_workflow
        reviewer = FakeReviewer([(valid_verdict(),
                                  canned_review_meta())])
        cfg = _cfg()
        store = lease_store()
        # Force the trigger via an ungrounded answer, then pass.
        expected_answer, expected_sources, _info = query_workflow(
            "What is the lock-in period in the lease deed?",
            config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."))
        got = query_reviewed(
            "What is the lock-in period in the lease deed?",
            config=cfg, vector_store=store,
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(got[0], expected_answer)
        self.assertEqual(got[1], expected_sources)
        self.assertEqual(got[4]["review_verdict"], "pass")
        self.assertFalse(got[4]["repair_attempted"])

    def test_repair_path(self):
        from answer_reviewer import query_reviewed
        store = FakeStore()
        store.add("The lock-in period in the lease deed is 36 months.",
                  "lease.pdf")
        store.add("The lease deed requires 3 months notice for renewal.",
                  "lease.pdf", cid="c2")
        reviewer = FakeReviewer([(repair_verdict(["notice renewal lease"]),
                                  canned_review_meta())])
        cfg = _cfg()

        class RepairLLM(FakeLLM):
            def __init__(self):
                super().__init__("Zebra savanna chartreuse quantum.")
                self.contexts = []

            def generate(self, context, question, prompt_template):
                self.contexts.append(context)
                self.calls += 1
                if self.calls == 1:
                    return "Zebra savanna chartreuse quantum."
                return "The lease needs 3 months notice for renewal."

        llm = RepairLLM()
        answer, sources, _info, _r, review = query_reviewed(
            "What is the lock-in period in the lease deed?",
            config=cfg, vector_store=store, llm_provider=llm,
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertTrue(review["review_used"])
        self.assertEqual(review["review_verdict"], "repair")
        self.assertTrue(review["repair_attempted"])
        self.assertTrue(review["repair_succeeded"])
        self.assertIn("3 months notice", answer)
        self.assertEqual(len(reviewer.calls), 1)  # exactly one cycle
        self.assertTrue(any("notice" in f for f in store.forms[2:]))

    def test_insufficient_preserves_no_search(self):
        from answer_reviewer import query_reviewed
        store = lease_store()
        reviewer = FakeReviewer([(valid_verdict(
            verdict="insufficient", missing_aspects=["renewal term"],
            unsupported_claims=["zebra claim"]), canned_review_meta())])
        cfg = _cfg()
        before_forms = None
        answer, sources, _info, _r, review = query_reviewed(
            "What is the lock-in period in the lease deed?",
            config=cfg, vector_store=store,
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        _ = before_forms
        self.assertEqual(review["review_verdict"], "insufficient")
        self.assertFalse(review["repair_attempted"])
        self.assertFalse(review["repair_succeeded"])
        self.assertIn("Zebra", answer)
        # No gap-fill: only the two deterministic retrieval forms ran.
        self.assertEqual(len(store.forms), 2)

    def test_repair_without_valid_queries_no_search(self):
        from answer_reviewer import query_reviewed
        store = lease_store()
        reviewer = FakeReviewer([(valid_verdict(
            verdict="repair", missing_aspects=["x"],
            unsupported_claims=["y"],
            suggested_followup_queries=[]), canned_review_meta())])
        cfg = _cfg()
        _a, _s, _i, _r, review = query_reviewed(
            "What is the lock-in period in the lease deed?",
            config=cfg, vector_store=store,
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertFalse(review["repair_attempted"])
        self.assertEqual(len(store.forms), 2)

    def test_guard_rerun_recorded(self):
        from answer_reviewer import query_reviewed
        store = FakeStore()
        store.add("The lock-in period in the lease deed is 36 months.",
                  "lease.pdf")
        store.add("The lease deed requires 3 months notice for renewal.",
                  "lease.pdf", cid="c2")
        reviewer = FakeReviewer([(repair_verdict(["notice renewal lease"]),
                                  canned_review_meta())])
        cfg = _cfg()

        class RepairLLM(FakeLLM):
            def __init__(self):
                super().__init__("")
                self.calls = 0

            def generate(self, context, question, prompt_template):
                self.calls += 1
                if self.calls == 1:
                    return "Zebra savanna chartreuse quantum."
                return "The lease needs 3 months notice for renewal."

        _a, _s, _i, _r, review = query_reviewed(
            "What is the lock-in period in the lease deed?",
            config=cfg, vector_store=store, llm_provider=RepairLLM(),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertIsNotNone(review["guard_before_flagged"])
        self.assertIsNotNone(review["guard_after_flagged"])
        self.assertGreater(review["guard_before_flagged"], 0)


class TestOneCycleAndMerge(unittest.TestCase):
    def test_exactly_one_cycle(self):
        from answer_reviewer import query_reviewed
        store = FakeStore()
        store.add("The lock-in period in the lease deed is 36 months.",
                  "lease.pdf")
        store.add("Notice terms for the lease deed are included here.",
                  "notice.pdf", cid="c9")
        reviewer = FakeReviewer([(repair_verdict(["notice lease deed"]),
                                  canned_review_meta())])
        cfg = _cfg()

        class TwoLLM(FakeLLM):
            def __init__(self):
                super().__init__("")
                self.calls = 0

            def generate(self, context, question, prompt_template):
                self.calls += 1
                if self.calls == 1:
                    return "Zebra savanna chartreuse quantum."
                return "Repaired answer about notice lease deed."

        llm = TwoLLM()
        query_reviewed(
            "What is the lock-in period in the lease deed?",
            config=cfg, vector_store=store, llm_provider=llm,
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(len(reviewer.calls), 1)
        self.assertEqual(llm.calls, 2)  # initial + exactly one repair

    def test_merge_deterministic(self):
        from answer_reviewer import merge_evidence
        a = [{"chunk_id": "c1", "score": 0.5, "content": "a"},
             {"chunk_id": "c2", "score": 0.9, "content": "b"}]
        b = [[{"chunk_id": "c2", "score": 0.4, "content": "b2"},
              {"chunk_id": "c3", "score": 0.8, "content": "c"}]]
        merged = merge_evidence(a, b)
        self.assertEqual([s["chunk_id"] for s in merged],
                         ["c2", "c3", "c1"])
        # Max score wins for duplicates.
        self.assertEqual(merged[0]["content"], "b")
        # Stable rerun.
        self.assertEqual(merge_evidence(a, b), merged)

    def test_gap_fill_uses_same_mechanism(self):
        from answer_reviewer import query_reviewed
        store = FakeStore()
        store.add("The lock-in period in the lease deed is 36 months.",
                  "lease.pdf")
        store.add("Zebra stripe pattern savanna study.", "zebra.pdf",
                  cid="cz")
        reviewer = FakeReviewer([(repair_verdict(["zebra stripe pattern"]),
                                  canned_review_meta())])
        cfg = _cfg()

        class GapLLM(FakeLLM):
            def __init__(self):
                super().__init__("")
                self.calls = 0

            def generate(self, context, question, prompt_template):
                self.calls += 1
                if self.calls == 1:
                    return "Zebra savanna chartreuse quantum."
                return "Evidence says zebra."

        _a, sources, _i, _r, review = query_reviewed(
            "What is the lock-in period in the lease deed?",
            config=cfg, vector_store=store, llm_provider=GapLLM(),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertTrue(review["repair_succeeded"])
        self.assertIn("zebra.pdf", {s["file_name"] for s in sources})


class TestFailures(unittest.TestCase):
    def test_timeout_preserves(self):
        from answer_reviewer import LLMAnswerReviewer, query_reviewed
        import time

        def slow(prompt_text, schema):
            time.sleep(5)
            return valid_verdict()

        reviewer = LLMAnswerReviewer(slow, timeout_s=0.05)
        from workflows import query_workflow
        cfg = _cfg()
        expected = query_workflow(
            "What is the lock-in period in the lease deed?", config=cfg,
            vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."))
        got = query_reviewed(
            "What is the lock-in period in the lease deed?", config=cfg,
            vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."),
            planner_bundle=None,
            reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(got[0], expected[0])
        self.assertEqual(got[4]["review_failure_category"], "timeout")
        self.assertFalse(got[4]["repair_attempted"])

    def test_credentials_failure(self):
        class NoCredentialsError(Exception):
            pass

        def failing(prompt_text, schema):
            raise NoCredentialsError("nope")

        from answer_reviewer import LLMAnswerReviewer
        reviewer = LLMAnswerReviewer(failing, timeout_s=5)
        from answer_reviewer import query_reviewed
        from workflows import query_workflow
        cfg = _cfg()
        expected = query_workflow(
            "What is the lock-in period in the lease deed?", config=cfg,
            vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."))
        out = query_reviewed(
            "What is the lock-in period in the lease deed?", config=cfg,
            vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(out[0], expected[0])
        self.assertEqual(out[4]["review_failure_category"], "credentials")

    def test_unavailable_failure(self):
        def failing(prompt_text, schema):
            raise ConnectionError("endpoint down")

        from answer_reviewer import LLMAnswerReviewer, query_reviewed
        from workflows import query_workflow
        reviewer = LLMAnswerReviewer(failing, timeout_s=5)
        cfg = _cfg()
        expected = query_workflow(
            "What is the lock-in period in the lease deed?", config=cfg,
            vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."))
        out = query_reviewed(
            "What is the lock-in period in the lease deed?", config=cfg,
            vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(out[0], expected[0])
        self.assertEqual(out[4]["review_failure_category"], "unavailable")

    def test_malformed_output(self):
        def prose(prompt_text, schema):
            return "The lock-in is 36 months."

        from answer_reviewer import LLMAnswerReviewer, query_reviewed
        from workflows import query_workflow
        reviewer = LLMAnswerReviewer(prose, timeout_s=5)
        cfg = _cfg()
        expected = query_workflow(
            "What is the lock-in period in the lease deed?", config=cfg,
            vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."))
        out = query_reviewed(
            "What is the lock-in period in the lease deed?", config=cfg,
            vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(out[0], expected[0])
        self.assertEqual(out[4]["review_failure_category"], "malformed")

    def test_schema_failure(self):
        def bad(prompt_text, schema):
            return {"verdict": "approve"}

        from answer_reviewer import LLMAnswerReviewer, query_reviewed
        from workflows import query_workflow
        reviewer = LLMAnswerReviewer(bad, timeout_s=5)
        cfg = _cfg()
        expected = query_workflow(
            "What is the lock-in period in the lease deed?", config=cfg,
            vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."))
        out = query_reviewed(
            "What is the lock-in period in the lease deed?", config=cfg,
            vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(out[0], expected[0])
        self.assertEqual(out[4]["review_failure_category"], "schema")

    def test_model_error(self):
        def bad_model(prompt_text, schema):
            raise ValueError("unknown model id xyz")

        from answer_reviewer import LLMAnswerReviewer, query_reviewed
        from workflows import query_workflow
        reviewer = LLMAnswerReviewer(bad_model, timeout_s=5)
        cfg = _cfg()
        expected = query_workflow(
            "What is the lock-in period in the lease deed?", config=cfg,
            vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."))
        out = query_reviewed(
            "What is the lock-in period in the lease deed?", config=cfg,
            vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(out[0], expected[0])
        self.assertEqual(out[4]["review_failure_category"], "model_error")

    def test_unsupported_dependency(self):
        def missing(prompt_text, schema):
            raise RuntimeError("langchain-aws is not installed")

        from answer_reviewer import LLMAnswerReviewer, query_reviewed
        from workflows import query_workflow
        reviewer = LLMAnswerReviewer(missing, timeout_s=5)
        cfg = _cfg()
        expected = query_workflow(
            "What is the lock-in period in the lease deed?", config=cfg,
            vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."))
        out = query_reviewed(
            "What is the lock-in period in the lease deed?", config=cfg,
            vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        self.assertEqual(out[0], expected[0])
        self.assertEqual(out[4]["review_failure_category"], "unsupported")

    def test_misconfigured_builder_loud(self):
        from answer_reviewer import build_llm_reviewer
        with self.assertRaises(ValueError):
            build_llm_reviewer(_cfg(review_provider="openai",
                                    review_model_id="m"))
        with self.assertRaises(ValueError):
            build_llm_reviewer(_cfg(review_provider="bedrock",
                                    review_model_id=""))
        with self.assertRaises(ValueError):
            build_llm_reviewer(_cfg(review_provider="bedrock",
                                    review_model_id="m",
                                    review_timeout_s="soon"))


class TestStreaming5E(unittest.TestCase):
    def test_stream_disabled_parity(self):
        from answer_reviewer import stream_reviewed_answer
        from query_planner import stream_planned_answer
        cfg = _cfg()
        q = "What is the lock-in period in the lease deed?"
        info1, stream1, _r1 = stream_planned_answer(
            q, config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("A"), bundle=None)
        info2, stream2, _r2, review = stream_reviewed_answer(
            q, config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("A"), planner_bundle=None,
            reviewer_bundle=None)
        self.assertEqual(list(stream1), list(stream2))
        self.assertEqual(info1, info2)
        self.assertFalse(review["review_used"])

    def test_stream_sanitized_after_repair(self):
        from answer_reviewer import stream_reviewed_answer
        store = FakeStore()
        store.add("The lock-in period in the lease deed is 36 months.",
                  "lease.pdf")
        store.add("The lease deed requires 3 months notice for renewal.",
                  "lease.pdf", cid="c2")
        reviewer = FakeReviewer([(repair_verdict(["notice renewal lease"]),
                                  canned_review_meta())])
        cfg = _cfg()

        class ThinkLLM(FakeLLM):
            def __init__(self):
                super().__init__("")
                self.calls = 0

            def generate(self, context, question, prompt_template):
                self.calls += 1
                if self.calls == 1:
                    return "Zebra savanna chartreuse quantum."
                return "REAL-ANSWER lease notice."

            def generate_stream(self, context, question, prompt_template):
                if self.calls == 0:
                    self.calls += 1
                    yield "<think>secret reasoning</think>Zebra savanna chartreuse quantum."
                else:
                    yield "REAL-ANSWER lease notice."

        _info, stream, _r, review = stream_reviewed_answer(
            "What is the lock-in period in the lease deed?",
            config=cfg, vector_store=store, llm_provider=ThinkLLM(),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        text = "".join(stream)
        self.assertNotIn("<think>", text)
        self.assertTrue(review["repair_succeeded"])

    def test_stream_failure_propagates_without_review(self):
        from answer_reviewer import stream_reviewed_answer

        class FailLLM(FakeLLM):
            def generate_stream(self, context, question, prompt_template):
                raise RuntimeError("stream down")

            def generate(self, context, question, prompt_template):
                return "FALLBACK lease."

        reviewer = FakeReviewer([(valid_verdict(),
                                  canned_review_meta())])
        _info, stream, _r, review = stream_reviewed_answer(
            "What is the lock-in period in the lease deed?",
            config=_cfg(), vector_store=lease_store(),
            llm_provider=FailLLM(), planner_bundle=None,
            reviewer_bundle=review_bundle_for(reviewer))
        with self.assertRaises(RuntimeError):
            list(stream)
        self.assertFalse(review["review_used"])


class TestTelemetry5E(unittest.TestCase):
    def test_meta_shape_metadata_only(self):
        from answer_reviewer import query_reviewed
        store = lease_store()
        store.add("The lease deed requires 3 months notice for renewal.",
                  "lease.pdf", cid="c2")
        reviewer = FakeReviewer([(repair_verdict(["notice renewal lease"]),
                                  canned_review_meta())])
        cfg = _cfg()

        class RepairLLM(FakeLLM):
            def __init__(self):
                super().__init__("")
                self.calls = 0

            def generate(self, context, question, prompt_template):
                self.calls += 1
                if self.calls == 1:
                    return "Zebra savanna chartreuse quantum."
                return "The lease needs 3 months notice for renewal."

        _a, _s, _i, _r, review = query_reviewed(
            "What is the lock-in period in the lease deed?",
            config=cfg, vector_store=store, llm_provider=RepairLLM(),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        blob = str(review).lower()
        self.assertNotIn("36 months", blob)
        self.assertNotIn("zebra savanna", blob)
        for key in ("review_used", "review_provider", "review_model",
                    "review_latency_ms", "review_verdict",
                    "review_failure_category", "review_trigger",
                    "repair_attempted", "repair_succeeded"):
            self.assertIn(key, review, key)

    def test_event_merge(self):
        from pilot_telemetry import PilotTelemetryStore, build_telemetry_event
        ev = build_telemetry_event(
            message_id=9, answer_label="Grounded", source_count=1,
            total_ms=100, timings={},
            review={"review_used": True, "review_provider": "fake",
                    "review_model": "fake-review-1",
                    "review_latency_ms": 4, "review_verdict": "pass",
                    "review_failure_category": None,
                    "review_trigger": "comparison",
                    "repair_attempted": False, "repair_succeeded": False})
        self.assertTrue(ev["review_used"])
        self.assertEqual(ev["review_trigger"], "comparison")
        self.assertNotIn("question", ev)
        store = PilotTelemetryStore()
        stored = store.record(ev)
        self.assertEqual(stored["review_latency_ms"], 4)
        self.assertEqual(stored["review_verdict"], "pass")

    def test_event_shape_unchanged_without_review(self):
        from pilot_telemetry import build_telemetry_event
        ev = build_telemetry_event(message_id=1, total_ms=10, timings={})
        self.assertNotIn("review_used", ev)
        self.assertNotIn("review_trigger", ev)

    def test_no_raw_output_logged(self):
        from answer_reviewer import query_reviewed
        reviewer = FakeReviewer([(repair_verdict(["notice lease"]),
                                  canned_review_meta())])
        cfg = _cfg()
        _a, _s, _i, _r, review = query_reviewed(
            "What is the lock-in period in the lease deed?",
            config=cfg, vector_store=lease_store(),
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."),
            planner_bundle=None, reviewer_bundle=review_bundle_for(reviewer))
        blob = str(review)
        self.assertNotIn("Zebra savanna", blob)
        self.assertNotIn("36 months", blob)


class TestIsolation5E(unittest.TestCase):
    def test_zero_cross_session_leakage(self):
        from answer_reviewer import query_reviewed
        store = FakeStore()
        store.add("lease lock-in alpha terms here", "lease.pdf")
        store.add("session bravo secret confidential terms", "sess.pdf",
                  scope="session", sid=SID_A)
        store.add("The lease deed requires 3 months notice for renewal.",
                  "lease.pdf", cid="c3")
        reviewer = FakeReviewer([(repair_verdict(["session bravo secret"]),
                                  canned_review_meta())])
        cfg = _cfg()
        _a, sources, _i, _r, review = query_reviewed(
            "lease lock-in alpha",
            config=cfg, vector_store=store,
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."),
            session_id=SID_B, planner_bundle=None,
            reviewer_bundle=review_bundle_for(reviewer))
        self.assertFalse(any(s["file_name"] == "sess.pdf" for s in sources))
        self.assertTrue(review["repair_attempted"])

    def test_suggested_queries_do_not_alter_scope(self):
        from answer_reviewer import query_reviewed
        store = FakeStore()
        store.add("lease lock-in alpha terms here", "lease.pdf")
        store.add("session bravo secret confidential terms", "sess.pdf",
                  scope="session", sid=SID_A)
        reviewer = FakeReviewer([(repair_verdict(["session bravo"]),
                                  canned_review_meta())])
        cfg = _cfg()
        _a, sources_a, _i, _r, _rev = query_reviewed(
            "session bravo",
            config=cfg, vector_store=store,
            llm_provider=FakeLLM("Zebra savanna chartreuse quantum."),
            session_id=SID_A, planner_bundle=None,
            reviewer_bundle=review_bundle_for(reviewer))
        self.assertTrue(any(s["file_name"] == "sess.pdf" for s in sources_a))


class TestConfig5E(unittest.TestCase):
    def test_defaults_disabled(self):
        cfg = _cfg()
        self.assertEqual(cfg.review_enabled, "0")
        self.assertEqual(cfg.review_provider, "")
        self.assertEqual(cfg.review_model_id, "")
        self.assertEqual(cfg.review_region, "us-east-1")
        self.assertEqual(cfg.review_timeout_s, 5)
        self.assertEqual(cfg.review_always, "0")

    def test_enabled_parsing(self):
        from answer_reviewer import review_always_enabled, review_enabled
        for on in ["1", "true", "yes", "on", "TRUE"]:
            self.assertTrue(review_enabled(_cfg(review_enabled=on)))
            self.assertTrue(review_always_enabled(_cfg(review_always=on)))
        for off in ["0", "", "false", "no"]:
            self.assertFalse(review_enabled(_cfg(review_enabled=off)))
            self.assertFalse(review_always_enabled(_cfg(review_always=off)))

    def test_bundle_build_rules(self):
        from answer_reviewer import get_reviewer_bundle
        self.assertIsNone(get_reviewer_bundle(_cfg()))
        with self.assertRaises(ValueError):
            get_reviewer_bundle(_cfg(review_enabled="1",
                                     review_provider="openai",
                                     review_model_id="m"))
        with self.assertRaises(ValueError):
            get_reviewer_bundle(_cfg(review_enabled="1",
                                     review_provider="bedrock",
                                     review_model_id=""))
        # Independent from answer/reasoning configuration.
        bundle_cfg = _cfg(review_enabled="1", review_provider="bedrock",
                          review_model_id="m1", review_region="us-east-1",
                          llm_provider="lmstudio",
                          reasoning_provider="bedrock",
                          reasoning_model_id="other")
        # Structurally valid configs pass field checks (bad configs above
        # already prove loud failures); construction itself needs
        # langchain-aws and is not exercised here.
        self.assertEqual(bundle_cfg.review_model_id, "m1")
        self.assertEqual(bundle_cfg.reasoning_model_id, "other")

    def test_structured_capability_documented(self):
        from answer_reviewer import supports_native_structured_output
        caps = supports_native_structured_output()
        self.assertEqual(caps["mode"], "forced-tool-calling")
        self.assertEqual(caps["native_outputConfig"], "unverified")
        self.assertIn("validator", caps)


class TestWorkflowsRegression5E(unittest.TestCase):
    def test_all_workflows_unchanged_with_deterministic(self):
        from answer_reviewer import (DeterministicReviewer, query_reviewed)
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
            bundle = review_bundle_for(DeterministicReviewer())
            got = query_reviewed(q, config=cfg, vector_store=s2,
                                 llm_provider=FakeLLM("A"),
                                 planner_bundle=None,
                                 reviewer_bundle=bundle)
            with self.subTest(q=q):
                self.assertEqual(stable_result(*got[:3]),
                                 stable_result(*expected))


class TestBenchmark5E(unittest.TestCase):
    def test_deterministic_parity(self):
        from tests.benchmarks.harness import run_all
        base = run_all()
        reviewed = run_all(reviewer="deterministic")
        bstatus = {r["id"]: r["status"] for r in base["results"]}
        pstatus = {r["id"]: r["status"] for r in reviewed["results"]}
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
                p = next(r for r in reviewed["results"] if r["id"] == rid)
                self.assertEqual(substance(p), substance(b))

    def test_reviewer_metrics_present(self):
        from tests.benchmarks.harness import run_all
        out = run_all(reviewer="deterministic")
        for r in out["results"]:
            if r["status"] in ("pass", "fail"):
                m = r["metrics"]
                for key in ("review_used", "review_verdict",
                            "review_trigger", "repair_attempted",
                            "repair_succeeded", "reviewer_latency_ms",
                            "guard_before_flagged",
                            "guard_after_flagged"):
                    self.assertIn(key, m, (r["id"], key))

    def test_repair_end_to_end(self):
        from tests.benchmarks.harness import run_all
        script = [(repair_verdict(["lock-in period lease deed"]),
                   canned_review_meta()) for _ in range(30)]
        reviewer = FakeReviewer(script)
        bundle = review_bundle_for(reviewer)
        out = run_all(reviewer=bundle)
        by_id = {r["id"]: r for r in out["results"]}
        self.assertIn("review", by_id["compare-lockin-notice"])
        self.assertIn("review_used", out["summary"])
        self.assertIn("repair_attempted", out["summary"])


if __name__ == "__main__":
    unittest.main()
