"""Answer-reasoning tests: support levels, partial/unsupported, follow-ups."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LEASE = ("The lock-in period in the lease deed is 36 months. Early "
         "termination requires 3 months notice.")
CONTRACT = ("The service contract renews annually with 30 days notice. "
            "Payment terms are net 15 days.")


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})
        self.id = None


def lease_src():
    return {"content": LEASE, "source": "/t/lease.pdf",
            "source_path": "/t/lease.pdf", "file_name": "lease.pdf",
            "page": 0, "score": 0.8, "document_id": "d1", "chunk_id": "c1"}


def contract_src():
    return {"content": CONTRACT, "source": "/t/contract.pdf",
            "source_path": "/t/contract.pdf", "file_name": "contract.pdf",
            "page": 0, "score": 0.7, "document_id": "d2", "chunk_id": "c2"}


class FakeStore:
    """Deterministic retrieval by keyword (no model needed)."""

    def __init__(self, chunks):
        self.chunks = chunks

    def build_index(self, chunks):
        self.chunks.extend(list(chunks))
        return len(chunks)

    def _match(self, query):
        from answer_support import _stem, content_terms
        qw = {_stem(w) for w in content_terms(query)}
        out = []
        for c in self.chunks:
            cw = {_stem(w) for w in content_terms(c.page_content)}
            if qw & cw:
                out.append((c, 0.9))
        return out

    def search(self, query, k=5):
        return self._match(query)[:k]

    def chunks_for_source(self, file_name, limit=8):
        return [(c, None) for c in self.chunks
                if c.metadata.get("file_name") == file_name][:limit]

    def get_status(self):
        return {"ready": True, "exists": True, "chunk_count": 1,
                "document_count": 1, "persist_directory": "fake"}


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, context, question, prompt_template):
        self.calls.append((context, question, prompt_template))
        marker = "PARTIAL" if "plainly" in prompt_template else "DIRECT"
        return f"{marker}:{context[:30]}"

    def is_reachable(self, timeout=3.0):
        return True

    @property
    def describe(self):
        return "Fake"


def make_store():
    def doc(text, fname, page=0):
        return FakeDoc(text, {"source_path": "/t/" + fname,
                              "source": "/t/" + fname, "file_name": fname,
                              "page": page, "document_id": fname,
                              "chunk_id": fname + str(page)})
    return FakeStore([doc(LEASE, "lease.pdf"), doc(CONTRACT, "contract.pdf")])


class TestSupportLevels(unittest.TestCase):
    def test_direct(self):
        from answer_support import DIRECT, assess_support
        for q in ["How often does the contract renew?",
                  "What are the payment terms?"]:
            level = assess_support(q, [lease_src(), contract_src()])["level"]
            self.assertEqual(level, DIRECT, q)

    def test_partial(self):
        from answer_support import PARTIAL, assess_support
        level = assess_support("Are there exceptions to renewal?",
                               [lease_src(), contract_src()])["level"]
        self.assertEqual(level, PARTIAL)

    def test_broad_overview_uses_scoping_prompt(self):
        import backend
        import config
        llm = FakeLLM()
        ans, src = backend.query_documents(
            "Tell me more about service.",
            config=config.load_config(), vector_store=make_store(),
            llm_provider=llm)
        self.assertTrue(ans.startswith("PARTIAL:"))
        self.assertTrue(src)

    def test_unsupported(self):
        from answer_support import UNSUPPORTED, assess_support
        level = assess_support("What is the supplier's insurance coverage?",
                               [lease_src(), contract_src()])["level"]
        self.assertEqual(level, UNSUPPORTED)


class TestPipeline(unittest.TestCase):
    def test_direct_uses_original_prompt(self):
        import backend
        import config
        llm = FakeLLM()
        ans, src = backend.query_documents(
            "How often does the contract renew?",
            config=config.load_config(), vector_store=make_store(),
            llm_provider=llm)
        self.assertTrue(ans.startswith("DIRECT:"))
        self.assertTrue(src)

    def test_partial_states_and_cites(self):
        import backend
        import config
        llm = FakeLLM()
        ans, src = backend.query_documents(
            "Are there exceptions to renewal?",
            config=config.load_config(), vector_store=make_store(),
            llm_provider=llm)
        # PARTIAL now clarifies first (no LLM call), naming both sides.
        self.assertTrue(ans.startswith("Quick check before I answer:"))
        self.assertIn("renewal", ans)
        self.assertIn("exceptions", ans)
        self.assertEqual(llm.calls, [])
        self.assertTrue(src)
        self.assertEqual(src[0]["file_name"], "contract.pdf")

    def test_unsupported_no_llm_call(self):
        import backend
        import config

        class NarrowStore(FakeStore):
            def search(self, query, k=5):
                return self._match("renewal")[:k]

        llm = FakeLLM()
        ans, src = backend.query_documents(
            "What is the supplier's insurance coverage?",
            config=config.load_config(), vector_store=NarrowStore(
                make_store().chunks),
            llm_provider=llm)
        self.assertEqual(llm.calls, [])
        self.assertIn("insurance coverage", ans)
        self.assertNotIn("I cannot find this information", ans)

    def test_followup_chain(self):
        import backend
        import config
        ctx = [{"role": "user", "content": "What are the renewal terms?"},
               {"role": "assistant",
                "content": "The service contract renews annually with "
                           "30 days notice before each renewal date."}]
        llm = FakeLLM()
        ans, src = backend.query_documents(
            "What about exceptions?", config=config.load_config(),
            vector_store=make_store(), llm_provider=llm,
            conversation_context=ctx)
        # Followup anchors renewal: retrieval hit, then one clarification.
        self.assertTrue(ans.startswith("Quick check before I answer:"))
        self.assertEqual(llm.calls, [])
        llm2 = FakeLLM()
        ans2, _ = backend.query_documents(
            "And payment?", config=config.load_config(),
            vector_store=make_store(), llm_provider=llm2,
            conversation_context=ctx)
        self.assertTrue(ans2.startswith("DIRECT:"))


class TestContextHygiene(unittest.TestCase):
    def test_greeting_no_contamination(self):
        import backend
        import config
        llm = FakeLLM()
        ctx = [{"role": "user", "content": "What is the lock-in period?"},
               {"role": "assistant", "content": "36 months, see lease."}]
        ans, src = backend.query_documents(
            "Thanks!", config=config.load_config(),
            vector_store=make_store(), llm_provider=llm,
            conversation_context=ctx)
        self.assertEqual(ans, "You're welcome!")
        self.assertEqual(src, [])
        self.assertEqual(llm.calls, [])

    def test_out_of_scope_no_contamination(self):
        import backend
        import config
        llm = FakeLLM()
        ans, src = backend.query_documents(
            "What is the capital of France?", config=config.load_config(),
            vector_store=make_store(), llm_provider=llm,
            conversation_context=[
                {"role": "user", "content": "Tell me about the lease."}])
        self.assertNotIn("Paris", ans)
        self.assertEqual(src, [])


class TestGrounding(unittest.TestCase):
    def test_no_fabricated_terms(self):
        import backend
        import config
        llm = FakeLLM()
        ans, _ = backend.query_documents(
            "Are there exceptions to renewal?",
            config=config.load_config(), vector_store=make_store(),
            llm_provider=llm)
        for invented in ["automatically renews unless", "Section 8",
                         "cancellation fee"]:
            self.assertNotIn(invented, ans)

    def test_sources_intact(self):
        import backend
        import config
        _, src = backend.query_documents(
            "How often does the contract renew?",
            config=config.load_config(), vector_store=make_store(),
            llm_provider=FakeLLM())
        for s in src:
            for key in ("source", "file_name", "page", "score"):
                self.assertIn(key, s)


class TestClarification(unittest.TestCase):
    CTX = [{"role": "user", "content": "What are the renewal terms?"},
           {"role": "assistant",
            "content": "The service contract renews annually with "
                       "30 days notice before each renewal date."}]

    def test_clarification_text(self):
        from answer_support import (CLARIFICATION_MARKER,
                                    build_clarification)
        text = build_clarification(
            "Are there exceptions to renewal?",
            {"covered": ["renewal"], "missing": ["exceptions"]},
            [contract_src()])
        self.assertTrue(text.startswith(CLARIFICATION_MARKER))
        self.assertIn("renewal", text)
        self.assertIn("exceptions", text)
        self.assertIn("contract.pdf", text)
        # No claims about unseen sections.
        self.assertNotIn("termination section says", text)

    def test_no_loop_second_partial_answers(self):
        import backend
        import config
        llm = FakeLLM()
        ans1, _ = backend.query_documents(
            "Are there exceptions to renewal?",
            config=config.load_config(), vector_store=make_store(),
            llm_provider=llm)
        self.assertTrue(ans1.startswith("Quick check"))
        ctx = [{"role": "user",
                "content": "Are there exceptions to renewal?"},
               {"role": "assistant", "content": ans1}]
        ans2, src2 = backend.query_documents(
            "yes", config=config.load_config(), vector_store=make_store(),
            llm_provider=llm, conversation_context=ctx)
        self.assertTrue(ans2.startswith("PARTIAL:"))
        self.assertTrue(src2)
        # A further "yes" does not clarify again (no loop).
        ctx2 = ctx + [{"role": "user", "content": "yes"},
                      {"role": "assistant", "content": ans2}]
        ans3, _ = backend.query_documents(
            "yes", config=config.load_config(), vector_store=make_store(),
            llm_provider=llm, conversation_context=ctx2)
        self.assertFalse(ans3.startswith("Quick check"))

    def test_yes_without_clarification_is_ordinary(self):
        import backend
        import config
        llm = FakeLLM()
        ans, _ = backend.query_documents(
            "yes", config=config.load_config(), vector_store=make_store(),
            llm_provider=llm)
        self.assertFalse(ans.startswith("Quick check"))

    def test_memory_object_context_works(self):
        import backend
        import config
        from conversation_memory import ConversationMemory
        mem = ConversationMemory()
        mem.add("user", "Are there exceptions to renewal?")
        llm = FakeLLM()
        ans, src = backend.query_documents(
            "Are there exceptions to renewal?",
            config=config.load_config(), vector_store=make_store(),
            llm_provider=llm, conversation_context=mem)
        self.assertTrue(ans.startswith("Quick check"))
        mem.add("assistant", ans)
        mem.add("user", "yes")
        ans2, _ = backend.query_documents(
            "yes", config=config.load_config(), vector_store=make_store(),
            llm_provider=llm, conversation_context=mem)
        self.assertTrue(ans2.startswith("PARTIAL:"))


class TestCitationGuard(unittest.TestCase):
    def test_grounded_passes_clean(self):
        from answer_support import citation_guard
        rep = citation_guard(
            "The contract renews annually with 30 days notice.",
            [contract_src()])
        self.assertEqual(rep["count"], 0)
        self.assertGreater(rep["total"], 0)

    def test_fabricated_clause_flagged(self):
        from answer_support import citation_guard
        rep = citation_guard(
            "The contract renews annually. There is a $500 cancellation "
            "fee under Section 8.",
            [contract_src()])
        self.assertGreater(rep["count"], 0)
        self.assertTrue(any("cancellation" in s or "Section 8" in s
                            for s in rep["flagged"]))

    def test_refusal_never_flagged(self):
        from answer_support import citation_guard
        for text in ["I could not find enough relevant information.",
                     "Quick check before I answer: x",
                     ""]:
            rep = citation_guard(text, [contract_src()])
            self.assertEqual(rep["count"], 0)


class TestGroundedSuggestions(unittest.TestCase):
    def test_exceptions_gated(self):
        from suggestions import followup_suggestions
        plain = followup_suggestions("renewal?", [contract_src()])
        self.assertFalse(any("exception" in s for s in plain))
        cued = followup_suggestions(
            "renewal?", [dict(contract_src(),
                              content=CONTRACT + " No exceptions unless agreed.")])
        self.assertTrue(any("exception" in s for s in cued))

    def test_exactly_three(self):
        from suggestions import followup_suggestions
        self.assertEqual(len(followup_suggestions("q", [lease_src()])), 3)


class TestQueryForms(unittest.TestCase):
    def test_plain_query_has_two_forms(self):
        import backend
        forms = backend.build_query_forms("What is the lock-in period?")
        self.assertEqual(len(forms), 2)
        self.assertIn("lock-in", forms[1])

    def test_original_always_participates(self):
        import backend
        forms = backend.build_query_forms("How long is the lock-in?")
        self.assertTrue(forms[0].lower().startswith("how long"))


if __name__ == "__main__":
    unittest.main()
