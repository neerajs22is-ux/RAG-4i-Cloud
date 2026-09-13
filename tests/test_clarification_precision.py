"""Clarification-gate precision tests (Step 14, mocked, $0).

The gate still clarifies genuinely ambiguous requests but no longer
fires on a single missing verb-form term against specific, strong
evidence (S04/S05/S13/S15 class). Refusal/safety paths untouched.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SLA = ("Cloud SLA guarantees 99.9% uptime measured monthly. Service "
       "credit of 10% of the monthly fee applies per breach.")
ITSEC = ("Password rotation every 90 days is mandatory. Multifactor "
         "authentication is required for email, VPN, and HR systems.")


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})
        self.id = None


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def _match(self, query):
        from answer_support import _stem, content_terms
        qw = {_stem(w) for w in content_terms(query)}
        return [(c, 0.9) for c in self.chunks
                if qw & {_stem(w) for w in content_terms(c.page_content)}]

    def search(self, query, k=5):
        return self._match(query)[:k]

    def chunks_for_source(self, file_name, limit=8):
        return [(c, None) for c in self.chunks
                if c.metadata.get("file_name") == file_name][:limit]

    def get_status(self):
        return {"ready": True, "exists": True}


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, context, question, prompt_template):
        self.calls.append((context, question, prompt_template))
        return "ANSWERED:" + context[:30]

    def is_reachable(self, timeout=3.0):
        return True

    @property
    def describe(self):
        return "Fake"


def _doc(text, fname):
    return FakeDoc(text, {"file_name": fname, "page": 0,
                          "document_id": "d", "chunk_id": "c-" + fname})


class TestNeedsClarification(unittest.TestCase):
    def test_ratio_boundaries(self):
        from answer_support import needs_clarification as nc
        self.assertFalse(nc({"missing": ["applies"],
                             "terms": ["uptime", "cloud", "guarantee",
                                       "credit", "applies"]}))
        self.assertFalse(nc({"missing": ["grant"],
                             "terms": ["severance", "termination",
                                       "template", "grant"]}))
        self.assertTrue(nc({"missing": ["exceptions"],
                            "terms": ["exceptions", "renewal"]}))
        self.assertTrue(nc({"missing": ["a", "b"],
                            "terms": ["a", "b", "c"]}))
        self.assertFalse(nc({"missing": [], "terms": ["a"]}))
        self.assertFalse(nc({"missing": ["a"], "terms": []}))
        self.assertFalse(nc({}))

    def test_specific_request_answers_despite_verb_gap(self):
        import backend
        import config
        store = FakeStore([_doc(SLA, "sla-cloud.pdf")])
        llm = FakeLLM()
        ans, _ = backend.query_documents(
            "What uptime does the cloud SLA guarantee and what credit "
            "applies?", config=config.load_config(), vector_store=store,
            llm_provider=llm)
        self.assertTrue(llm.calls)
        self.assertNotIn("Quick check before I answer:", ans)

    def test_ambiguous_request_still_clarifies(self):
        import backend
        import config
        store = FakeStore([_doc("Renewal terms here.", "c.pdf")])
        llm = FakeLLM()
        ans, _ = backend.query_documents(
            "Are there exceptions to renewal?",
            config=config.load_config(), vector_store=store,
            llm_provider=llm)
        self.assertTrue(ans.startswith("Quick check before I answer:"))
        self.assertEqual(llm.calls, [])

    def test_bare_followup_still_clarifies(self):
        # Only the user's new word matters: "exceptions" matches nothing,
        # even though expanded context terms do. Same routing conditions
        # as the pinned follow-up chain test.
        import backend
        import config
        ctx = [{"role": "user",
                "content": "What are the renewal terms?"},
               {"role": "assistant",
                "content": "The service contract renews annually with "
                           "30 days notice before each renewal date."}]
        store = FakeStore([_doc("The service contract renews annually "
                                "with 30 days notice. Payment terms are "
                                "net 15 days.", "c.pdf")])
        llm = FakeLLM()
        ans, _ = backend.query_documents(
            "What about exceptions?", config=config.load_config(),
            vector_store=store, llm_provider=llm,
            conversation_context=ctx)
        self.assertTrue(ans.startswith("Quick check before I answer:"))
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
