"""Phase 4.2.2 tests: casual generalization + suggestions (deterministic)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCasualGeneralization(unittest.TestCase):
    """Unseen phrasings (not full sentences in query_router.py)."""

    UNSEEN_CASUAL = [
        "how's it going?",
        "how are things?",
        "how's your day?",
        "how is your day going?",
        "how have you been?",
        "hope you're doing well",
        "hope your day is going splendidly",
        "what's going on?",
        "what is new?",
        "hey there",
        "good day",
    ]

    def test_unseen_casual_is_conversation(self):
        from query_router import CONVERSATION, observe_query
        for msg in self.UNSEEN_CASUAL:
            with self.subTest(msg=msg):
                obs = observe_query(msg)
                self.assertEqual(obs.intent, CONVERSATION)
                self.assertFalse(obs.needs_retrieval)

    def test_capability_variations(self):
        from query_router import CAPABILITY, observe_query
        for msg in ["what can you do to assist?",
                    "in what ways can you help?",
                    "am I able to ask about contracts?"]:
            with self.subTest(msg=msg):
                obs = observe_query(msg)
                # capability or safe document fallback both acceptable;
                # must never be out-of-scope silence.
                self.assertIn(obs.intent, (CAPABILITY, "document_query"))

    def test_adversarial_substantive_stays_document(self):
        from query_router import DOCUMENT_QUERY, observe_query
        for msg in ["How is the lock-in going?",
                    "What is up with the lease?",
                    "How are the parties defined?",
                    "Hope the rent clause is clear: what is the rent?",
                    "Tell me the termination notice period"]:
            with self.subTest(msg=msg):
                obs = observe_query(msg)
                self.assertEqual(obs.intent, DOCUMENT_QUERY)
                self.assertTrue(obs.needs_retrieval)


class TestInitialSuggestions(unittest.TestCase):
    def test_exactly_three(self):
        from suggestions import initial_suggestions
        self.assertEqual(len(initial_suggestions(["lease.pdf"])), 3)
        self.assertEqual(len(initial_suggestions([])), 3)
        self.assertEqual(len(initial_suggestions(["a.pdf", "b.pdf", "c.pdf", "d.pdf"])), 3)

    def test_grounded_in_filenames(self):
        from suggestions import initial_suggestions
        out = initial_suggestions(["lease.pdf", "contract.pdf"])
        self.assertTrue(any("lease.pdf" in s for s in out))
        self.assertTrue(any("contract.pdf" in s for s in out))

    def test_empty_fallback_is_generic(self):
        from suggestions import initial_suggestions
        out = initial_suggestions([])
        self.assertEqual(len(out), 3)
        for s in out:
            self.assertNotIn("lease", s.lower())
            self.assertNotIn("contract", s.lower())


class TestGroundedInitialSuggestions(unittest.TestCase):
    def _retrieve(self, hits):
        def _fn(question):
            return [{"content": "x"}] if question in hits else []
        return _fn

    def test_keeps_only_retrieving_questions(self):
        from suggestions import grounded_initial_suggestions
        out = grounded_initial_suggestions(
            self._retrieve({"Tell me about lease.pdf"}),
            ["lease.pdf", "contract.pdf"])
        self.assertIn("Tell me about lease.pdf", out)
        self.assertEqual(len(out), 3)

    def test_falls_back_when_nothing_retrieves(self):
        from suggestions import grounded_initial_suggestions, initial_suggestions
        out = grounded_initial_suggestions(lambda q: [],
                                           ["lease.pdf", "contract.pdf"])
        self.assertEqual(out, initial_suggestions(["lease.pdf", "contract.pdf"]))

    def test_retrieval_errors_are_skipped(self):
        from suggestions import grounded_initial_suggestions

        def _boom(question):
            raise RuntimeError("down")

        out = grounded_initial_suggestions(_boom, ["lease.pdf"])
        self.assertEqual(len(out), 3)


class TestFollowupSuggestions(unittest.TestCase):
    def _src(self, content="The lock-in period is 36 months. Renewal needs notice.",
             fname="lease.pdf"):
        return [{"content": content, "file_name": fname, "page": 0,
                 "score": 0.8, "source": "/t/" + fname}]

    def test_exactly_three(self):
        from suggestions import followup_suggestions
        self.assertEqual(len(followup_suggestions("q", self._src())), 3)
        self.assertEqual(len(followup_suggestions("q", [])), 3)

    def test_adapts_to_context(self):
        from suggestions import followup_suggestions
        lease = followup_suggestions("lock-in?", self._src())
        rent = followup_suggestions(
            "rent?", self._src("Monthly rent is Rs 50000, deposit is 6 months.",
                               "rent.pdf"))
        self.assertNotEqual(lease, rent)
        self.assertTrue(any("lease" in s.lower() for s in " ".join(lease).split(". ")))

    def test_no_invented_facts(self):
        from suggestions import followup_suggestions
        out = followup_suggestions(
            "lock-in?", self._src("The lock-in period is 36 months.", "lease.pdf"))
        blob = " ".join(out).lower()
        for invented in ["paris", "france", "bitcoin", "cricket"]:
            self.assertNotIn(invented, blob)
        # Every topic word must come from context text or filenames.
        allowed = set("the lock-in period is 36 months lease pdf".split())
        generic = {"about", "else", "does", "say", "exceptions", "dates",
                   "parties", "involved", "mentioned", "described", "details",
                   "documents", "other", "these", "what", "key", "terms",
                   "obligations", "there"}
        for s in out:
            words = [w.strip("?.!,").lower().replace(".pdf", "")
                     for w in s.split()]
            content_words = [w for w in words if len(w) >= 5]
            for w in content_words:
                self.assertTrue(w in allowed or w in generic,
                                f"unsupported topic word: {w} in {s!r}")

    def test_empty_context_safe_fallback(self):
        from suggestions import followup_suggestions
        out = followup_suggestions("anything?", [])
        self.assertEqual(len(out), 3)
        self.assertTrue(all(isinstance(s, str) and s for s in out))


if __name__ == "__main__":
    unittest.main()
