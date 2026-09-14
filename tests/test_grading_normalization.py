"""Grader normalization tests (Step 15, mocked, $0).

Formatting-only differences (digit grouping, possessives) must not flip
grades. Semantic distinctions (different numbers, units, negations,
partial facts) must still fail. Golds and scenarios untouched.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _scenario(required=(), forbidden=()):
    return {"query": "Q?", "required_facts": list(required),
            "forbidden_claims": list(forbidden),
            "expected_documents": [], "expected_workflow": None,
            "expected_support_level": "direct",
            "expected_clarification": None, "expected_scope": "persistent",
            "severity": "material"}


def _grade(answer, required=(), forbidden=()):
    from tests.benchmarks.isolations import grade_deterministic
    return grade_deterministic(
        scenario=_scenario(required, forbidden), answer=answer,
        sources=[], info={}, evidence=[])


class TestNumericEquivalence(unittest.TestCase):
    def test_comma_grouping_matches(self):
        g = _grade("Revenue was Rs 450,000,000.",
                   required=["450000000"])
        self.assertEqual(
            [f for f in g["failures"] if f.startswith("missing")], [])

    def test_space_grouping_matches(self):
        g = _grade("Revenue was Rs 450 000 000.",
                   required=["450000000"])
        self.assertEqual(
            [f for f in g["failures"] if f.startswith("missing")], [])

    def test_correct_longer_number_not_banned(self):
        # S07-ARM1 class: correct Rs 450000000 must not trip the
        # shorter banned Rs 45000000 via substring collision.
        g = _grade("Revenue was Rs 450000000.",
                   required=["450000000"],
                   forbidden=["Rs 45000000"])
        self.assertEqual(g["failures"], [])

    def test_genuinely_banned_number_still_fires(self):
        g = _grade("Revenue was Rs 45000000.", forbidden=["Rs 45000000"])
        self.assertTrue(any(f.startswith("forbidden") for f in g["failures"]))

    def test_different_number_still_misses(self):
        g = _grade("Revenue was Rs 450000001.",
                   required=["450000000"])
        self.assertTrue(any(f.startswith("missing") for f in g["failures"]))

    def test_partial_number_is_not_a_match(self):
        # 1000000 required; answer holding only 10000000 must not pass.
        g = _grade("Cap is 10000000.", required=["1000000"])
        self.assertTrue(any(f.startswith("missing") for f in g["failures"]))


class TestPossessiveEquivalence(unittest.TestCase):
    def test_s07_style_possessive_matches(self):
        g = _grade("Grants one month's salary per year.",
                   required=["one month salary"])
        self.assertEqual(
            [f for f in g["failures"] if f.startswith("missing")], [])

    def test_curly_apostrophe_matches(self):
        g = _grade("Grants one month’s salary per year.",
                   required=["one month salary"])
        self.assertEqual(
            [f for f in g["failures"] if f.startswith("missing")], [])

    def test_unrelated_fact_still_misses(self):
        g = _grade("Grants two weeks notice.",
                   required=["one month salary"])
        self.assertTrue(any(f.startswith("missing") for f in g["failures"]))


class TestSemanticProtections(unittest.TestCase):
    def test_units_preserved(self):
        g = _grade("Rotation every 90 days.", required=["90 days"])
        self.assertEqual(
            [f for f in g["failures"] if f.startswith("missing")], [])
        g = _grade("Rotation every 90 hours.", required=["90 days"])
        self.assertTrue(any(f.startswith("missing") for f in g["failures"]))

    def test_negation_preserved(self):
        g = _grade("Termination is not permitted.",
                   required=["not permitted"])
        self.assertEqual(
            [f for f in g["failures"] if f.startswith("missing")], [])
        g = _grade("Termination is permitted.",
                   required=["not permitted"])
        self.assertTrue(any(f.startswith("missing") for f in g["failures"]))

    def test_percent_preserved(self):
        g = _grade("Escalation is 5 percent.", required=["5 percent"])
        self.assertEqual(
            [f for f in g["failures"] if f.startswith("missing")], [])
        g = _grade("Escalation is 5 percent.", forbidden=["7 percent"])
        self.assertEqual(g["failures"], [])


if __name__ == "__main__":
    unittest.main()
