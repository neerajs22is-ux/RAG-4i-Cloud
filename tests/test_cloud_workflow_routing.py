"""Routing-correctness tests for the Step-12 harness fix (mocked, $0).

Proves run_cloud_scenario selects the same workflow the production app
would select for S22/S23/S24 (comparison), S25/S27 (summary), S40
(normal + session binding) -- without paid calls or architecture edits.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmarks"))


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})


class FakeStore:
    """Duck-typed store recording scope bindings."""

    def __init__(self):
        self.builds = []
        self.searches = []

    def build_index(self, chunks, session_id=None):
        chunks = list(chunks or [])
        self.builds.append((len(chunks), session_id))
        return len(chunks)

    def search(self, query, k=5, session_id=None):
        self.searches.append((query, session_id))
        return [(FakeDoc("lease lock-in period text alpha",
                         {"file_name": "lease.pdf", "page": 0,
                          "document_id": "d1", "chunk_id": "c1"}), 0.9)]

    def chunks_for_source(self, file_name, limit=8, session_id=None):
        return [(FakeDoc("source top-up text",
                         {"file_name": file_name, "page": 0,
                          "document_id": "d9", "chunk_id": "c9"}), None)]

    def list_sources(self, limit=50, session_id=None):
        return [{"file_name": n, "document_id": f"d{i}"} for i, n in
                enumerate(["lease-riverside.pdf", "lease-harbourview.pdf",
                           "employee-handbook.pdf", "msa-techservices.pdf",
                           "service-maintenance.pdf",
                           "service-catering.pdf"])]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, context, question, prompt_template):
        self.calls.append((context, question))
        return "canned answer"


def _scenarios():
    import json
    path = os.path.join(os.path.dirname(__file__), "benchmarks",
                        "scenarios_v1.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {s["scenario_id"]: s for s in data["scenarios"]}


class TestCloudRouting(unittest.TestCase):
    SIX = ["S22-compare-parties", "S23-compare-service",
           "S24-compare-maint", "S25-summary-handbook",
           "S27-summary-msa", "S40-session"]

    def _run(self, sid):
        from cloud_workflow import run_cloud_scenario
        import config
        config.reset_config_cache()
        store, llm = FakeStore(), FakeLLM()
        out = run_cloud_scenario(_scenarios()[sid], config=config.load_config(),
                                 vector_store=store, llm_provider=llm)
        return out, store, llm

    def test_comparison_routes(self):
        for sid in self.SIX[:3]:
            out, _, llm = self._run(sid)
            self.assertEqual(out["workflow"], "comparison", sid)
            self.assertTrue(llm.calls, sid)  # comparison generates

    def test_summary_routes(self):
        for sid in self.SIX[3:5]:
            out, _, _ = self._run(sid)
            self.assertEqual(out["workflow"], "summary", sid)

    def test_session_binds_own(self):
        out, store, _ = self._run("S40-session")
        self.assertEqual(out["workflow"], "normal")
        self.assertIsNotNone(out["session_id"])
        self.assertEqual(len(store.builds), 1)
        n_chunks, build_sid = store.builds[0]
        self.assertGreater(n_chunks, 0)
        self.assertEqual(build_sid, out["session_id"])
        # Retrieval ran under the same bound session.
        self.assertTrue(store.searches)
        for _, search_sid in store.searches:
            self.assertEqual(search_sid, out["session_id"])

    def test_no_paid_calls_possible(self):
        # The harness path uses only the injected provider; routing
        # itself never touches the network (detection is structural).
        from workflows import detect_workflow
        for sid in self.SIX:
            det = detect_workflow(_scenarios()[sid]["query"])
            out, _, _ = self._run(sid)
            self.assertEqual(out["workflow"], det["workflow"], sid)


if __name__ == "__main__":
    unittest.main()
