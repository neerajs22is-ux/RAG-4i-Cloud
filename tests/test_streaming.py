"""Streaming tests: deterministic mocked provider, sanitizer, fallback."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FINAL = "The lock-in period is 36 months."


class MockStreamProvider:
    """Emits scripted chunks via generate_stream; whole via generate."""

    def __init__(self, chunks, whole=None, fail_at=None):
        self.chunks = list(chunks)
        self.whole = whole if whole is not None else "".join(chunks)
        self.fail_at = fail_at
        self.generate_calls = 0
        self.stream_calls = 0
        self.seen = []

    def generate(self, context, question, prompt_template):
        self.generate_calls += 1
        self.seen.append((context, question, prompt_template))
        return self.whole

    def generate_stream(self, context, question, prompt_template):
        self.stream_calls += 1
        self.seen.append((context, question, prompt_template))
        for i, chunk in enumerate(self.chunks):
            if self.fail_at is not None and i >= self.fail_at:
                raise ConnectionError("stream dropped")
            yield chunk

    def is_reachable(self, timeout=3.0):
        return True

    @property
    def describe(self):
        return "MockStream"


def src(score=0.8):
    return {"content": "The lock-in period is 36 months.",
            "source": "/t/lease.pdf", "source_path": "/t/lease.pdf",
            "file_name": "lease.pdf", "page": 0, "score": score,
            "document_id": "d", "chunk_id": "c"}


def stream_text(provider, **kw):
    import backend
    import config
    info, stream = backend.stream_answer(
        "What is the lock-in period?", config=config.load_config(),
        llm_provider=provider, **kw)
    return info, "".join(stream)


class TestSanitizer(unittest.TestCase):
    def _run(self, chunks):
        from output_safety import ThinkStreamSanitizer
        s = ThinkStreamSanitizer()
        out = "".join(s.feed(c) for c in chunks) + s.flush()
        return out

    def test_think_within_one_chunk(self):
        self.assertEqual(
            self._run(["<think>hmm</think>Final answer"]), "Final answer")

    def test_think_split_across_chunks(self):
        self.assertEqual(
            self._run(["<think>", "private reasoning", "</think>",
                       "Final answer"]),
            "Final answer")

    def test_opening_tag_split(self):
        self.assertEqual(self._run(["<thi", "nk>secret</think>Final answer"]),
                         "Final answer")

    def test_closing_tag_split(self):
        self.assertEqual(self._run(["<think>secret</th", "ink>Final answer"]),
                         "Final answer")

    def test_multiple_blocks(self):
        self.assertEqual(
            self._run(["<think>a</think>mid", "<think>b</think>end"]),
            "midend")

    def test_no_partial_leak(self):
        from output_safety import ThinkStreamSanitizer
        s = ThinkStreamSanitizer()
        partials = [s.feed(c) for c in ["<think>", "secret"]]
        self.assertEqual("".join(partials), "")

    def test_lookalike_literal(self):
        self.assertIn("<thinker>", self._run(["a <thinker> b"]))

    def test_unclosed_drops_open_block(self):
        # Deliberate, documented divergence from strip_think_blocks():
        # a still-open block at stream end is dropped rather than shown,
        # because it is indistinguishable from reasoning in progress.
        self.assertEqual(self._run(["text <think>oops"]), "text ")
        # A merely held "<"-prefix with no think content emits literally.
        self.assertEqual(self._run(["a < b"]), "a < b")


class TestStreamBackend(unittest.TestCase):
    def test_normal_streamed_response(self):
        p = MockStreamProvider(["The lock-in ", "period is 36 months."])
        info, text = stream_text(p, vector_store=_Store())
        self.assertEqual(text, FINAL)
        self.assertTrue(info["retrieved"])

    def test_stream_equals_nonstream(self):
        import backend
        import config
        chunks = ["<think>reasoning here</think>", "The answer is ",
                  "36 months."]
        p1 = MockStreamProvider(chunks)
        _, streamed = stream_text(p1, vector_store=_Store())
        p2 = MockStreamProvider(chunks)
        ans, _ = backend.query_documents(
            "What is the lock-in period?", config=config.load_config(),
            vector_store=_Store(), llm_provider=p2)
        self.assertEqual(streamed, ans)

    def test_unsupported_never_calls_llm(self):
        import backend
        import config

        class EmptyStore(_Store):
            def search(self, query, k=5):
                return []

        p = MockStreamProvider(["x"])
        info, text = stream_text(p, vector_store=EmptyStore())
        self.assertEqual(p.stream_calls + p.generate_calls, 0)
        self.assertIn("could not find", text.lower())

    def test_midstream_failure_raises_for_fallback(self):
        import backend
        import config
        p = MockStreamProvider(["partial ", "more"], fail_at=1)
        info, stream = backend.stream_answer(
            "What is the lock-in period?", config=config.load_config(),
            vector_store=_Store(), llm_provider=p)
        with self.assertRaises(ConnectionError):
            list(stream)
        # Fallback reuses the same evidence without re-retrieving.
        fb = backend.generate_answer(
            "What is the lock-in period?", info["retrieved"],
            config=config.load_config(), llm_provider=p)
        self.assertTrue(fb)

    def test_prompt_byte_identical(self):
        import backend
        p = MockStreamProvider(["ok"])
        stream_text(p, vector_store=_Store())
        tpl = p.seen[0][2]
        self.assertEqual(tpl, backend.PROMPT_TEMPLATE)


class FakeDoc:
    page_content = "The lock-in period is 36 months."
    metadata = {"source_path": "/t/lease.pdf",
                "source": "/t/lease.pdf", "file_name": "lease.pdf",
                "page": 0, "document_id": "d", "chunk_id": "c"}
    id = None


class _Store:
    def search(self, query, k=5):
        return [(FakeDoc(), 0.9)]

    def chunks_for_source(self, file_name, limit=8):
        return []

    def get_status(self):
        return {"ready": True}


if __name__ == "__main__":
    unittest.main()
