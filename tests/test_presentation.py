"""Presentation tests: excerpts, strength signal, copy, keyboard hints."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def src(content="The lock-in period is 36 months.", score=0.84):
    return {"content": content, "file_name": "lease.pdf", "page": 1,
            "score": score, "source": "/t/lease.pdf",
            "source_path": "/t/lease.pdf", "document_id": "d",
            "chunk_id": "c"}


class TestExcerpts(unittest.TestCase):
    def test_excerpt_present_and_escaped(self):
        from ui.components import source_row_html
        row = {"file_name": "lease.pdf", "page": 1, "score_text": "0.84",
               "content": "The lock-in period is 36 months. " * 40}
        html = source_row_html(row)
        self.assertIn("lock-in period", html)
        self.assertIn("lease.pdf", html)

    def test_script_like_text_escaped(self):
        from ui.components import source_row_html
        row = {"file_name": "<img src=x onerror=alert(1)>.pdf", "page": 0,
               "score_text": "0.50",
               "content": "<script>alert('xss')</script> lease terms here"}
        html = source_row_html(row)
        self.assertNotIn("<script>alert", html)
        self.assertNotIn("<img src=x", html)
        self.assertIn("lease terms here", html)

    def test_excerpt_trims_word_boundary(self):
        from ui.components import source_row_html
        row = {"file_name": "a.pdf", "page": 0, "score_text": "0.50",
               "content": "word " * 200}
        html = source_row_html(row)
        self.assertLess(len(html), len("word " * 200) + 400)


class TestStrength(unittest.TestCase):
    def test_high_medium_low(self):
        from ui.components import retrieval_strength
        self.assertEqual(
            retrieval_strength([src(score=0.84)]), "Retrieval strength · 0.84")
        self.assertEqual(
            retrieval_strength([src(score=0.55)]), "Retrieval strength · 0.55")
        self.assertEqual(
            retrieval_strength([src(score=0.31)]), "Retrieval strength · 0.31")

    def test_missing_and_unsupported(self):
        from ui.components import retrieval_strength
        self.assertIsNone(retrieval_strength([]))
        self.assertIsNone(retrieval_strength([src(score=None)]))
        self.assertNotIn("confidence", retrieval_strength([src()]).lower())
        self.assertNotIn("probability",
                         retrieval_strength([src()]).lower())


class TestCopy(unittest.TestCase):
    def test_payload_is_final_answer(self):
        from ui.components import copy_button_html
        html = copy_button_html("Final persisted answer.", "copy_7")
        self.assertIn("Final persisted answer.", html)
        self.assertIn('id=', html)

    def test_ids_unique_per_message(self):
        from ui.components import copy_button_html
        a = copy_button_html("one", "copy_3")
        b = copy_button_html("two", "copy_9")
        self.assertNotEqual(a, b)
        self.assertIn("id='rag-copy-copy_3'", a)
        self.assertIn("id='rag-copy-copy_9'", b)
        self.assertNotIn("previousElementSibling", a)


if __name__ == "__main__":
    unittest.main()
