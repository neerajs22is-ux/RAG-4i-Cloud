"""Python 3.11 compatibility gate (matches deploy/setup_ec2.sh).

EC2 runs python3.11; local dev may be newer. Syntax legal only on 3.12+
— notably backslash sequences inside f-string expressions (PEP 701),
which pass the local suite yet break the deployed app at import —
must be rejected here.

Method: AST walk for FormattedValue nodes whose source segment contains
a backslash (the exact 3.11 restriction; literal f-string text is
unaffected), plus a feature_version=(3, 11) parse for other grammar
drift. No 3.11 interpreter needed.
"""

import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")

SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "chroma_db"}


def iter_app_sources():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        # Third-party / reference trees are not shipped to EC2.
        if "Bolt" in dirpath or "SKILLS" in dirpath:
            continue
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def fstring_backslashes(src):
    """(lineno, segment) for f-string expressions containing backslashes."""
    hits = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FormattedValue):
            seg = ast.get_source_segment(src, node)
            if seg and "\\" in seg:
                hits.append((node.lineno, seg.strip()[:100]))
    return hits


class TestPy311Grammar(unittest.TestCase):
    def test_no_backslash_in_fstring_expressions(self):
        failures, checked = [], 0
        for path in iter_app_sources():
            with open(path, encoding="utf-8") as f:
                src = f.read()
            checked += 1
            for lineno, seg in fstring_backslashes(src):
                failures.append(
                    f"{os.path.relpath(path, ROOT)}:{lineno}: {seg}")
        self.assertTrue(checked > 10, "expected to check the app sources")
        self.assertEqual(
            failures, [],
            f"SyntaxError under Python 3.11 (see deploy/setup_ec2.sh): {failures}")

    def test_all_app_sources_parse_as_311(self):
        checked = 0
        for path in iter_app_sources():
            with open(path, encoding="utf-8") as f:
                src = f.read()
            checked += 1
            ast.parse(src, filename=path, feature_version=(3, 11))
        self.assertTrue(checked > 10, "expected to check the app sources")

    def test_full_source_html_renders(self):
        # The exact construct class that broke EC2 (re.sub inside f-string).
        from ui.components import full_source_html
        html = full_source_html({"file_name": "lease.pdf", "page": 1,
                                 "score_text": "0.84",
                                 "content": "Terms  here.\nNext line."})
        self.assertIn("Terms here.", html)
        self.assertNotIn("/t/", html)


if __name__ == "__main__":
    unittest.main()
