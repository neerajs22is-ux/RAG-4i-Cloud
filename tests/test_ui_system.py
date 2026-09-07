"""Frontend system tests: tokens, stylesheets, presentation helpers."""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _hexes(text):
    return re.findall(r"#[0-9a-fA-F]{6}\b", text)


def _luminance(hexcode):
    c = [int(hexcode[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast(fg, bg):
    a, b = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


class TestTokens(unittest.TestCase):
    def test_required_groups_and_keys(self):
        from ui.tokens import TOKENS
        for key in ("background", "surface", "text", "text-secondary",
                    "text-muted", "border", "accent", "success", "warning",
                    "error"):
            self.assertIn(key, TOKENS["color"])
            self.assertIn(key, TOKENS["dark"])
        for key in ("xs", "sm", "md", "lg", "xl"):
            self.assertIn(key, TOKENS["space"])
        for key in ("sm", "md", "lg", "pill"):
            self.assertIn(key, TOKENS["radius"])

    def test_text_contrast_aa(self):
        from ui.tokens import TOKENS
        for mode, bg in (("color", TOKENS["color"]["background"]),
                         ("dark", TOKENS["dark"]["background"])):
            for fg_key in ("text", "text-secondary", "text-muted"):
                fg = TOKENS[mode][fg_key]
                self.assertGreaterEqual(
                    _contrast(fg, bg), 4.5, f"{mode}.{fg_key} on bg")

    def test_soft_pair_contrast(self):
        from ui.tokens import TOKENS
        pairs = [("accent-text", "accent-soft"), ("success", "success-soft"),
                 ("warning", "warning-soft"), ("error", "error-soft")]
        for mode in ("color", "dark"):
            for fg_key, bg_key in pairs:
                ratio = _contrast(TOKENS[mode][fg_key], TOKENS[mode][bg_key])
                self.assertGreaterEqual(ratio, 3.0, f"{mode}.{fg_key}")

    def test_css_variables_render(self):
        from ui.tokens import css_variables
        css = css_variables()
        self.assertIn(":root", css)
        self.assertIn("--rag-color-accent", css)
        self.assertIn("prefers-color-scheme: dark", css)


class TestStylesheets(unittest.TestCase):
    FILES = ["ui/css/base.css", "ui/css/chat.css"]

    def test_files_exist_and_reference_tokens(self):
        for name in self.FILES:
            with open(os.path.join(ROOT, name), encoding="utf-8") as f:
                css = f.read()
            self.assertIn("var(--rag-", css, name)

    def test_no_hardcoded_hex_outside_tokens(self):
        """Colors live in tokens.py; stylesheets use variables."""
        for name in self.FILES:
            with open(os.path.join(ROOT, name), encoding="utf-8") as f:
                css = f.read()
            for hx in _hexes(css):
                self.assertIn(
                    hx.lower(), ("#0f172a",),
                    f"{name} hardcodes {hx} (move to tokens)")

    def test_accessibility_rules_present(self):
        with open(os.path.join(ROOT, "ui/css/base.css"), encoding="utf-8") as f:
            base = f.read()
        self.assertIn(":focus-visible", base)
        self.assertIn("prefers-reduced-motion", base)
        self.assertIn("max-width: 640px", base)

    def test_no_banned_aesthetics(self):
        import glob
        for path in glob.glob(os.path.join(ROOT, "ui/css/*.css")):
            with open(path, encoding="utf-8") as f:
                css = f.read().lower()
            self.assertNotIn("linear-gradient", css)
            self.assertNotIn("backdrop-filter", css)
            self.assertNotIn("neon", css)


class TestComponents(unittest.TestCase):
    def test_source_rows_preserve_data(self):
        from ui.components import format_source_rows
        rows = format_source_rows([
            {"file_name": "lease.pdf", "page": 1, "score": 0.866},
            {"file_name": "lease.pdf", "page": 1, "score": 0.5},
            {"file_name": None, "page": None, "score": None},
        ])
        self.assertEqual(len(rows), 2)  # deduped
        self.assertEqual(rows[0]["file_name"], "lease.pdf")
        self.assertEqual(rows[0]["page"], 1)
        self.assertEqual(rows[0]["score_text"], "0.87")
        self.assertEqual(rows[1]["score_text"], "—")

    def test_friendly_error_mapping(self):
        from ui.components import friendly_error
        mapped = friendly_error("LLM endpoint is unreachable. Make sure LM Studio Server is running!")
        self.assertIn("temporarily unable", mapped)
        self.assertNotIn("LM Studio", mapped)
        mapped2 = friendly_error("I could not find enough relevant information in the documents.")
        self.assertIn("connected documents", mapped2)
        self.assertEqual(friendly_error("Hi there!"), "Hi there!")

    def test_welcome_copy(self):
        from ui.components import WELCOME_BODY, WELCOME_TITLE
        self.assertEqual(WELCOME_TITLE, "RAG-4i")
        self.assertIn("document", WELCOME_BODY.lower())

    def test_load_styles_combines(self):
        from ui.components import load_styles
        css = load_styles()
        self.assertIn("<style>", css)
        self.assertIn("--rag-color-background", css)
        self.assertIn("stChatMessage", css)


if __name__ == "__main__":
    unittest.main()
