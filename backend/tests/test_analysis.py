from __future__ import annotations

import unittest

from backend.analysis import (
    BaseSemanticMarkdownAnalyzer,
    assign_score_bands,
    build_sentence_token_spans,
    markdown_to_analysis_text,
    normalize_scores,
    pack_sentence_windows,
    TokenSpan,
)


class DummyTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        class Encoded:
            def __init__(self, size: int) -> None:
                self.input_ids = list(range(size))

        return Encoded(max(1, len(text.split())))


class DummyAnalyzer(BaseSemanticMarkdownAnalyzer):
    def compute_raw_scores(self, sentences):
        return [float(index) for index, _ in enumerate(sentences)]


class AnalysisHelpersTestCase(unittest.TestCase):
    def test_markdown_preprocessing_ignores_code_math_and_images(self):
        markdown = """# Heading

Paragraph text with [a link](https://example.com) and `inline code`.

```python
print("skip me")
```

![diagram](image.png)

| Col A | Col B |
| --- | --- |
| Keep | This |

Math: $x^2 + y^2$.
"""
        processed = markdown_to_analysis_text(markdown)

        self.assertIn("Heading", processed)
        self.assertIn("Paragraph text with a link and .", processed)
        self.assertIn("Col A. Col B", processed)
        self.assertIn("Keep. This", processed)
        self.assertNotIn("print", processed)
        self.assertNotIn("diagram", processed)
        self.assertNotIn("x^2", processed)

    def test_normalize_scores_uses_midpoint_when_scores_are_equal(self):
        self.assertEqual(normalize_scores([0.0, 0.0, 0.0]), [0.5, 0.5, 0.5])

    def test_assign_score_bands_uses_adaptive_document_thresholds(self):
        bands = assign_score_bands([0.0, 0.12, 0.29, 0.41, 0.57, 0.74, 0.93])

        self.assertEqual(bands.count("high"), 2)
        self.assertGreaterEqual(bands.count("medium"), 2)

    def test_assign_score_bands_keeps_flat_documents_readable(self):
        self.assertEqual(assign_score_bands([0.5, 0.5, 0.5]), ["medium", "medium", "medium"])

    def test_window_packing_creates_overlapping_windows_for_long_documents(self):
        tokenizer = DummyTokenizer()
        sentences = [
            "alpha beta gamma delta epsilon",
            "one two three four five",
            "lorem ipsum dolor sit amet",
            "delta gamma beta alpha omega",
        ]

        windows = pack_sentence_windows(sentences, tokenizer, max_window_tokens=11, overlap_sentences=1)

        self.assertGreaterEqual(len(windows), 2)
        self.assertEqual(windows[0].start_index, 0)
        self.assertEqual(windows[1].start_index, 1)

    def test_sentence_token_spans_skip_special_tokens(self):
        spans = build_sentence_token_spans([2, 3], [1, 0, 0, 0, 0, 0, 1])

        self.assertEqual(spans, [TokenSpan(start=1, end=3), TokenSpan(start=3, end=6)])

    def test_sentence_token_spans_validate_available_tokens(self):
        with self.assertRaises(ValueError):
            build_sentence_token_spans([3], [1, 0, 1])


class AnalyzerBehaviorTestCase(unittest.TestCase):
    def test_empty_input_returns_no_sentences(self):
        analyzer = DummyAnalyzer()
        self.assertEqual(analyzer.analyze_markdown(""), [])

    def test_single_sentence_defaults_to_half_weight(self):
        analyzer = DummyAnalyzer()
        result = analyzer.analyze_markdown("Only one sentence.")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["weight"], 0.5)
        self.assertEqual(result[0]["band"], "medium")

    def test_duplicate_sentences_keep_distinct_indices(self):
        analyzer = DummyAnalyzer()
        result = analyzer.analyze_markdown("Repeat this. Repeat this. Something else.")

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["normalizedSentence"], result[1]["normalizedSentence"])
        self.assertNotEqual(result[0]["index"], result[1]["index"])


if __name__ == "__main__":
    unittest.main()
