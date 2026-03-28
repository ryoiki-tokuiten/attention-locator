from __future__ import annotations

import unittest

from backend.analysis import BaseSemanticMarkdownAnalyzer, assign_score_bands, normalize_scores


class DummyAnalyzer(BaseSemanticMarkdownAnalyzer):
    def compute_raw_scores(self, sentences):
        return [float(index) for index, _ in enumerate(sentences)]


class AnalysisHelpersTestCase(unittest.TestCase):
    def test_normalize_scores_uses_midpoint_when_scores_are_equal(self):
        self.assertEqual(normalize_scores([0.0, 0.0, 0.0]), [0.5, 0.5, 0.5])

    def test_normalize_scores_stabilizes_non_finite_values(self):
        self.assertEqual(normalize_scores([float("nan"), 5.0, float("inf")]), [0.0, 1.0, 0.0])

    def test_assign_score_bands_uses_adaptive_document_thresholds(self):
        bands = assign_score_bands([0.0, 0.12, 0.29, 0.41, 0.57, 0.74, 0.93])

        self.assertEqual(bands.count("high"), 2)
        self.assertGreaterEqual(bands.count("medium"), 2)

    def test_assign_score_bands_keeps_flat_documents_readable(self):
        self.assertEqual(assign_score_bands([0.5, 0.5, 0.5]), ["medium", "medium", "medium"])


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
        self.assertIsNone(result[0]["sourceStart"])
        self.assertIsNone(result[0]["sourceEnd"])

    def test_duplicate_sentences_keep_distinct_indices(self):
        analyzer = DummyAnalyzer()
        result = analyzer.analyze_markdown("Repeat this. Repeat this. Something else.")

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["normalizedSentence"], result[1]["normalizedSentence"])
        self.assertNotEqual(result[0]["index"], result[1]["index"])

    def test_markdown_is_not_precleaned_before_sentence_scoring(self):
        analyzer = DummyAnalyzer()
        result = analyzer.analyze_markdown("Heading with `code`. Another sentence.")

        self.assertEqual(len(result), 2)
        self.assertIn("`code`", result[0]["sentence"])


if __name__ == "__main__":
    unittest.main()
