from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.analysis import BaseSemanticMarkdownAnalyzer
from backend.main import create_app


class StubAnalyzer(BaseSemanticMarkdownAnalyzer):
    model_name = "stub-model"
    device = "cpu"

    def compute_raw_scores(self, sentences):
        return [float(index + 1) for index, _ in enumerate(sentences)]


class ApiTestCase(unittest.TestCase):
    def test_health_reports_ready_when_stub_is_injected(self):
        app = create_app(analyzer=StubAnalyzer())

        with TestClient(app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ready"])
        self.assertEqual(response.json()["modelName"], "stub-model")

    def test_analyze_returns_sentence_payloads(self):
        app = create_app(analyzer=StubAnalyzer())

        with TestClient(app) as client:
            response = client.post("/analyze", json={"text": "First sentence. Second sentence."})

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["index"], 0)
        self.assertIn("normalizedSentence", payload[0])


if __name__ == "__main__":
    unittest.main()
