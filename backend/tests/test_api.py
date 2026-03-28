from __future__ import annotations

import unittest

from httpx import ASGITransport, AsyncClient

from backend.analysis import BaseSemanticMarkdownAnalyzer
from backend.main import create_app


class StubAnalyzer(BaseSemanticMarkdownAnalyzer):
    model_name = "stub-model"
    device = "cpu"
    max_input_tokens = 128

    def compute_raw_scores(self, sentences):
        return [float(index + 1) for index, _ in enumerate(sentences)]

    def analyze_document(self, markdown_text: str):
        sentence_weights = self.analyze_markdown(markdown_text)
        return {
            "analysisText": markdown_text,
            "tokenCount": 6,
            "sentenceCount": len(sentence_weights),
            "sentenceWeights": sentence_weights,
            "sentenceAttention": [[1.0 if row == column else 0.5 for column in range(len(sentence_weights))] for row in range(len(sentence_weights))],
            "warnings": [],
        }

    def analyze_sentence_pair(self, markdown_text: str, sentence_x_index: int, sentence_y_index: int):
        return {
            "sentenceXIndex": sentence_x_index,
            "sentenceYIndex": sentence_y_index,
            "sentenceX": "First sentence.",
            "sentenceY": "Second sentence.",
            "columnTokens": ["First", "sentence"],
            "rowTokens": ["Second", "sentence"],
            "attention": [[0.1, 0.2], [0.3, 0.4]],
            "warnings": [],
        }


class ApiTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_health_reports_ready_when_stub_is_injected(self):
        app = create_app(analyzer=StubAnalyzer())

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ready"])
        self.assertEqual(response.json()["modelName"], "stub-model")
        self.assertEqual(response.json()["maxInputTokens"], 128)

    async def test_analyze_returns_sentence_payloads(self):
        app = create_app(analyzer=StubAnalyzer())

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post("/analyze", json={"text": "First sentence. Second sentence."})

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["index"], 0)
        self.assertIn("normalizedSentence", payload[0])

    async def test_attention_document_returns_sentence_matrix_payload(self):
        app = create_app(analyzer=StubAnalyzer())

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post("/attention/document", json={"text": "First sentence. Second sentence."})

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["sentenceCount"], 2)
        self.assertEqual(len(payload["sentenceAttention"]), 2)
        self.assertEqual(len(payload["sentenceWeights"]), 2)

    async def test_attention_pair_returns_token_matrix_payload(self):
        app = create_app(analyzer=StubAnalyzer())

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post(
                "/attention/pair",
                json={"text": "First sentence. Second sentence.", "sentenceXIndex": 0, "sentenceYIndex": 1},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["sentenceXIndex"], 0)
        self.assertEqual(payload["sentenceYIndex"], 1)
        self.assertEqual(payload["attention"][1][1], 0.4)


if __name__ == "__main__":
    unittest.main()
