"""用模拟 API 验证向量对齐、归一化与入库流程。"""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from document.element import Block
from embedding.embed import DashScopeEmbedder, embed_chunks, normalize_vectors


def vector(x, y):
    return [x, y] + [0.0] * 1534


def chunk(text):
    return Block(text, "Paragraph", 2, 10, 20, 100, 200)


class EmbeddingTests(unittest.TestCase):
    def model(self):
        return DashScopeEmbedder(api_key="test-only-key", api_host="https://dashscope.aliyuncs.com")

    def test_batching_response_order_and_normalization(self):
        batches = []

        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "text-embedding-v2")
            self.assertEqual(payload["parameters"]["text_type"], "document")
            texts = payload["input"]["texts"]
            batches.append(texts)
            items = [{"text_index": i, "embedding": vector(int(text) + 1, 1)}
                     for i, text in enumerate(texts)]
            return httpx.Response(200, json={"output": {"embeddings": list(reversed(items))}})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("embedding.embed.httpx.Client", return_value=client):
            actual = embed_chunks((chunk(str(i)) for i in range(27)), embedder=self.model())
        self.assertEqual([len(batch) for batch in batches], [25, 2])
        expected = normalize_vectors(vector(i + 1, 1) for i in range(27))
        self.assertEqual(actual, expected)

    def test_query_uses_query_type(self):
        def handler(request):
            self.assertEqual(json.loads(request.content)["parameters"]["text_type"], "query")
            return httpx.Response(200, json={"output": {"embeddings": [
                {"text_index": 0, "embedding": vector(3, 4)}]}})
        with patch("embedding.embed.httpx.Client", return_value=httpx.Client(transport=httpx.MockTransport(handler))):
            self.assertEqual(self.model().embed_query("question")[:2], [0.6, 0.8])

    def test_invalid_response_is_rejected(self):
        for items in ([], [{"text_index": 1, "embedding": vector(1, 0)}],
                      [{"text_index": 0, "embedding": [1, 0]}],
                      [{"text_index": 0, "embedding": vector(0, 0)}]):
            with self.subTest(items=items[:1]):
                client = httpx.Client(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json={"output": {"embeddings": items}})))
                with patch("embedding.embed.httpx.Client", return_value=client), self.assertRaises(ValueError):
                    self.model().embed_documents(["text"])

    def test_http_failure_is_not_silently_ignored(self):
        client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(401, text="sensitive-response")))
        with patch("embedding.embed.httpx.Client", return_value=client), self.assertRaises(RuntimeError) as error:
            self.model().embed_query("text")
        self.assertNotIn("sensitive-response", str(error.exception))

    def test_empty_invalid_input_and_vectors(self):
        with patch("embedding.embed.httpx.Client") as client:
            self.assertEqual(self.model().embed_documents([]), [])
            self.assertEqual(embed_chunks([]), [])
            client.assert_not_called()
        for value in ([" "], [1], "text"):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                self.model().embed_documents(value)
        for vectors in ([[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], [[1], [1, 2]]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                normalize_vectors(vectors)
        for size in (0, 26, True, 1.5):
            with self.subTest(size=size), self.assertRaises((TypeError, ValueError)):
                DashScopeEmbedder(api_key="test", batch_size=size)


class PipelineTests(unittest.TestCase):
    def test_pdf_chunks_reach_embedding_and_store(self):
        from RAG.pipeline import index_pdf
        model = unittest.mock.Mock()
        model.embedding_id = "test-model"
        model.embed_documents.return_value = [[1, 0], [0, 1]]
        with patch("document.langchain.pdf_parser", return_value=[]) as parser, \
                patch("document.langchain.LayoutParser") as layout, \
                patch("RAG.pipeline.create_vector_store") as store:
            layout.return_value.parse.return_value = [chunk("123456789")]
            store.return_value.add_chunks.return_value = ["first", "second"]
            ids = index_pdf("test.pdf", embedder=model, chunk_size=6, chunk_overlap=2)
        self.assertEqual(ids, ["first", "second"])
        model.embed_documents.assert_called_once_with(["123456", "56789"])
        self.assertEqual(parser.call_args.args[0], Path("test.pdf").resolve())
        self.assertEqual(store.return_value.add_chunks.call_args.kwargs["source"], str(Path("test.pdf").resolve()))

    def test_api_failure_does_not_open_database(self):
        from RAG.pipeline import index_pdf
        model = unittest.mock.Mock()
        model.embed_documents.side_effect = RuntimeError("API unavailable")
        with patch("document.langchain.pdf_parser", return_value=[]), \
                patch("document.langchain.LayoutParser") as layout, \
                patch("RAG.pipeline.create_vector_store") as store:
            layout.return_value.parse.return_value = [chunk("body")]
            with self.assertRaises(RuntimeError):
                index_pdf("test.pdf", embedder=model)
            store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
