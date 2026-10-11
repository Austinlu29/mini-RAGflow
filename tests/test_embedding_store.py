"""用模拟 API 和真实 ChromaDB 验证向量对齐、持久化与余弦排序。"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb
import httpx

from document.element import Block
from embedding.embed import DashScopeEmbedder, embed_chunks, normalize_vectors
from vectorstore.chromadb import ChromaStore


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


class ChromaStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ChromaStore(embedding_model="test-model", persist_directory=self.temp.name)
        self.addCleanup(self.store.client._system.stop)

    def test_real_cosine_ranking_metadata_and_persistence(self):
        ids = self.store.add_chunks([chunk("same"), chunk("orthogonal"), chunk("opposite")],
                                    [[3, 0], [0, 4], [-5, 0]], source="source.pdf")
        results = self.store.search([10, 0], top_k=9)
        self.assertEqual([r["text"] for r in results], ["same", "orthogonal", "opposite"])
        for result, expected in zip(results, (1.0, 0.0, -1.0)):
            self.assertAlmostEqual(result["similarity"], expected, places=5)
        self.assertEqual(results[0]["id"], ids[0])
        self.assertEqual(results[0]["metadata"]["page_number"], 2)
        self.assertEqual(results[0]["metadata"]["x0"], 10)
        self.assertTrue((Path(self.temp.name) / "chroma.sqlite3").is_file())
        # 在独立进程中重开数据库，确认数据实际落盘。
        script = ("import sys; from vectorstore.chromadb import ChromaStore; "
                  "s=ChromaStore(embedding_model='test-model',persist_directory=sys.argv[1]); "
                  "print(s.collection.count())")
        result = subprocess.run([sys.executable, "-c", script, self.temp.name],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "3")

    def test_reimport_removes_stale_chunks_without_affecting_other_sources(self):
        original = self.store.add_chunks([chunk("old"), chunk("stale")], [[1, 0], [0, 1]], source="a")
        self.store.add_chunks([chunk("other")], [[0, 1]], source="b")
        updated = self.store.add_chunks([chunk("new")], [[1, 0]], source="a")
        self.assertEqual(updated, original[:1])
        self.assertEqual(self.store.collection.count(), 2)
        self.assertEqual(self.store.search([1, 0], source="a")[0]["text"], "new")
        self.assertEqual(self.store.search([0, 1], source="b")[0]["text"], "other")
        self.assertEqual(self.store.search([1, 0], source="missing"), [])

    def test_empty_and_invalid_vectors_do_not_write(self):
        self.assertEqual(self.store.add_chunks([], [], source="a"), [])
        self.assertEqual(self.store.search([1, 0]), [])
        for chunks, vectors in (([chunk("a")], []), ([chunk(" ")], [[1, 0]]),
                                ([chunk("a")], [[0, 0]])):
            with self.subTest(chunks=chunks), self.assertRaises(ValueError):
                self.store.add_chunks(chunks, vectors, source="a")
        self.assertEqual(self.store.collection.count(), 0)
        for top_k in (0, -1, True, 1.5):
            with self.subTest(top_k=top_k), self.assertRaises((TypeError, ValueError)):
                self.store.search([1, 0], top_k=top_k)

    def test_model_and_distance_mismatch_are_rejected(self):
        with self.assertRaises(ValueError):
            ChromaStore(embedding_model="different-model", persist_directory=self.temp.name)
        self.store.client.create_collection(name="wrong-distance", embedding_function=None,
                                            metadata={"embedding_model": "test-model"})
        with self.assertRaises(ValueError):
            ChromaStore(embedding_model="test-model", persist_directory=self.temp.name,
                        collection_name="wrong-distance")


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
