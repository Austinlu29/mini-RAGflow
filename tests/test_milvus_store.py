"""使用内存 Milvus 客户端验证云端写入与余弦检索，不读取凭据或访问网络。"""

import copy
import json
import math
import unittest
from hashlib import sha256
from unittest.mock import patch

from pymilvus import DataType, MilvusClient

from document.element import Block
from document.langchain import block_to_document
from vectorstore.langchain import ChunkVectorStore, EvidenceRetriever
from vectorstore.milvus import MilvusStore


def chunk(text):
    return Block(text, "Paragraph", 2, 10, 20, 100, 200)


class FakeIterator:
    def __init__(self, rows):
        self.rows = rows
        self.position = 0
        self.closed = False
        self.failure = None

    def next(self):
        if self.failure is not None:
            raise self.failure
        batch = self.rows[self.position:self.position + 1]
        self.position += len(batch)
        return batch

    def close(self):
        self.closed = True


class FakeMilvusClient:
    create_schema = staticmethod(MilvusClient.create_schema)
    prepare_index_params = staticmethod(MilvusClient.prepare_index_params)

    def __init__(self):
        self.schema = None
        self.indexes = {}
        self.rows = {}
        self.iterators = []
        self.upsert_calls = []
        self.delete_calls = []
        self.search_calls = []
        self.iterator_calls = []
        self.create_calls = []
        self.closed = False
        self.fail_upsert_at = None
        self.fail_iterator = False

    def has_collection(self, **kwargs):
        return self.schema is not None

    def create_collection(self, **kwargs):
        self.create_calls.append(kwargs)
        self.schema = kwargs["schema"].to_dict()
        self.indexes = {item.index_name: item.to_dict() for item in kwargs["index_params"]}

    def describe_collection(self, **kwargs):
        return copy.deepcopy(self.schema)

    def list_indexes(self, **kwargs):
        return list(self.indexes)

    def describe_index(self, collection_name, index_name, **kwargs):
        return copy.deepcopy(self.indexes[index_name])

    def load_collection(self, **kwargs):
        pass

    def get_load_state(self, **kwargs):
        return {"state": "Loaded"}

    def _filtered(self, expression=""):
        rows = list(self.rows.values())
        if expression:
            field, value = expression.split("==", 1)
            assert field.strip() == "source", expression
            source = json.loads(value.strip())
            rows = [row for row in rows if row["source"] == source]
        return rows

    def query_iterator(self, **kwargs):
        self.iterator_calls.append(kwargs)
        iterator = FakeIterator([{"id": row["id"]} for row in self._filtered(kwargs.get("filter", ""))])
        if self.fail_iterator:
            iterator.failure = RuntimeError("iterator unavailable")
        self.iterators.append(iterator)
        return iterator

    def upsert(self, **kwargs):
        self.upsert_calls.append(copy.deepcopy(kwargs))
        if len(self.upsert_calls) == self.fail_upsert_at:
            raise RuntimeError("upsert unavailable")
        for row in kwargs["data"]:
            self.rows[row["id"]] = copy.deepcopy(row)
        return {"upsert_count": len(kwargs["data"])}

    def delete(self, **kwargs):
        self.delete_calls.append(kwargs)
        for row_id in kwargs["ids"]:
            self.rows.pop(row_id, None)
        return {"delete_count": len(kwargs["ids"])}

    def query(self, **kwargs):
        rows = self._filtered(kwargs.get("filter", ""))
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": len(rows)}]
        return copy.deepcopy(rows)

    def get_collection_stats(self, **kwargs):
        return {"row_count": len(self.rows)}

    def search(self, **kwargs):
        self.search_calls.append(copy.deepcopy(kwargs))
        query = kwargs["data"][0]
        hits = []
        for row in self._filtered(kwargs.get("filter", "")):
            cosine = sum(a * b for a, b in zip(query, row["vector"]))
            hits.append({"id": row["id"], "distance": cosine, "entity": copy.deepcopy(row)})
        return [sorted(hits, key=lambda hit: hit["distance"], reverse=True)[:kwargs["limit"]]]

    def close(self):
        self.closed = True


class MilvusStoreTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeMilvusClient()
        self.constructor = self.enterContext(patch("vectorstore.milvus.MilvusClient", return_value=self.client))
        # 即使构造函数主动加载 dotenv，测试也不会读取项目的真实密钥。
        self.enterContext(patch("vectorstore.milvus.load_dotenv"))
        self.constructor.create_schema = FakeMilvusClient.create_schema
        self.constructor.prepare_index_params = FakeMilvusClient.prepare_index_params
        self.store = self.open_store()

    def open_store(self, **kwargs):
        options = dict(embedding_model="test-model", uri="https://example.invalid", token="test-only-token",
                       collection_name="test_chunks", dimension=2, batch_size=2)
        options.update(kwargs)
        return MilvusStore(**options)

    def test_langchain_adapter_reads_and_replaces_existing_milvus_chunks(self):
        embeddings = unittest.mock.Mock()
        embeddings.embed_query.return_value = [1, 0]
        embeddings.embed_documents.return_value = [[1, 0]]
        original = self.store.add_chunks([chunk("old"), chunk("stale")], [[1, 0], [0, 1]], source="manual.pdf")
        adapter = ChunkVectorStore(store=self.store, embeddings=embeddings)
        retrieved = EvidenceRetriever(vectorstore=adapter).invoke("question")
        self.assertEqual([document.id for document in retrieved], original[:1])
        self.assertEqual(retrieved[0].metadata["page_number"], 2)
        updated = adapter.add_documents([block_to_document(chunk("new"), source="manual.pdf")])
        self.assertEqual(updated, original[:1])
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(adapter.similarity_search("question")[0].page_content, "new")

    def test_new_collection_has_explicit_cosine_schema_and_model(self):
        self.assertEqual(len(self.client.create_calls), 1)
        schema = self.client.schema
        self.assertFalse(schema["auto_id"])
        self.assertFalse(schema["enable_dynamic_field"])
        self.assertEqual(json.loads(schema["description"])["embedding_model"], "test-model")
        fields = {field["name"]: field for field in schema["fields"]}
        self.assertEqual(set(fields), {"id", "vector", "text", "source", "chunk_index", "metadata"})
        self.assertTrue(fields["id"]["is_primary"])
        self.assertEqual(fields["id"]["type"], DataType.VARCHAR)
        self.assertEqual(int(fields["id"]["params"]["max_length"]), 64)
        self.assertEqual(fields["vector"]["type"], DataType.FLOAT_VECTOR)
        self.assertEqual(int(fields["vector"]["params"]["dim"]), 2)
        self.assertEqual(fields["chunk_index"]["type"], DataType.INT64)
        self.assertEqual(fields["metadata"]["type"], DataType.JSON)
        self.assertEqual(self.client.indexes["vector_idx"]["metric_type"], "COSINE")
        self.assertEqual(self.client.indexes["vector_idx"]["index_type"], "AUTOINDEX")

    def test_existing_collection_reopens_without_recreation(self):
        self.open_store()
        self.assertEqual(len(self.client.create_calls), 1)

    def test_model_dimension_metric_and_schema_mismatch_are_rejected(self):
        original_schema = copy.deepcopy(self.client.schema)
        original_indexes = copy.deepcopy(self.client.indexes)
        mutations = (
            lambda: self.client.schema.update(description=json.dumps({"embedding_model": "other-model"})),
            lambda: self.client.schema.update(description="untracked collection"),
            lambda: next(field for field in self.client.schema["fields"] if field["name"] == "vector")
                ["params"].update(dim=3),
            lambda: self.client.indexes["vector_idx"].update(metric_type="L2"),
            lambda: self.client.schema.update(fields=[field for field in self.client.schema["fields"]
                                                     if field["name"] != "metadata"]),
            lambda: next(field for field in self.client.schema["fields"] if field["name"] == "chunk_index")
                .update(type=DataType.VARCHAR),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mismatch=index):
                self.client.schema = copy.deepcopy(original_schema)
                self.client.indexes = copy.deepcopy(original_indexes)
                mutate()
                with self.assertRaises(ValueError):
                    self.open_store()
        self.assertEqual(self.client.upsert_calls, [])

    def test_normalization_preserves_chunk_alignment_and_batches(self):
        ids = self.store.add_chunks((chunk(text) for text in ("first", "second", "third")),
                                    (vector for vector in ([3, 4], [0, 7], [-8, 0])), source="source.pdf")
        self.assertEqual([len(call["data"]) for call in self.client.upsert_calls], [2, 1])
        rows = [self.client.rows[row_id] for row_id in ids]
        self.assertEqual([row["text"] for row in rows], ["first", "second", "third"])
        self.assertEqual([row["vector"] for row in rows], [[0.6, 0.8], [0.0, 1.0], [-1.0, 0.0]])
        self.assertEqual([row["chunk_index"] for row in rows], [0, 1, 2])
        self.assertEqual(rows[0]["metadata"]["page_number"], 2)
        self.assertEqual(rows[0]["metadata"]["block_type"], "Paragraph")
        self.assertEqual(rows[0]["metadata"]["x0"], 10)
        self.assertTrue(all(iterator.closed for iterator in self.client.iterators))

    def test_stable_ids_reimport_and_source_filter_keep_other_documents(self):
        source = 'a "quoted" \\ path.pdf'
        first = self.store.add_chunks([chunk("old"), chunk("stale")], [[1, 0], [0, 1]], source=source)
        other = self.store.add_chunks([chunk("other")], [[0, 1]], source="b")
        updated = self.store.add_chunks([chunk("new")], [[1, 0]], source=source)
        expected = sha256(f"{source}\0{0}".encode("utf-8")).hexdigest()
        self.assertEqual(updated, first[:1])
        self.assertEqual(updated, [expected])
        self.assertEqual(set(self.client.rows), set(updated + other))
        self.assertEqual(self.store.count(), 2)
        self.assertEqual(self.store.count(source=source), 1)
        self.assertEqual(self.store.search([1, 0], source=source)[0]["text"], "new")
        self.assertEqual(self.store.search([0, 1], source="b")[0]["text"], "other")
        self.assertEqual(self.store.search([1, 0], source="missing"), [])
        self.assertTrue(all(iterator.closed for iterator in self.client.iterators))

    def test_cosine_score_is_similarity_and_distance_is_one_minus_score(self):
        ids = self.store.add_chunks([chunk("same"), chunk("orthogonal"), chunk("opposite")],
                                    [[3, 0], [0, 4], [-5, 0]], source="a")
        results = self.store.search([10, 0], top_k=9)
        self.assertEqual([result["id"] for result in results], ids)
        self.assertEqual([result["text"] for result in results], ["same", "orthogonal", "opposite"])
        for result, similarity in zip(results, (1.0, 0.0, -1.0)):
            self.assertAlmostEqual(result["similarity"], similarity)
            self.assertAlmostEqual(result["distance"], 1.0 - similarity)
            self.assertEqual(result["metadata"]["page_number"], 2)
        call = self.client.search_calls[-1]
        self.assertEqual(call["data"], [[1.0, 0.0]])
        self.assertEqual(call["search_params"]["metric_type"], "COSINE")

    def test_failed_upsert_does_not_delete_old_chunks(self):
        original = self.store.add_chunks([chunk("old"), chunk("stale")], [[1, 0], [0, 1]], source="a")
        self.client.fail_upsert_at = len(self.client.upsert_calls) + 1
        with self.assertRaises(RuntimeError):
            self.store.add_chunks([chunk("new")], [[1, 0]], source="a")
        self.assertEqual(set(self.client.rows), set(original))
        self.assertEqual(self.client.delete_calls, [])
        self.assertTrue(self.client.iterators[-1].closed)

    def test_failed_later_batch_does_not_delete_old_chunks(self):
        self.store.add_chunks([chunk(str(index)) for index in range(4)], [[1, 0]] * 4, source="a")
        self.client.fail_upsert_at = len(self.client.upsert_calls) + 2
        with self.assertRaises(RuntimeError):
            self.store.add_chunks([chunk("new") for _ in range(3)], [[1, 0]] * 3, source="a")
        self.assertEqual(self.client.delete_calls, [])
        self.assertEqual(len(self.client.rows), 4)

    def test_iterator_is_closed_on_read_failure_and_no_write_occurs(self):
        self.client.fail_iterator = True
        with self.assertRaises(RuntimeError):
            self.store.add_chunks([chunk("body")], [[1, 0]], source="a")
        self.assertTrue(self.client.iterators[-1].closed)
        self.assertEqual(self.client.upsert_calls, [])
        self.assertEqual(self.client.delete_calls, [])

    def test_empty_input_does_not_erase_existing_document(self):
        ids = self.store.add_chunks([chunk("body")], [[1, 0]], source="a")
        writes = len(self.client.upsert_calls)
        self.assertEqual(self.store.add_chunks([], [], source="a"), [])
        self.assertEqual(set(self.client.rows), set(ids))
        self.assertEqual(len(self.client.upsert_calls), writes)
        self.assertEqual(self.client.delete_calls, [])

    def test_invalid_chunks_and_vectors_do_not_query_or_write(self):
        cases = (([chunk("a")], []), ([chunk(" ")], [[1, 0]]), ([chunk("a")], [[0, 0]]),
                 ([chunk("a")], [[1, 0, 0]]), ([chunk("a")], [[float("nan"), 1]]),
                 ([chunk("a")], [[float("inf"), 1]]))
        for chunks, vectors in cases:
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                self.store.add_chunks(chunks, vectors, source="a")
        for source in ("", " "):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.store.add_chunks([chunk("a")], [[1, 0]], source=source)
        self.assertEqual(self.client.iterator_calls, [])
        self.assertEqual(self.client.upsert_calls, [])
        self.assertEqual(self.client.delete_calls, [])

    def test_search_rejects_invalid_size_and_vector(self):
        for top_k in (0, -1, True, 1.5):
            with self.subTest(top_k=top_k), self.assertRaises((TypeError, ValueError)):
                self.store.search([1, 0], top_k=top_k)
        for vector in ([0, 0], [1], [1, 0, 0], [math.nan, 1], [math.inf, 1]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                self.store.search(vector)
        self.assertEqual(self.client.search_calls, [])

    def test_close_releases_client(self):
        self.store.close()
        self.assertTrue(self.client.closed)


if __name__ == "__main__":
    unittest.main()
