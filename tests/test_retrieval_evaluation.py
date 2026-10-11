"""验证多证据指标、数据隔离、缓存以及真实本地向量库评测链路。"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.crud_dataset import DEFAULT_DATASET, TASKS, build_dataset, digest, load_dataset, validate_dataset
from evaluation.retrieval import EmbeddingCache, make_chunks, run, score_ranking
from vectorstore.factory import create_vector_store


class RetrievalMetricTests(unittest.TestCase):
    def test_duplicate_chunks_do_not_inflate_multi_document_recall(self):
        scores = score_ranking(["a", "a", "irrelevant", "b"], ["a", "b"], (1, 3, 5))
        self.assertEqual(scores["hit@1"], 1)
        self.assertEqual(scores["document_recall@3"], 0.5)
        self.assertEqual(scores["all_evidence@3"], 0)
        self.assertEqual(scores["all_evidence@5"], 1)
        self.assertEqual(scores["mrr@5"], 1)

    def test_mrr_obeys_rank_and_cutoff_and_misses_are_zero(self):
        scores = score_ranking(["x", "y", "a"], ["a"], (1, 3))
        self.assertEqual(scores["hit@1"], 0)
        self.assertEqual(scores["mrr@3"], 1 / 3)
        self.assertEqual(score_ranking(["x", "y", "a"], ["a"], (1, 2))["mrr@2"], 0)
        with self.assertRaises(ValueError):
            score_ranking([], [])


class DatasetTests(unittest.TestCase):
    def test_deterministic_stratification_and_answer_leakage_isolation(self):
        raw = {}
        for n, task in enumerate(TASKS, 1):
            raw[task] = []
            for i in range(20):
                record = {"ID": f"event-{n}-{i}", "questions": f"QUESTION-{n}-{i}",
                          "answers": f"ANSWER-MARKER-{n}-{i}", "thoughts": "THOUGHT-MARKER"}
                for j in range(1, n + 1):
                    record[f"news{j}"] = f"新闻 {n}-{i}-{j} " + "真实新闻正文。" * 30
                raw[task].append(record)
        dataset = build_dataset(raw, corpus_size=100)
        self.assertEqual(dataset, build_dataset(raw, corpus_size=100))
        validate_dataset(dataset)
        self.assertEqual(len({q["event_id"] for q in dataset["questions"]}), 30)
        for task in TASKS:
            self.assertEqual(sum(q["task"] == task for q in dataset["questions"]), 10)
        selected_events = {q["event_id"] for q in dataset["questions"]}
        evidence = {doc_id for q in dataset["questions"] for doc_id in q["relevant_document_ids"]}
        for document in dataset["documents"]:
            self.assertNotIn("ANSWER-MARKER", document["text"])
            self.assertNotIn("QUESTION-", document["text"])
            self.assertNotIn("THOUGHT-MARKER", document["text"])
            if document["id"] not in evidence:
                self.assertTrue(all(origin["event_id"] not in selected_events
                                    for origin in document["origins"]))
        dataset["documents"][0]["text"] += "tampered"
        with self.assertRaises(ValueError):
            validate_dataset(dataset)

    @unittest.skipUnless(DEFAULT_DATASET.is_file(),
                         "先运行 evaluation.download_crud 和 evaluation.crud_dataset 生成本地评测数据")
    def test_local_subset_preserves_official_questions_and_evidence(self):
        dataset = load_dataset()
        self.assertEqual(len(dataset["documents"]), 1000)
        chunks, mapping = make_chunks(dataset["documents"])
        self.assertEqual(len(chunks), len(mapping))
        self.assertEqual({entry["document_id"] for entry in mapping},
                         {doc["id"] for doc in dataset["documents"]})
        raw_path = Path(__file__).resolve().parents[1] / "data/crud_rag/upstream/data/crud_split/split_merged.json"
        if raw_path.exists():
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            docs = {doc["id"]: doc for doc in dataset["documents"]}
            for question in dataset["questions"]:
                record = raw[question["task"]][question["upstream_record_index"]]
                self.assertEqual(question["question"], record["questions"])
                self.assertEqual(question["reference_answer"], record["answers"])
                for doc_id in question["relevant_document_ids"]:
                    self.assertIn(docs[doc_id]["text"], [record.get(f"news{i}") for i in (1, 2, 3)])


class FakeEmbedder:
    embedding_id = "unit-test:one-hot"
    model = "unit-test-only"
    batch_size = 25

    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    @staticmethod
    def vector(text):
        values = [0.0] * 1536
        values[int(text.split()[1])] = 1.0
        return values

    def embed_documents(self, texts):
        self.document_calls += 1
        return [self.vector(text) for text in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return self.vector(text)


class CacheAndIntegrationTests(unittest.TestCase):
    def test_cache_keeps_document_and_query_separate_and_reuses_vectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = FakeEmbedder()
            cache = EmbeddingCache(Path(temporary) / "cache.sqlite3", model)
            try:
                cache.documents(["topic 1"])
                _, cached = cache.query("topic 1")
                self.assertFalse(cached)
                cache.documents(["topic 1"])
                self.assertTrue(cache.query("topic 1")[1])
                self.assertEqual(model.document_calls, 1)
                self.assertEqual(model.query_calls, 1)
            finally:
                cache.close()

    def test_real_chroma_ranking_mapping_and_reports(self):
        # 这里的假向量只测试指标/映射链路，用户报告由真实 DashScope/Milvus 单独生成。
        documents = [{"id": f"doc{i}", "text": f"topic {i}"} for i in range(30)]
        questions = [{"id": f"Q{i + 1:02d}", "task": TASKS[i // 10],
                      "event_id": f"event{i}", "upstream_record_index": i,
                      "question": f"topic {i}", "reference_answer": f"answer{i}",
                      "relevant_document_ids": [f"doc{i}"]} for i in range(30)]
        dataset = {"dataset_sha256": digest("test fixture"), "documents": documents,
                   "questions": questions, "provenance": {"seed": 1, "evidence_document_count": 30}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stores = []

            def factory(**options):
                store = create_vector_store(**options)
                stores.append(store)
                return store

            try:
                with patch("evaluation.retrieval.PROJECT_ROOT", root), \
                        patch("evaluation.retrieval.DashScopeEmbedder", FakeEmbedder), \
                        patch("evaluation.retrieval.create_vector_store", factory), \
                        contextlib.redirect_stdout(io.StringIO()):
                    report = run(dataset, backend="chroma", output=root / "output")
            finally:
                for store in stores:
                    store.client._system.stop()
            self.assertEqual(report["metrics"]["hit@1"], 1)
            self.assertEqual(len(report["results"]), 30)
            self.assertEqual(report["results"][12]["retrieved"][0]["document_id"], "doc12")
            self.assertTrue((root / "output/report.md").exists())
            self.assertEqual(json.loads((root / "output/results.json").read_text(encoding="utf-8"))
                             ["configuration"]["embedding_model"], "unit-test-only")


if __name__ == "__main__":
    unittest.main()
