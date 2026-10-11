"""使用项目现有 Embedding 和向量库运行 CRUD-RAG 30 问纯向量检索。"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import struct
import time

import httpx

from document.element import Block
from document.splitter import split_document
from embedding.embed import DashScopeEmbedder, normalize_vectors
from vectorstore.factory import create_vector_store

from .crud_dataset import DEFAULT_DATASET, digest, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "evaluation" / "crud30"


def retry_embedding(operation):
    for attempt in range(4):
        try:
            return operation()
        except (httpx.TransportError, RuntimeError) as error:
            transient = isinstance(error, httpx.TransportError) or any(
                f"HTTP {status}" in str(error) for status in (429, 500, 502, 503, 504))
            if not transient or attempt == 3:
                raise
            print(f"Embedding transient failure; retry {attempt + 1}/3", flush=True)
            time.sleep(2 ** attempt)


class EmbeddingCache:
    """缓存以模型、document/query 类型和文本摘要隔离，逐批提交便于失败后续跑。"""

    def __init__(self, path, model):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.model = model
        self.connection.execute("CREATE TABLE IF NOT EXISTS embeddings "
                                "(model TEXT, kind TEXT, text_hash TEXT, vector BLOB, "
                                "PRIMARY KEY(model, kind, text_hash))")

    def get(self, text, kind):
        row = self.connection.execute(
            "SELECT vector FROM embeddings WHERE model=? AND kind=? AND text_hash=?",
            (self.model.embedding_id, kind, digest(text))).fetchone()
        if row is None:
            return None
        if len(row[0]) != 1536 * 8:
            raise ValueError("Embedding 缓存维度损坏")
        return normalize_vectors([struct.unpack("<1536d", row[0])])[0]

    def put(self, text, kind, vector):
        values = normalize_vectors([vector])[0]
        if len(values) != 1536:
            raise ValueError("Embedding 必须为 1536 维")
        self.connection.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                                (self.model.embedding_id, kind, digest(text),
                                 struct.pack("<1536d", *values)))

    def documents(self, texts):
        result = []
        for start in range(0, len(texts), self.model.batch_size):
            batch = texts[start:start + self.model.batch_size]
            vectors = [self.get(text, "document") for text in batch]
            missing = [index for index, vector in enumerate(vectors) if vector is None]
            if missing:
                fresh = retry_embedding(lambda: self.model.embed_documents([batch[i] for i in missing]))
                if len(fresh) != len(missing):
                    raise ValueError("Embedding 数量与输入不一致")
                for index, vector in zip(missing, fresh):
                    self.put(batch[index], "document", vector)
                    vectors[index] = vector
                self.connection.commit()
            result.extend(vectors)
            completed = min(start + len(batch), len(texts))
            if start == 0 or completed % 250 == 0 or completed == len(texts):
                print(f"Embedded/cache {completed}/{len(texts)} chunks", flush=True)
        return result

    def query(self, text):
        vector = self.get(text, "query")
        if vector is not None:
            return vector, True
        vector = retry_embedding(lambda: self.model.embed_query(text))
        self.put(text, "query", vector)
        self.connection.commit()
        return vector, False

    def close(self):
        self.connection.close()


def make_chunks(documents, *, chunk_size=300, chunk_overlap=50):
    chunks = []
    mapping = []
    for ordinal, document in enumerate(documents, 1):
        # 文本新闻没有 PDF 页码/坐标，page_number 在本评测专用集合表示文档序号。
        block = Block(document["text"], "Paragraph", ordinal, 0, 0, 0, 0)
        parts = split_document([block], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks.extend(parts)
        mapping.extend({"document_id": document["id"], "document_ordinal": ordinal,
                        "document_chunk_index": index} for index in range(len(parts)))
    return chunks, mapping


def score_ranking(retrieved_document_ids, relevant_document_ids, ks=(1, 3, 5, 10)):
    """K 指返回的 chunk 数；同一参考文档多次出现只计一次 recall。"""
    gold = set(relevant_document_ids)
    if not gold:
        raise ValueError("参考文档不能为空")
    scores = {}
    for k in ks:
        if k <= 0:
            raise ValueError("K 必须大于 0")
        found = set(retrieved_document_ids[:k]) & gold
        scores[f"hit@{k}"] = float(bool(found))
        scores[f"document_recall@{k}"] = len(found) / len(gold)
        scores[f"all_evidence@{k}"] = float(found == gold)
    cutoff = max(ks)
    first = next((rank for rank, doc_id in enumerate(retrieved_document_ids[:cutoff], 1)
                  if doc_id in gold), None)
    scores[f"mrr@{cutoff}"] = 1 / first if first else 0.0
    return scores


def aggregate(rows):
    if not rows:
        raise ValueError("不能汇总空评测")
    names = rows[0]["metrics"]
    return {name: sum(row["metrics"][name] for row in rows) / len(rows) for name in names}


def render_report(report):
    config = report["configuration"]
    metrics = report["metrics"]
    ks = config["ks"]
    lines = ["# CRUD-RAG 30 问向量检索实测", "",
             f"- 时间（UTC）：{report['created_at']}",
             f"- 模型：{config['embedding_model']}",
             f"- 后端：{config['backend']}；独立集合：`{config['collection_name']}`",
             f"- 语料：{config['document_count']} 篇新闻 / {config['chunk_count']} 个片段；"
             f"{config['evidence_document_count']} 篇参考新闻，其余为不同事件的干扰新闻。",
             f"- 问题：30 问，单篇/两篇/三篇新闻任务各 10 问；seed={config['seed']}。",
             f"- 切片：{config['chunk_size']} 字符，重叠 {config['chunk_overlap']} 字符，复用项目 split_document。",
             f"- 数据 SHA256：`{config['dataset_sha256']}`", "",
             "## 汇总", "",
             "| K（片段数） | Hit@K | 参考文档 Recall@K | 全部参考文档命中率 |",
             "| --- | --- | --- | --- |"]
    for k in ks:
        lines.append(f"| {k} | {metrics[f'hit@{k}']:.2%} | "
                     f"{metrics[f'document_recall@{k}']:.2%} | {metrics[f'all_evidence@{k}']:.2%} |")
    cutoff = max(ks)
    lines.extend(["", f"MRR@{cutoff}：{metrics[f'mrr@{cutoff}']:.4f}", "",
                  "## 分任务", "",
                  f"| 任务 | 问数 | Hit@{cutoff} | 文档 Recall@{cutoff} | 全部证据@{cutoff} |",
                  "| --- | --- | --- | --- | --- |"])
    for task, group in report["by_task"].items():
        m = group["metrics"]
        lines.append(f"| {task} | {group['count']} | {m[f'hit@{cutoff}']:.2%} | "
                     f"{m[f'document_recall@{cutoff}']:.2%} | {m[f'all_evidence@{cutoff}']:.2%} |")
    if report.get("execution_note"):
        lines.extend(["", f"执行说明：{report['execution_note']}"])
    lines.extend(["", "## 指标口径与限制", "",
                  "- 仅测试现有纯向量检索，不使用 BM25、reranker 或生成式问答。",
                  "- 官方原始问题基于真实新闻构建，非真实用户日志；问题、答案、thoughts/event 均未入库。",
                  "- 标签取上游 news 字段，为文档级银标；命中参考文档中的任意片段视为命中文档，"
                  "不等于该片段已含完整答案。未标注的其他相关文档可能造成低估。",
                  "- K 是返回片段数，Recall 对命中文档去重并逐问题宏平均；"
                  "Hit 只要求一篇，全部证据命中率要求所有参考文档。",
                  f"- 相同正文的参考新闻去重；此子集共 {config['evidence_document_count']} 篇独立参考新闻。",
                  f"- 这是固定 {config['document_count']} 篇新闻的本地子集基线，不可与官方约 8 万篇语料的完整成绩直接比较。",
                  "- 参考答案仅用于人工核验，未自动评估生成答案准确率。",
                  "", "## 30 个问题及检索结果", ""])
    for row in report["results"]:
        m = row["metrics"]
        lines.extend([f"### {row['id']} · {row['task']}", "", row["question"], "",
                      f"参考答案：{row['reference_answer']}", "",
                      f"原始记录：{row['event_id']} / index={row['upstream_record_index']}。",
                      f"Recall@{cutoff}={m[f'document_recall@{cutoff}']:.2%}；"
                      f"相关片段排名：{row['relevant_chunk_ranks']}；"
                      f"遗漏文档：{row['missing_document_ids']}。", "",
                      "| 排名 | 文档 | similarity | 参考文档 | 片段预览 |",
                      "| --- | --- | --- | --- | --- |"])
        for hit in row["retrieved"]:
            preview = hit["text"][:160].replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {hit['rank']} | {hit['document_id']} | {hit['similarity']:.4f} | "
                         f"{'是' if hit['relevant_document'] else '否'} | {preview} |")
        lines.append("")
    return "\n".join(lines)


def run(dataset, *, backend=None, output=DEFAULT_OUTPUT, chunk_size=300,
        chunk_overlap=50, ks=(1, 3, 5, 10)):
    from dotenv import load_dotenv
    import os

    load_dotenv(PROJECT_ROOT / ".env")
    backend = backend or os.getenv("VECTOR_STORE", "milvus").lower()
    chunks, mapping = make_chunks(dataset["documents"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    model = DashScopeEmbedder()
    fingerprint = digest(json.dumps([dataset["dataset_sha256"], chunk_size, chunk_overlap,
                                     model.embedding_id]))[:20]
    collection_name = f"crud30_{fingerprint}"
    source = f"crud-rag:{fingerprint}"
    print(f"Corpus: {len(dataset['documents'])} documents / {len(chunks)} chunks; "
          f"backend={backend}; collection={collection_name}", flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(PROJECT_ROOT / "data/crud_rag/cache/embeddings.sqlite3", model)
    store = None
    started = time.perf_counter()
    try:
        vectors = cache.documents([chunk.text for chunk in chunks])
        store = create_vector_store(embedding_model=model.embedding_id, backend=backend,
                                    collection_name=collection_name,
                                    persist_directory=PROJECT_ROOT / "data/crud_rag/chroma")
        ids = store.add_chunks(chunks, vectors, source=source)
        index = {chunk_id: info for chunk_id, info in zip(ids, mapping)}
        count = store.count() if hasattr(store, "count") else store.collection.count()
        if len(ids) != len(chunks) or count != len(chunks):
            raise ValueError("评测集合片段数量不匹配")
        print(f"Indexed {count} chunks; querying 30 questions", flush=True)
        rows = []
        for question in dataset["questions"]:
            query_started = time.perf_counter()
            vector, cached = cache.query(question["question"])
            search_started = time.perf_counter()
            results = store.search(vector, top_k=max(ks))
            search_seconds = time.perf_counter() - search_started
            hits = []
            gold = set(question["relevant_document_ids"])
            for rank, result in enumerate(results, 1):
                if result["id"] not in index:
                    raise ValueError("检索返回非本次语料的片段")
                info = index[result["id"]]
                hits.append({"rank": rank, "chunk_id": result["id"], **info,
                             "similarity": result["similarity"], "text": result["text"],
                             "relevant_document": info["document_id"] in gold})
            scores = score_ranking([hit["document_id"] for hit in hits], gold, ks)
            found = {hit["document_id"] for hit in hits} & gold
            row = {**question, "metrics": scores, "retrieved": hits,
                   "relevant_chunk_ranks": [hit["rank"] for hit in hits if hit["relevant_document"]],
                   "missing_document_ids": sorted(gold - found),
                   "query_embedding_cached": cached, "search_seconds": search_seconds,
                   "query_total_seconds": time.perf_counter() - query_started}
            rows.append(row)
            print(f"{question['id']} hit@{max(ks)}={scores[f'hit@{max(ks)}']:.0f} "
                  f"recall={scores[f'document_recall@{max(ks)}']:.3f}", flush=True)
        provenance = dataset["provenance"]
        report = {"created_at": datetime.now(timezone.utc).isoformat(),
                  "configuration": {"backend": backend, "embedding_model": model.model,
                                    "embedding_id": model.embedding_id, "collection_name": collection_name,
                                    "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                                    "ks": list(ks), "document_count": len(dataset["documents"]),
                                    "chunk_count": len(chunks), "question_count": len(rows),
                                    "seed": provenance["seed"],
                                    "evidence_document_count": provenance["evidence_document_count"],
                                    "dataset_sha256": dataset["dataset_sha256"], "provenance": provenance},
                  "elapsed_seconds": time.perf_counter() - started,
                  "metrics": aggregate(rows), "by_task": {}, "results": rows}
        for task in Counter(row["task"] for row in rows):
            group = [row for row in rows if row["task"] == task]
            report["by_task"][task] = {"count": len(group), "metrics": aggregate(group)}
        (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "report.md").write_text(render_report(report), encoding="utf-8")
        (output / "chunk_manifest.json").write_text(
            json.dumps({"source": source, "chunks": index}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2), flush=True)
        print(f"Report: {output / 'report.md'}", flush=True)
        return report
    finally:
        cache.close()
        if store is not None and hasattr(store, "close"):
            store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--backend", choices=("milvus", "chroma"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    args = parser.parse_args()
    if not args.ks or min(args.ks) <= 0 or max(args.ks) > 16384:
        parser.error("ks 必须在 1 到 16384 之间")
    run(load_dataset(args.dataset), backend=args.backend, output=args.output,
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, ks=tuple(sorted(set(args.ks))))


if __name__ == "__main__":
    main()
