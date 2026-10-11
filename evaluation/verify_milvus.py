"""通过 HTTPS 只读验证云 Milvus 中的 CRUD-RAG 评测语料。"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import httpx

from embedding.embed import DashScopeEmbedder
from .crud_dataset import DEFAULT_DATASET, digest, load_dataset
from .retrieval import PROJECT_ROOT, make_chunks


def verify_rows(rows, chunks, source):
    expected = {sha256(f"{source}\0{index}".encode("utf-8")).hexdigest(): index
                for index in range(len(chunks))}
    actual = {row["id"]: row for row in rows}
    if len(actual) != len(rows) or set(actual) != set(expected):
        raise ValueError("云端片段 ID 与本地语料不一致")
    for chunk_id, index in expected.items():
        row = actual[chunk_id]
        # REST V2 有些部署将 JSON 字段返回为 JSON 字符串；SDK 返回 dict。
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if (row["text"] != chunks[index].text or row["source"] != source
                or row["chunk_index"] != index
                or metadata["page_number"] != chunks[index].page_number):
            raise ValueError(f"云端第 {index} 个片段的文本或来源信息不一致")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "output/evaluation/crud30_milvus/verification.json")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    dataset = load_dataset(args.dataset)
    model = DashScopeEmbedder()
    chunks, _ = make_chunks(dataset["documents"], chunk_size=args.chunk_size,
                          chunk_overlap=args.chunk_overlap)
    fingerprint = digest(json.dumps([dataset["dataset_sha256"], args.chunk_size,
                                     args.chunk_overlap, model.embedding_id]))[:20]
    collection = f"crud30_{fingerprint}"
    source = f"crud-rag:{fingerprint}"
    uri = os.environ["MILVUS_URI"].rstrip("/")
    with httpx.Client(timeout=httpx.Timeout(60, connect=15),
                      headers={"Authorization": "Bearer " + os.environ["MILVUS_TOKEN"]}) as client:
        def post(operation, body):
            for attempt in range(4):
                try:
                    response = client.post(uri + "/v2/vectordb/" + operation, json=body)
                    break
                except httpx.TransportError:
                    if attempt == 3:
                        raise
                    print(f"Cloud verification connection retry {attempt + 1}/3", flush=True)
                    time.sleep(2 ** attempt)
            if response.status_code != 200:
                raise RuntimeError(f"Milvus HTTPS HTTP {response.status_code}")
            payload = response.json()
            if payload.get("code") != 0:
                raise RuntimeError(f"Milvus HTTPS API code {payload.get('code')}")
            return payload["data"]

        info = post("collections/describe", {"collectionName": collection})
        fields = {field["name"]: field for field in info["fields"]}
        dimension = int(next(param["value"] for param in fields["vector"]["params"] if param["key"] == "dim"))
        metric = next(index["metricType"] for index in info["indexes"] if index["fieldName"] == "vector")
        if (dimension != 1536 or metric != "COSINE"
                or json.loads(info["description"])["embedding_model"] != model.embedding_id):
            raise ValueError("云端向量空间与本地模型不一致")
        count_result = post("entities/query", {"collectionName": collection, "filter": "",
                                              "outputFields": ["count(*)"], "consistencyLevel": "Strong"})
        count = count_result[0]["count(*)"]
        if count != len(chunks):
            raise ValueError(f"云端当前 {count} 条，本地预期 {len(chunks)} 条")
        rows = post("entities/query", {"collectionName": collection,
                                       "filter": "source == " + json.dumps(source),
                                       "outputFields": ["id", "source", "chunk_index", "text", "metadata"],
                                       "limit": len(chunks), "consistencyLevel": "Strong"})
        verify_rows(rows, chunks, source)
    report = {"verified_at": datetime.now(timezone.utc).isoformat(), "backend": "cloud_milvus",
              "verification_transport": "https_rest_v2", "collection_name": collection,
              "dataset_sha256": dataset["dataset_sha256"], "document_count": len(dataset["documents"]),
              "expected_chunk_count": len(chunks), "cloud_chunk_count": count,
              "verified_id_text_source_count": len(rows), "dimension": dimension,
              "metric": metric, "status": "verified"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
