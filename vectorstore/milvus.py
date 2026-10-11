"""连接 Zilliz Cloud，保存文档片段并执行余弦检索。"""

from collections.abc import Iterable
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient

from document.element import Block
from embedding.embed import normalize_vectors


class MilvusStore:
    def __init__(
        self,
        *,
        embedding_model: str,
        uri: str | None = None,
        token: str | None = None,
        collection_name: str | None = None,
        dimension: int = 1536,
        batch_size: int = 100,
        timeout: float = 30.0,
    ):
        """读取项目 .env，连接云端并创建或校验同模型、同维度的集合。"""
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        uri = uri if uri is not None else os.getenv("MILVUS_URI", "")
        token = token if token is not None else os.getenv("MILVUS_TOKEN", "")
        parsed = urlsplit(uri)
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("MILVUS_URI 必须是控制台提供的 HTTPS 集群 Endpoint")
        if not token.strip():
            raise ValueError("请在 .env 或环境变量中配置 MILVUS_TOKEN")
        if not embedding_model.strip():
            raise ValueError("embedding_model 不能为空")
        for name, value in (("dimension", dimension), ("batch_size", batch_size)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} 必须是整数")
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout 必须是有限正数")
        self.collection_name = (collection_name if collection_name is not None
                                else os.getenv("MILVUS_COLLECTION", "miniragflow_chunks"))
        if not self.collection_name.strip():
            raise ValueError("collection_name 不能为空")
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.batch_size = batch_size
        self.client = MilvusClient(uri=uri, token=token, timeout=timeout)
        try:
            self._ensure_collection()
        except Exception:
            self.client.close()
            raise

    def _ensure_collection(self):
        if not self.client.has_collection(collection_name=self.collection_name):
            schema = self.client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
                description=json.dumps({"embedding_model": self.embedding_model}, ensure_ascii=False),
            )
            schema.add_field(field_name="id", datatype=DataType.VARCHAR,
                             is_primary=True, max_length=64)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=4096)
            schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
            schema.add_field(field_name="metadata", datatype=DataType.JSON)
            indexes = self.client.prepare_index_params()
            indexes.add_index(field_name="vector", index_name="vector_idx",
                              index_type="AUTOINDEX", metric_type="COSINE")
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=indexes,
                consistency_level="Strong",
            )
        info = self.client.describe_collection(collection_name=self.collection_name)
        try:
            model = json.loads(info.get("description", ""))["embedding_model"]
        except (ValueError, TypeError, KeyError):
            raise ValueError("现有集合缺少 embedding 模型信息，请使用新的集合名称") from None
        if model != self.embedding_model:
            raise ValueError("现有集合使用其他 embedding 模型，请使用新的集合名称")
        fields = {field["name"]: field for field in info["fields"]}
        required = {"id": DataType.VARCHAR, "vector": DataType.FLOAT_VECTOR,
                    "text": DataType.VARCHAR, "source": DataType.VARCHAR,
                    "chunk_index": DataType.INT64, "metadata": DataType.JSON}
        if any(fields.get(name, {}).get("type") != dtype for name, dtype in required.items()):
            raise ValueError("现有集合的字段结构与文档片段不兼容")
        if (info.get("auto_id", False) or not fields["id"].get("is_primary")
                or int(fields["id"].get("params", {}).get("max_length", 0)) < 64):
            raise ValueError("集合必须使用手动生成的字符串主键")
        if int(fields["vector"].get("params", {}).get("dim", 0)) != self.dimension:
            raise ValueError("现有集合的向量维度与 embedding 模型不一致")
        index = self.client.describe_index(collection_name=self.collection_name, index_name="vector_idx")
        if index.get("field_name") != "vector" or index.get("metric_type") != "COSINE":
            raise ValueError("现有集合必须使用 vector 字段的 COSINE 索引")

    def _vectors(self, embeddings):
        vectors = normalize_vectors(embeddings)
        if any(len(vector) != self.dimension for vector in vectors):
            raise ValueError(f"向量必须是 {self.dimension} 维")
        return vectors

    @staticmethod
    def _source_filter(source: str) -> str:
        # JSON 字符串转义保留路径中的反斜杠、引号，不直接拼接裸值。
        return "source == " + json.dumps(source, ensure_ascii=False)

    def _source_ids(self, source: str) -> set[str]:
        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            filter=self._source_filter(source),
            output_fields=["id"],
            batch_size=1000,
            consistency_level="Strong",
        )
        ids = set()
        try:
            while batch := iterator.next():
                ids.update(row["id"] for row in batch)
        finally:
            iterator.close()
        return ids

    def add_chunks(
        self,
        chunks: Iterable[Block],
        embeddings: Iterable[Iterable[float]],
        *,
        source: str,
    ) -> list[str]:
        """保存向量与来源信息；同路径和序号使用固定 ID，不重复增加记录。

        全部 upsert 成功后删除该来源的旧多余片段，不影响其他来源。
        空输入不改变集合；网络分批写入不具备事务性。
        """
        chunks = list(chunks)
        vectors = self._vectors(embeddings)
        if len(chunks) != len(vectors):
            raise ValueError("片段数量与向量数量必须一致")
        if not source.strip() or len(source.encode("utf-8")) > 4096:
            raise ValueError("source 不能为空，且 UTF-8 长度不能超过 4096 字节")
        if any(not c.text.strip() or len(c.text.encode("utf-8")) > 65535 for c in chunks):
            raise ValueError("片段不能为空，且 UTF-8 长度不能超过 65535 字节")
        if not chunks:
            return []
        ids = [sha256(f"{source}\0{index}".encode("utf-8")).hexdigest()
               for index in range(len(chunks))]
        records = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            metadata = {"source": source, "chunk_index": index, "page_number": chunk.page_number,
                        "block_type": chunk.block_type, "x0": chunk.x0, "y0": chunk.y0,
                        "x1": chunk.x1, "y1": chunk.y1}
            records.append({"id": ids[index], "vector": vector, "text": chunk.text,
                            "source": source, "chunk_index": index, "metadata": metadata})
        previous_ids = self._source_ids(source)
        for start in range(0, len(records), self.batch_size):
            self.client.upsert(collection_name=self.collection_name,
                               data=records[start:start + self.batch_size])
        stale_ids = sorted(previous_ids - set(ids))
        for start in range(0, len(stale_ids), self.batch_size):
            self.client.delete(collection_name=self.collection_name,
                               ids=stale_ids[start:start + self.batch_size])
        return ids

    def search(
        self,
        query_embedding: Iterable[float],
        *,
        top_k: int = 3,
        source: str | None = None,
    ) -> list[dict]:
        """返回与 ChromaStore 相同的结果结构；COSINE 分数越大越相似。"""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k 必须是整数")
        if not 1 <= top_k <= 16384:
            raise ValueError("top_k 必须在 1 到 16384 之间")
        vector = self._vectors([query_embedding])[0]
        result = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field="vector",
            limit=top_k,
            filter=self._source_filter(source) if source is not None else "",
            output_fields=["text", "metadata"],
            search_params={"metric_type": "COSINE"},
            consistency_level="Strong",
        )
        return [
            {"id": hit["id"], "text": hit["entity"]["text"],
             "metadata": hit["entity"]["metadata"],
             "similarity": float(hit["distance"]), "distance": 1.0 - float(hit["distance"])}
            for hit in result[0]
        ]

    def count(self, *, source: str | None = None) -> int:
        """使用强一致性读取可见记录数，避免后台统计滞后。"""
        result = self.client.query(
            collection_name=self.collection_name,
            filter=self._source_filter(source) if source is not None else "",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        return int(result[0]["count(*)"])

    def close(self):
        self.client.close()
