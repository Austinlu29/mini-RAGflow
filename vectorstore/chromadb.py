"""持久化保存文档片段，并通过余弦距离检索。"""

from collections.abc import Iterable
from hashlib import sha256
from os import PathLike

import chromadb
from chromadb.config import Settings

from document.element import Block
from embedding.embed import normalize_vectors


class ChromaStore:
    def __init__(
        self,
        *,
        embedding_model: str,
        persist_directory: str | PathLike[str] = "data/chroma",
        collection_name: str = "documents",
    ):
        """同一集合只能使用同一 embedding 模型和 cosine 距离。"""
        if not embedding_model.strip():
            raise ValueError("embedding_model 不能为空")
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"embedding_model": embedding_model},
            configuration={"hnsw": {"space": "cosine"}},
        )
        if (self.collection.metadata or {}).get("embedding_model") != embedding_model:
            raise ValueError("集合使用了其他 embedding 模型，请更换集合名称")
        if self.collection.configuration["hnsw"]["space"] != "cosine":
            raise ValueError("现有集合不是 cosine 距离，请更换集合名称")

    def add_chunks(
        self,
        chunks: Iterable[Block],
        embeddings: Iterable[Iterable[float]],
        *,
        source: str,
    ) -> list[str]:
        """写入向量、文本和元数据，返回 ID；重复导入同一来源会更新片段。

        ID 由 source 和片段序号决定。成功写入后删除该来源的旧多余片段，
        不影响其他来源。空输入不改变集合；数据库批次写入不具备事务性。
        """
        chunks = list(chunks)
        vectors = normalize_vectors(embeddings)
        if len(chunks) != len(vectors):
            raise ValueError("片段数量与向量数量必须一致")
        if not source.strip():
            raise ValueError("source 不能为空")
        if any(not chunk.text.strip() for chunk in chunks):
            raise ValueError("不能存储空白片段")
        if not chunks:
            return []

        ids = [sha256(f"{source}\0{index}".encode("utf-8")).hexdigest()
               for index in range(len(chunks))]
        metadatas = [
            {"source": source, "chunk_index": index, "page_number": chunk.page_number,
             "block_type": chunk.block_type, "x0": chunk.x0, "y0": chunk.y0,
             "x1": chunk.x1, "y1": chunk.y1}
            for index, chunk in enumerate(chunks)
        ]
        previous_ids = self.collection.get(where={"source": source}, include=[])["ids"]
        batch_size = self.client.get_max_batch_size()
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=vectors[start:end],
                documents=[c.text for c in chunks[start:end]],
                metadatas=metadatas[start:end],
            )
        stale_ids = sorted(set(previous_ids) - set(ids))
        for start in range(0, len(stale_ids), batch_size):
            self.collection.delete(ids=stale_ids[start:start + batch_size])
        return ids

    def search(
        self,
        query_embedding: Iterable[float],
        *,
        top_k: int = 3,
        source: str | None = None,
    ) -> list[dict]:
        """返回文本、元数据、余弦距离和 similarity=1-distance，按相关性排序。"""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k 必须是整数")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        vector = normalize_vectors([query_embedding])[0]
        count = self.collection.count()
        if not count:
            return []
        options = {"where": {"source": source}} if source is not None else {}
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
            **options,
        )
        return [
            {"id": chunk_id, "text": text, "metadata": metadata,
             "distance": distance, "similarity": 1.0 - distance}
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0], result["documents"][0],
                result["metadatas"][0], result["distances"][0],
            )
        ]
