"""按配置创建云端 Milvus 存储。"""

import os
from pathlib import Path

from dotenv import load_dotenv


def create_vector_store(
    *,
    embedding_model: str,
    backend: str | None = None,
    collection_name: str | None = None,
):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    backend = (backend if backend is not None else os.getenv("VECTOR_STORE", "milvus")).lower()
    if backend == "milvus":
        from .milvus import MilvusStore
        return MilvusStore(embedding_model=embedding_model, collection_name=collection_name)
    raise ValueError("VECTOR_STORE 仅支持 milvus")
