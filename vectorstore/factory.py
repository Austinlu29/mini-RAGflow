"""按配置选择云端 Milvus 或本地 Chroma 存储。"""

import os
from os import PathLike
from pathlib import Path

from dotenv import load_dotenv


def create_vector_store(
    *,
    embedding_model: str,
    backend: str | None = None,
    collection_name: str | None = None,
    persist_directory: str | PathLike[str] = "data/chroma",
):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    backend = (backend if backend is not None else os.getenv("VECTOR_STORE", "milvus")).lower()
    if backend == "milvus":
        from .milvus import MilvusStore
        return MilvusStore(embedding_model=embedding_model, collection_name=collection_name)
    if backend == "chroma":
        from .chromadb import ChromaStore
        return ChromaStore(embedding_model=embedding_model,
                           collection_name=collection_name or "documents",
                           persist_directory=persist_directory)
    raise ValueError("VECTOR_STORE 必须是 milvus 或 chroma")
