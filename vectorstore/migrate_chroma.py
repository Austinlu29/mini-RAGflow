"""将现有 Chroma 片段与向量迁移到云端，不重新调用 embedding API。"""

import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import chromadb
from chromadb.config import Settings

from document.element import Block
from .milvus import MilvusStore


def migrate_chroma(
    *,
    persist_directory: str | Path | None = None,
    chroma_collection: str = "documents",
    milvus_collection: str | None = None,
) -> int:
    path = (Path(persist_directory) if persist_directory is not None
            else Path(__file__).resolve().parents[1] / "data" / "chroma")
    if not (path / "chroma.sqlite3").is_file():
        raise FileNotFoundError("没有找到现有 Chroma 数据库")
    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(name=chroma_collection, embedding_function=None)
    model = (collection.metadata or {}).get("embedding_model")
    if not model:
        raise ValueError("Chroma 集合缺少 embedding 模型信息")
    data = collection.get(include=["documents", "embeddings", "metadatas"])
    if not data["ids"]:
        return 0
    groups = defaultdict(list)
    for chunk_id, text, vector, metadata in zip(
        data["ids"], data["documents"], data["embeddings"], data["metadatas"]
    ):
        if metadata is None or not metadata.get("source") or "chunk_index" not in metadata:
            raise ValueError("Chroma 片段缺少来源或序号")
        groups[metadata["source"]].append((chunk_id, text, vector, metadata))
    # 先检查 ID 和序号，确保沿用原有去重规则，再进行云端写入。
    for source, rows in groups.items():
        rows.sort(key=lambda row: row[3]["chunk_index"])
        for index, (chunk_id, _, _, metadata) in enumerate(rows):
            expected_id = sha256(f"{source}\0{index}".encode("utf-8")).hexdigest()
            if metadata["chunk_index"] != index or chunk_id != expected_id:
                raise ValueError("Chroma 主键或片段序号与当前去重规则不一致")
    store = MilvusStore(embedding_model=model, collection_name=milvus_collection)
    migrated = 0
    try:
        for source, rows in groups.items():
            chunks = [Block(text=text, block_type=metadata["block_type"],
                            page_number=metadata["page_number"], x0=metadata["x0"],
                            y0=metadata["y0"], x1=metadata["x1"], y1=metadata["y1"])
                      for _, text, _, metadata in rows]
            ids = store.add_chunks(chunks, [row[2] for row in rows], source=source)
            if store.count(source=source) != len(ids):
                raise RuntimeError("云端可见片段数量与迁移数据不一致")
            migrated += len(ids)
    finally:
        store.close()
    return migrated


def main():
    parser = argparse.ArgumentParser(description="迁移 Chroma 向量到云端 Milvus")
    parser.add_argument("--persist-directory", type=Path, default=None)
    parser.add_argument("--chroma-collection", default="documents")
    parser.add_argument("--milvus-collection", default=None)
    args = parser.parse_args()
    count = migrate_chroma(persist_directory=args.persist_directory,
                           chroma_collection=args.chroma_collection,
                           milvus_collection=args.milvus_collection)
    print(f"已迁移 {count} 个片段到云端 Milvus，未重新调用 embedding API")


if __name__ == "__main__":
    main()
