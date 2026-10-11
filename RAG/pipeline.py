"""PDF 解析、切片、向量化与入库的调用链。"""

from os import PathLike
from pathlib import Path

from document.layout_parser import LayoutParser
from document.pdf_parser import pdf_parser
from document.splitter import split_document
from document.table_detector import TableDetector
from embedding.embed import DashScopeEmbedder, embed_chunks
from vectorstore.factory import create_vector_store


def index_pdf(
    file_path: str | PathLike[str],
    *,
    embedder: DashScopeEmbedder | None = None,
    persist_directory: str | PathLike[str] = "data/chroma",
    collection_name: str | None = None,
    backend: str | None = None,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[str]:
    """将 PDF 的切片保存到配置的向量库，返回片段 ID。

    先完成所有片段的 API 向量化，再写入数据库。API 出错时不会改动
    集合。source 使用文件绝对路径，重复调用会更新同一文件的片段。
    """
    path = Path(file_path).resolve()
    elements = TableDetector().detect(pdf_parser(path))
    blocks = LayoutParser().parse(elements)
    chunks = split_document(blocks, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return []
    model = embedder if embedder is not None else DashScopeEmbedder()
    vectors = embed_chunks(chunks, embedder=model)
    store = create_vector_store(
        embedding_model=model.embedding_id,
        backend=backend,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    try:
        return store.add_chunks(chunks, vectors, source=str(path))
    finally:
        if hasattr(store, "close"):
            store.close()
