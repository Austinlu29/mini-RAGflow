"""LangChain PDF 入库链：加载 → 布局切片 → 向量化 → 写入。"""

from os import PathLike
from langchain_core.runnables import RunnableBranch, RunnableLambda

from document.langchain import LayoutDocumentSplitter, LayoutPDFLoader
from embedding.embed import DashScopeEmbedder
from vectorstore.factory import create_vector_store
from vectorstore.langchain import ChunkVectorStore


def create_indexing_chain(
    *, embedder=None, persist_directory="data/chroma", collection_name=None,
    backend=None, chunk_size=300, chunk_overlap=50,
):
    """返回 Runnable；invoke 接收 PDF 路径，输出稳定片段 ID。"""
    splitter = LayoutDocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def embed(documents):
        model = embedder if embedder is not None else DashScopeEmbedder()
        vectors = model.embed_documents([document.page_content for document in documents])
        if len(vectors) != len(documents):
            raise ValueError("片段数量与向量数量必须一致")
        return {"documents": documents, "vectors": vectors, "model": model}

    def persist(state):
        model = state["model"]
        store = create_vector_store(embedding_model=model.embedding_id, backend=backend,
                                    persist_directory=persist_directory, collection_name=collection_name)
        try:
            vectorstore = ChunkVectorStore(store=store, embeddings=model)
            return vectorstore.add_embeddings(state["documents"], state["vectors"])
        finally:
            if hasattr(store, "close"):
                store.close()

    return (
        RunnableLambda(lambda path: LayoutPDFLoader(path).load(), name="load_pdf")
        | RunnableLambda(splitter.split_documents, name="split_layout")
        | RunnableBranch(
            (lambda documents: not documents, RunnableLambda(lambda documents: [])),
            RunnableLambda(embed, name="embed_documents") | RunnableLambda(persist, name="persist_chunks"),
        )
    )


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
    return create_indexing_chain(embedder=embedder, persist_directory=persist_directory,
                                 collection_name=collection_name, backend=backend,
                                 chunk_size=chunk_size, chunk_overlap=chunk_overlap).invoke(file_path)
