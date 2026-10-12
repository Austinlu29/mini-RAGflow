"""LangChain 存储/检索接口，复用既有 Milvus 集合和稳定 ID。"""

from collections.abc import Iterable
from math import isfinite

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict, Field

from document.langchain import document_to_block


class ChunkVectorStore(VectorStore):
    """同一 source 的 add_documents 表示完整替换该来源的片段。"""

    def __init__(self, *, store, embeddings: Embeddings):
        self.store = store
        self._embeddings = embeddings

    @property
    def embeddings(self):
        return self._embeddings

    def add_documents(self, documents: list[Document], **kwargs) -> list[str]:
        if kwargs:
            raise ValueError("片段 ID 由来源和序号生成，不支持额外写入参数")
        documents = list(documents)
        if not documents:
            return []
        self._validate_documents(documents)
        vectors = self.embeddings.embed_documents([document.page_content for document in documents])
        return self.add_embeddings(documents, vectors)

    @staticmethod
    def _validate_documents(documents):
        sources = [document.metadata.get("source") for document in documents]
        if any(not isinstance(source, str) or not source.strip() for source in sources):
            raise ValueError("每个 Document 必须包含非空 source")
        if len(set(sources)) != 1:
            raise ValueError("每次写入必须提供同一来源的完整片段")
        if any(not document.page_content.strip()
               or type(document.metadata.get("page_number")) is not int
               or document.metadata["page_number"] < 1 for document in documents):
            raise ValueError("每个 Document 必须包含非空文本和正整数页码")

    def add_embeddings(self, documents, vectors) -> list[str]:
        """写入已经完成的向量；用于先向量化、后建立数据库连接的入库链。"""
        documents = list(documents)
        if not documents:
            return []
        self._validate_documents(documents)
        blocks = [document_to_block(document) for document in documents]
        return self.store.add_chunks(blocks, vectors, source=documents[0].metadata["source"])

    def add_texts(self, texts: Iterable[str], metadatas=None, **kwargs) -> list[str]:
        texts = list(texts)
        if metadatas is None or len(metadatas) != len(texts):
            raise ValueError("必须为每段文本提供来源和页码元数据")
        return self.add_documents([Document(page_content=text, metadata=metadata)
                                   for text, metadata in zip(texts, metadatas)], **kwargs)

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, *, store, **kwargs):
        vectorstore = cls(store=store, embeddings=embedding)
        vectorstore.add_texts(texts, metadatas=metadatas, **kwargs)
        return vectorstore

    def similarity_search_with_cosine_scores(self, query: str, k: int = 4, *, source=None, **kwargs):
        """返回原始 COSINE 分数（-1 到 1），越大越相似。"""
        if kwargs:
            raise ValueError("只支持 k 和 source 检索参数")
        options = {"source": source} if source is not None else {}
        hits = self.store.search(self.embeddings.embed_query(query), top_k=k, **options)
        result = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            text, metadata, score = hit.get("text"), hit.get("metadata"), hit.get("similarity")
            if (not isinstance(text, str) or not isinstance(metadata, dict)
                    or type(score) not in (int, float) or not isfinite(score) or not -1 <= score <= 1):
                continue
            document = Document(id=hit.get("id"), page_content=text, metadata=dict(metadata))
            result.append((document, score))
        return result

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        """LangChain 距离接口：返回 1-cosine（0 到 2），越小越相似。"""
        return [(document, 1.0 - score) for document, score in
                self.similarity_search_with_cosine_scores(query, k=k, **kwargs)]

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> list[Document]:
        return [document for document, _ in self.similarity_search_with_score(query, k=k, **kwargs)]

    def _select_relevance_score_fn(self):
        # LangChain 的 relevance score 要求为 0..1，包括反向量。
        return lambda distance: 1.0 - distance / 2.0


class EvidenceRetriever(BaseRetriever):
    """只有相似度合格且带来源/页码的文本才能进入证据链。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    vectorstore: ChunkVectorStore
    top_k: int = Field(default=5, ge=1, le=50, strict=True)
    min_similarity: float = Field(default=0.65, ge=0, le=1, allow_inf_nan=False)
    source: str | None = None

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        result = []
        for document, score in self.vectorstore.similarity_search_with_cosine_scores(
                query, k=self.top_k, source=self.source):
            source, page = document.metadata.get("source"), document.metadata.get("page_number")
            if (not self.min_similarity <= score <= 1 or not document.page_content.strip()
                    or not isinstance(source, str) or not source.strip()
                    or type(page) is not int or page < 1):
                continue
            result.append(document)
        return result
