"""将布局解析接入 LangChain，保留页码、坐标和不可拆分的表格。"""

from collections.abc import Iterable, Iterator, Sequence
from os import PathLike
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import BaseDocumentTransformer, Document

from .element import Block
from .layout_parser import LayoutParser
from .pdf_parser import pdf_parser
from .splitter import split_document
from .table_detector import TableDetector


def block_to_document(block: Block, *, source: str) -> Document:
    return Document(page_content=block.text, metadata={
        "source": source, "page_number": block.page_number,
        "block_type": block.block_type,
        "x0": block.x0, "y0": block.y0, "x1": block.x1, "y1": block.y1,
    })


def document_to_block(document: Document) -> Block:
    metadata = document.metadata
    return Block(document.page_content, metadata.get("block_type", "Paragraph"),
                 metadata["page_number"], *(metadata.get(key, 0) for key in ("x0", "y0", "x1", "y1")))


class LayoutPDFLoader(BaseLoader):
    """输出每个布局块一个 Document；source 为 PDF 的绝对路径。"""

    def __init__(self, file_path: str | PathLike[str]):
        self.path = Path(file_path).resolve()

    def lazy_load(self) -> Iterator[Document]:
        elements = TableDetector().detect(pdf_parser(self.path))
        for block in LayoutParser().parse(elements):
            yield block_to_document(block, source=str(self.path))


class LayoutDocumentSplitter(BaseDocumentTransformer):
    """正文按句界切片，标题和表格保持完整，额外元数据也随片段保留。"""

    def __init__(self, *, chunk_size: int = 300, chunk_overlap: int = 50):
        # 复用同一参数校验，避免两套切分规则产生不同的入库结果。
        split_document([], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        result = []
        for document in documents:
            blocks = split_document([document_to_block(document)],
                                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            result.extend(Document(page_content=block.text, metadata=dict(document.metadata))
                          for block in blocks)
        return result

    def transform_documents(self, documents: Sequence[Document], **kwargs) -> Sequence[Document]:
        return self.split_documents(documents)

    async def atransform_documents(self, documents: Sequence[Document], **kwargs) -> Sequence[Document]:
        from asyncio import to_thread
        return await to_thread(self.transform_documents, documents, **kwargs)
