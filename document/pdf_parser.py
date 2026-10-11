"""提取 PDF 文本行和原生表格，供布局解析器使用。"""

import logging
from os import PathLike

import pymupdf

from .element import Element

logger = logging.getLogger(__name__)


def pdf_parser(
    file_path: str | PathLike[str],
    *,
    password: str | None = None,
    start_page: int = 1,
    end_page: int | None = None,
    detect_tables: bool = True,
) -> list[Element]:
    """返回文本行和表格单元格，页码从 1 开始，页码范围包含两端。

    默认使用 PyMuPDF 的线框表格检测；关闭 detect_tables 可保留原始
    文本行，再交给 TableDetector 进行坐标推断。扫描图片不会自动 OCR。
    文件、密码和页码错误直接抛出异常，所有路径都保证关闭文档。
    """
    if isinstance(start_page, bool) or not isinstance(start_page, int):
        raise TypeError("start_page 必须是整数")
    if start_page < 1:
        raise ValueError("start_page 必须从 1 开始")
    if end_page is not None:
        if isinstance(end_page, bool) or not isinstance(end_page, int):
            raise TypeError("end_page 必须是整数或 None")
        if end_page < start_page:
            raise ValueError("end_page 不能小于 start_page")

    elements = []
    with pymupdf.open(file_path) as doc:
        if doc.needs_pass and not doc.authenticate(password or ""):
            raise ValueError("PDF 已加密，请提供正确的 password")
        if not doc.is_pdf:
            raise ValueError("输入文件必须是 PDF")
        last_page = doc.page_count if end_page is None else end_page
        if start_page > doc.page_count or last_page > doc.page_count:
            raise ValueError(f"页码范围超出 PDF 总页数 {doc.page_count}")

        for page_index in range(start_page - 1, last_page):
            page = doc[page_index]
            page_number = page_index + 1
            table_elements, table_regions = (
                _extract_tables(page, page_number) if detect_tables else ([], [])
            )
            # dict 默认包含图片数据；这里只需要文本，避免读取大幅图片字节。
            flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
            blocks = page.get_text("dict", flags=flags).get("blocks", [])
            for block_index, block in enumerate(blocks):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(span.get("text", "") for span in spans).strip()
                    if not text:
                        continue
                    bbox = line["bbox"]
                    if any(_inside_table(bbox, region) for region in table_regions):
                        continue
                    sizes = [
                        span["size"]
                        for span in spans
                        if span.get("text", "").strip() and span.get("size", 0) > 0
                    ]
                    elements.append(
                        Element(
                            text=text,
                            element_type="Text",
                            page_number=page_number,
                            x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
                            font_size=max(sizes, default=None),
                            source_block=block_index,
                        )
                    )
            elements.extend(table_elements)

    return sorted(elements, key=lambda e: (e.page_number, e.y0, e.x0))


def _extract_tables(page, page_number: int) -> tuple[list[Element], list[tuple]]:
    """在页面存活期间复制单元格结构，不保留依赖页面生命周期的对象。"""
    if not hasattr(page, "find_tables"):
        return [], []
    try:
        tables = page.find_tables().tables
    except (RuntimeError, ValueError) as error:
        logger.warning("第 %s 页表格检测失败，改为提取文本：%s", page_number, error)
        return [], []

    elements = []
    regions = []
    for table_id, table in enumerate(tables):
        data = table.extract()
        if not data:
            continue
        regions.append(tuple(table.bbox))
        for row_index, row in enumerate(table.rows):
            for column_index, bbox in enumerate(row.cells):
                # 合并单元格的缺失位置仍保留空值，防止列发生偏移。
                cell_bbox = bbox if bbox is not None else row.bbox
                text = data[row_index][column_index] or ""
                elements.append(
                    Element(
                        text=text.strip(),
                        element_type="TableText",
                        page_number=page_number,
                        x0=cell_bbox[0], y0=cell_bbox[1],
                        x1=cell_bbox[2], y1=cell_bbox[3],
                        table_id=table_id,
                        table_row=row_index,
                        table_column=column_index,
                    )
                )
    return elements, regions


def _inside_table(bbox, region) -> bool:
    # 字体上下沿可能略超出单元格，使用文本行中心点判断。
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]
