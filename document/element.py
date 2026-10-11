from dataclasses import dataclass
from typing import Optional

@dataclass
class Element:
    text: str
    element_type: str
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float

    font_size: Optional[float] = None

    # PDF 原始文本块，辅助段落边界判断和换行标题合并。
    source_block: Optional[int] = None
    # 原生表格的结构信息。未提供时仍可使用 TableDetector 的坐标推断。
    table_id: Optional[int] = None
    table_row: Optional[int] = None
    table_column: Optional[int] = None


@dataclass
class Block:
    text: str
    block_type: str
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float
