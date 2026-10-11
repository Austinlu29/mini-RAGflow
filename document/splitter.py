"""将布局解析后的文本块切分为适合检索的文档片段。"""

from collections.abc import Iterable
from dataclasses import replace

from .element import Block


def split_document(
    blocks: Iterable[Block],
    *,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Block]:
    """按字符数切分正文，保留输入顺序、类型、页码和来源块坐标。

    优先在后半段的段落、换行或句末切分，找不到边界时按长度截断。
    相邻正文片段重叠 chunk_overlap 个字符；不同块之间不重叠、不合并。
    Title、Heading 和 Table 保持完整，允许超过 chunk_size，避免破坏
    标题和表格结构。空白块忽略，输入块不会被修改。

    坐标指向原始 Block，不能精确表示切片在页面上的位置。
    """
    for name, value in (("chunk_size", chunk_size), ("chunk_overlap", chunk_overlap)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} 必须是整数")
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap 必须满足 0 <= chunk_overlap < chunk_size")

    chunks = []
    for block in blocks:
        if not block.text.strip():
            continue
        if block.block_type in {"Title", "Heading", "Table"}:
            chunks.append(replace(block))
            continue
        for text in _split_text(block.text, chunk_size, chunk_overlap):
            if text.strip():
                chunks.append(replace(block, text=text))
    return chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # 边界必须越过重叠区，保证每次切分都向前推进。
            boundary_start = start + max((chunk_size + 1) // 2, chunk_overlap + 1)
            for separators in (("\n\n",), ("\n",), ("。", "！", "？", ". ", "! ", "? ")):
                boundaries = [
                    position + len(separator)
                    for separator in separators
                    if (position := text.rfind(
                        separator, max(start, boundary_start - len(separator)), end
                    )) >= 0
                ]
                if boundaries:
                    end = max(boundaries)
                    break
        yield text[start:end]
        if end == len(text):
            break
        start = end - chunk_overlap
