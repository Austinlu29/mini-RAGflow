"""将文本行和表格单元格恢复为适合 RAG 使用的文本块。"""

from collections import Counter
from dataclasses import replace
from math import isfinite

from .element import Element, Block


class LayoutParser:
    def __init__(
        self,
        *,
        title_font_size: float | None = None,
        heading_font_size: float | None = None,
    ):
        """默认按正文字号识别标题；也可指定绝对字号阈值。"""
        for size in (title_font_size, heading_font_size):
            if size is not None and (not isfinite(size) or size <= 0):
                raise ValueError("标题字号阈值必须是正数")
        if (
            title_font_size is not None
            and heading_font_size is not None
            and title_font_size < heading_font_size
        ):
            raise ValueError("title_font_size 不能小于 heading_font_size")
        self.title_font_size = title_font_size
        self.heading_font_size = heading_font_size

    def parse(self, elements) -> list[Block]:
        """按页面恢复阅读顺序并构建文本块；不修改输入元素。"""
        # 复制输入，避免分类修改调用方的数据；原生表格空单元格需要保留。
        elements = [
            replace(element)
            for element in elements
            if element.text.strip() or element.element_type == "TableText"
        ]
        elements = self.sort_elements(elements)
        elements = self.classify_elements(elements)
        return self.merge_elements(elements)

    def sort_elements(self, elements):
        # 表格作为整体排序，避免旁边的正文插入单元格序列。
        tables = {}
        units = []
        for element in elements:
            if element.element_type == "TableText" and element.table_id is not None:
                key = (element.page_number, element.table_id)
                if key not in tables:
                    tables[key] = []
                    units.append(tables[key])
                tables[key].append(element)
            else:
                units.append([element])
        units.sort(key=lambda unit: (
            unit[0].page_number,
            min(e.y0 for e in unit),
            min(e.x0 for e in unit),
        ))
        return [
            element
            for unit in units
            for element in sorted(unit, key=lambda e: (e.y0, e.x0))
        ]

    def classify_elements(self, elements):
        elements = list(elements)
        # 按字符数投票，避免短小页眉或大量小字号单元格影响正文字号。
        sizes = Counter()
        for element in elements:
            if element.element_type == "Text" and self.valid_font_size(element):
                sizes[round(element.font_size, 1)] += len(element.text.strip())
        body_size = sizes.most_common(1)[0][0] if sizes else 12.0
        title_size = (
            self.title_font_size if self.title_font_size is not None else body_size * 1.6
        )
        heading_size = (
            self.heading_font_size if self.heading_font_size is not None else body_size * 1.2
        )
        for element in elements:
            # 已由上游标记的标题、表格等保持原类型。
            if element.element_type != "Text" or not self.valid_font_size(element):
                continue
            if element.font_size >= title_size:
                element.element_type = "Title"
            elif element.font_size >= heading_size:
                element.element_type = "Heading"

            else:
                element.element_type = "Text"

        return elements

    @staticmethod
    def valid_font_size(element):
        return (
            element.font_size is not None
            and isfinite(element.font_size)
            and element.font_size > 0
        )

    def merge_elements(self, elements):
        """合并同段正文、同原始块的换行标题，以及同一张表格的单元格。"""
        elements = list(elements)
        blocks = []
        index = 0
        while index < len(elements):
            element = elements[index]
            if element.element_type == "TableText":
                table_elements = [element]
                index += 1
                while (
                    index < len(elements)
                    and elements[index].element_type == "TableText"
                    and self.is_same_table(table_elements, elements[index])
                ):
                    table_elements.append(elements[index])
                    index += 1
                blocks.append(self.build_table_block(table_elements))
                continue

            block_type = "Paragraph" if element.element_type == "Text" else element.element_type
            current = Block(
                text=element.text,
                block_type=block_type,
                page_number=element.page_number,
                x0=element.x0, y0=element.y0, x1=element.x1, y1=element.y1,
            )
            previous = element
            index += 1
            while index < len(elements):
                following = elements[index]
                if following.element_type != element.element_type:
                    break
                # 标题必须来自同一个原始块才允许合并，避免合并相邻章节。
                if element.element_type != "Text" and (
                    element.source_block is None or element.source_block != following.source_block
                ):
                    break
                if not self.is_same_paragraph(previous, following):
                    break
                current.text += "\n" + following.text
                current.x0 = min(current.x0, following.x0)
                current.y0 = min(current.y0, following.y0)
                current.x1 = max(current.x1, following.x1)
                current.y1 = max(current.y1, following.y1)
                previous = following
                index += 1
            blocks.append(current)
        return blocks

    def build_table_block(self, elements):
        """把一张表格的所有单元格重组成可用文本。"""
        elements = list(elements)
        if not elements:
            raise ValueError("构建表格至少需要一个单元格")
        if len({element.page_number for element in elements}) != 1:
            raise ValueError("一个表格块不能跨页")

        lines = []
        if all(e.table_row is not None and e.table_column is not None for e in elements):
            # 使用原生行列信息保留空列、合并单元格位置和单元格内换行。
            rows = {}
            column_count = max(e.table_column for e in elements) + 1
            for element in elements:
                row = rows.setdefault(element.table_row, [""] * column_count)
                row[element.table_column] = " ".join(element.text.splitlines())
            lines = [" | ".join(rows[index]) for index in sorted(rows)]
        else:
            for row in self.cluster_rows(elements):
                lines.append(" | ".join(self.group_columns(row)))

        return Block(
            text="\n".join(lines),
            block_type="Table",
            page_number=elements[0].page_number,
            x0=min(element.x0 for element in elements),
            y0=min(element.y0 for element in elements),
            x1=max(element.x1 for element in elements),
            y1=max(element.y1 for element in elements),
        )

    def is_same_table(self, table_elements, current):
        first = table_elements[0]
        if first.page_number != current.page_number:
            return False
        if first.table_id is not None or current.table_id is not None:
            return first.table_id == current.table_id
        # 旧 TableDetector 没有表格 ID，只能用行高和空白间隔区分表格。
        bottom = max(e.y1 for e in table_elements)
        previous = table_elements[-1]
        line_height = max(previous.y1 - previous.y0, current.y1 - current.y0, 12.0)
        return current.y0 - bottom <= line_height * 2.5

    def cluster_rows(self, elements):

        """按纵向重叠把表格单元格聚成行。"""

        elements = sorted(
            elements,
            key=lambda e: (e.y0, e.x0)
        )

        rows = []

        current_row = []
        current_y1 = None

        for element in elements:

            if not current_row:
                current_row = [element]
                current_y1 = element.y1
                continue

            if element.y0 < current_y1:
                current_row.append(element)
                # 使用共同重叠区，避免较高的单元格把多行串成一行。
                current_y1 = min(current_y1, element.y1)
            else:
                rows.append(current_row)
                current_row = [element]
                current_y1 = element.y1

        if current_row:
            rows.append(current_row)

        return rows

    def group_columns(self, row):
        """把一行内 x 方向重叠的单元格合并成列，并保持阅读顺序。"""

        row = sorted(
            row,
            key=lambda e: (e.x0, e.y0)
        )

        columns = []

        for element in row:

            merged = False

            for column in columns:
                if self.overlap(
                    column["x0"], column["x1"],
                    element.x0, element.x1,
                ):
                    column["x0"] = min(column["x0"], element.x0)
                    column["x1"] = max(column["x1"], element.x1)
                    column["parts"].append(element)
                    merged = True
                    break

            if not merged:
                columns.append(
                    {
                        "x0": element.x0,
                        "x1": element.x1,
                        "parts": [element],
                    }
                )

        columns.sort(key=lambda column: column["x0"])

        texts = []
        for column in columns:
            column["parts"].sort(key=lambda e: e.y0)
            texts.append(
                " ".join(part.text for part in column["parts"])
            )

        return texts

    @staticmethod
    def overlap(a0, a1, b0, b1):
        return min(a1, b1) - max(a0, b0) > 0

    def is_same_paragraph(self, prev, current):

        # 防止 prev 是 None
        if prev is None:
            return False

        # 不同页面不能合并
        if prev.page_number != current.page_number:
            return False

        # 字体大小差太多，不能合并
        if (
            self.valid_font_size(prev)
            and self.valid_font_size(current)
            and abs(prev.font_size - current.font_size) > 1
        ):
            return False

        # x坐标差太多，不能合并
        if abs(prev.x0 - current.x0) > 10:
            return False

        # 计算两行之间的垂直距离
        vertical_gap = current.y0 - prev.y1

        # 间距太大，说明不是同一个段落
        line_size = (
            prev.font_size if self.valid_font_size(prev)
            else max(prev.y1 - prev.y0, 12.0)
        )
        # PDF 的文本块不是语义段落，有些生成器为每行创建独立块。
        # 跨块时允许紧密的连续行，但对段落空白采用更严格的阈值。
        if (
            prev.source_block is not None
            and current.source_block is not None
            and prev.source_block != current.source_block
            and vertical_gap > line_size * 0.8
        ):
            return False
        if vertical_gap > line_size * 1.5:
            return False

        # 出现明显重叠，也不合并
        if vertical_gap < -3:
            return False

        return True
