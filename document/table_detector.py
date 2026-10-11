class TableRegion:
    """表格区域：保存包含的行以及整体包围盒。"""

    def __init__(self, rows):
        self.rows = rows

        cells = [
            element
            for row in rows
            for element in row
        ]

        self.x0 = min(element.x0 for element in cells)
        self.y0 = min(element.y0 for element in cells)
        self.x1 = max(element.x1 for element in cells)
        self.y1 = max(element.y1 for element in cells)


class TableDetector:

    # 一行至少要有多少个单元格，才可能是表格行
    MIN_CELLS_PER_ROW = 2

    # 一个表格至少要有多少行
    MIN_ROWS = 2

    # 判断两行是否属于同一表格时，需要共享的列数
    MIN_SHARED_COLUMNS = 2

    # 行间距超过“基准行高 * GAP_FACTOR”时，认为表格结束
    GAP_FACTOR = 2.5
    GAP_BASE = 12.0

    # 列重叠判定时允许的误差
    OVERLAP_EPS = 1.0

    def detect(self, elements):

        # 1. 按页面分组
        pages = self.group_by_page(elements)

        for page_number, page_elements in pages.items():

            # 原生表格使用单元格边框，尺寸大于文本行；不能再次参与
            # 行高/间距推断，否则附近的正文、页脚会被误吸收到表格中。
            candidate_elements = [
                element for element in page_elements
                if element.element_type == "Text" and element.table_id is None
            ]

            # 2. 找出表格区域
            tables = self.find_table_candidates(
                candidate_elements
            )

            # 3. 落在表格区域内的元素标记为表格文本
            for table in tables:

                for element in candidate_elements:

                    if self.is_inside(element, table):
                        element.element_type = "TableText"

        return elements

    def group_by_page(self, elements):

        pages = {}

        for element in elements:
            pages.setdefault(
                element.page_number,
                []
            ).append(element)

        return pages

    def find_table_candidates(self, elements):
        """
        先按纵向重叠把元素聚成行，再把满足“列对齐、间距不大”的
        连续单元格行聚成候选表格。
        返回 TableRegion 列表。
        """

        rows = self.group_rows(elements)

        candidates = []

        current_rows = []
        reference_columns = None
        previous_bottom = None

        for row in rows:

            # 单元格太少，不是表格行，结束当前候选
            if len(row) < self.MIN_CELLS_PER_ROW:
                self.flush(current_rows, candidates)
                current_rows = []
                reference_columns = None
                previous_bottom = None
                continue

            # 候选表格的第一行
            if not current_rows:
                current_rows = [row]
                reference_columns = self.columns_of(row)
                previous_bottom = self.row_bottom(row)
                continue

            gap = self.row_top(row) - previous_bottom
            gap_limit = (
                max(
                    self.row_height(current_rows[-1]),
                    self.row_height(row),
                    self.GAP_BASE,
                )
                * self.GAP_FACTOR
            )
            shared_columns = self.count_shared_columns(
                reference_columns,
                row,
            )

            # 间距过大或者列对不上，说明上一张表结束了
            if (
                gap > gap_limit
                or shared_columns < self.MIN_SHARED_COLUMNS
            ):
                self.flush(current_rows, candidates)
                current_rows = [row]
                reference_columns = self.columns_of(row)
            else:
                current_rows.append(row)

            previous_bottom = self.row_bottom(row)

        self.flush(current_rows, candidates)

        return candidates

    def flush(self, rows, candidates):

        if self.is_table(rows):
            candidates.append(TableRegion(rows))

    def group_rows(self, elements):
        """按纵向重叠把元素聚成行，能正确处理单元格换行的情况。"""

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

            # 与当前行在纵向上有重叠，归为同一行
            if element.y0 < current_y1:
                current_row.append(element)
                current_y1 = max(current_y1, element.y1)
            else:
                rows.append(current_row)
                current_row = [element]
                current_y1 = element.y1

        if current_row:
            rows.append(current_row)

        return rows

    def columns_of(self, row):

        return [
            (element.x0, element.x1)
            for element in row
        ]

    def count_shared_columns(self, reference_columns, row):

        count = 0

        for column_x0, column_x1 in reference_columns:

            if any(
                self.overlap(
                    column_x0, column_x1,
                    element.x0, element.x1,
                )
                for element in row
            ):
                count += 1

        return count

    @staticmethod
    def overlap(a0, a1, b0, b1):
        return min(a1, b1) - max(a0, b0) > TableDetector.OVERLAP_EPS

    @staticmethod
    def row_top(row):
        return min(element.y0 for element in row)

    @staticmethod
    def row_bottom(row):
        return max(element.y1 for element in row)

    @staticmethod
    def row_height(row):
        return TableDetector.row_bottom(row) - TableDetector.row_top(row)

    def is_table(self, rows):

        # 至少需要两行
        if len(rows) < self.MIN_ROWS:
            return False

        # 统计每一行的 Element 数量
        row_lengths = [
            len(row)
            for row in rows
        ]

        # 至少要有两行包含多个单元格
        multi_cell_rows = sum(
            count >= self.MIN_CELLS_PER_ROW
            for count in row_lengths
        )

        if multi_cell_rows < self.MIN_ROWS:
            return False

        return True

    def is_inside(self, element, table):
        return (
                element.x0 >= table.x0
                and element.y0 >= table.y0
                and element.x1 <= table.x1
                and element.y1 <= table.y1
        )
