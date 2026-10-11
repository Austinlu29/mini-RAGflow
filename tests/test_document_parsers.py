"""解析器回归测试：合成 PDF 验证提取，坐标样例验证布局边界。"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf

from document.element import Element
from document.layout_parser import LayoutParser
from document.pdf_parser import pdf_parser
from document.table_detector import TableDetector


def element(text="body", *, page=1, x=50, y=50, width=100, height=12,
            size=10, kind="Text", **metadata):
    return Element(text, kind, page, x, y, x + width, y + height, size, **metadata)


def draw_table(page, *, top=100, left=50, values=None):
    values = values or [["Name", "Value", "Note"], ["Alice", "", "two\nlines"]]
    width, height = 100, 40
    for row in range(len(values) + 1):
        page.draw_line((left, top + row * height), (left + 3 * width, top + row * height))
    for column in range(4):
        page.draw_line((left + column * width, top),
                       (left + column * width, top + len(values) * height))
    for row, cells in enumerate(values):
        for column, text in enumerate(cells):
            if text:
                page.insert_text((left + column * width + 5, top + row * height + 15),
                                 text, fontsize=10)


class PdfParserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.pdf"

    def save(self, build, **options):
        with pymupdf.open() as doc:
            build(doc)
            doc.save(self.path, **options)

    def test_text_pages_and_original_page_numbers(self):
        def build(doc):
            for text in ("First page", "Second page"):
                doc.new_page().insert_text((50, 50), text, fontsize=11)
        self.save(build)
        elements = pdf_parser(self.path, start_page=2, end_page=2)
        self.assertEqual([(e.text, e.page_number) for e in elements], [("Second page", 2)])
        self.assertAlmostEqual(elements[0].font_size, 11)
        self.assertIsNotNone(elements[0].source_block)

    def test_blank_page(self):
        self.save(lambda doc: doc.new_page())
        self.assertEqual(pdf_parser(self.path), [])

    def test_page_range_validation(self):
        self.save(lambda doc: doc.new_page())
        for options in ({"start_page": 0}, {"start_page": 2}, {"end_page": 2},
                        {"start_page": 2, "end_page": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                pdf_parser(self.path, **options)
        for options in ({"start_page": True}, {"end_page": 1.5}):
            with self.subTest(options=options), self.assertRaises(TypeError):
                pdf_parser(self.path, **options)

    def test_missing_file_is_not_silently_ignored(self):
        with self.assertRaises(pymupdf.FileNotFoundError):
            pdf_parser(self.path)

    def test_encrypted_pdf(self):
        self.save(lambda doc: doc.new_page().insert_text((50, 50), "Secret"),
                  encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="reader")
        for password in (None, "wrong"):
            with self.subTest(password=password), self.assertRaises(ValueError):
                pdf_parser(self.path, password=password)
        self.assertEqual(pdf_parser(self.path, password="reader")[0].text, "Secret")

    def test_document_is_closed_on_extraction_error(self):
        doc = MagicMock()
        doc.__enter__.return_value = doc
        doc.needs_pass = False
        doc.is_pdf = True
        doc.page_count = 1
        doc.__getitem__.return_value.get_text.side_effect = RuntimeError("bad page")
        with patch("document.pdf_parser.pymupdf.open", return_value=doc):
            with self.assertRaisesRegex(RuntimeError, "bad page"):
                pdf_parser(self.path, detect_tables=False)
        doc.__exit__.assert_called_once()

    def test_missing_span_sizes_and_empty_lines(self):
        doc = MagicMock()
        doc.__enter__.return_value = doc
        doc.needs_pass = False
        doc.is_pdf = True
        doc.page_count = 1
        page = doc.__getitem__.return_value
        page.get_text.return_value = {"blocks": [{"type": 0, "lines": [
            {"bbox": (0, 0, 50, 12), "spans": [{"text": "hello"}]},
            {"bbox": (0, 20, 50, 32), "spans": []},
        ]}]}
        with patch("document.pdf_parser.pymupdf.open", return_value=doc):
            result = pdf_parser(self.path, detect_tables=False)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].font_size)
        flags = page.get_text.call_args.kwargs["flags"]
        self.assertFalse(flags & pymupdf.TEXT_PRESERVE_IMAGES)

    def test_native_tables_preserve_empty_and_multiline_cells_without_duplicates(self):
        def build(doc):
            page = doc.new_page()
            page.insert_text((50, 50), "Introduction", fontsize=10)
            draw_table(page)
            draw_table(page, top=210, values=[["X", "Y", "Z"], ["1", "2", "3"]])
            draw_table(doc.new_page())
        self.save(build)
        elements = pdf_parser(self.path)
        self.assertEqual([e.text for e in elements if e.element_type == "Text"], ["Introduction"])
        self.assertTrue(any(e.text == "" for e in elements))
        self.assertTrue(any(e.text == "two\nlines" for e in elements))
        # 验证与 main.py 当前调用链兼容。
        blocks = LayoutParser().parse(TableDetector().detect(elements))
        tables = [b for b in blocks if b.block_type == "Table"]
        self.assertEqual(len(tables), 3)
        self.assertEqual([b.page_number for b in tables], [1, 1, 2])
        self.assertEqual(tables[0].text, "Name | Value | Note\nAlice |  | two lines")
        self.assertEqual(tables[1].text, "X | Y | Z\n1 | 2 | 3")

    def test_table_detection_can_be_disabled(self):
        self.save(lambda doc: draw_table(doc.new_page()))
        elements = pdf_parser(self.path, detect_tables=False)
        self.assertTrue(elements)
        self.assertTrue(all(e.element_type == "Text" for e in elements))

    def test_merged_cell_keeps_missing_column_position(self):
        def build(doc):
            page = doc.new_page()
            for y in (100, 140, 180):
                page.draw_line((50, y), (350, y))
            for x in (50, 250, 350):
                page.draw_line((x, 100), (x, 180))
            page.draw_line((150, 140), (150, 180))
            for x, y, text in ((55, 115, "Merged"), (255, 115, "Right"),
                               (55, 155, "A"), (155, 155, "B"), (255, 155, "C")):
                page.insert_text((x, y), text, fontsize=10)
        self.save(build)
        blocks = LayoutParser().parse(pdf_parser(self.path))
        self.assertEqual([b.text for b in blocks], ["Merged |  | Right\nA | B | C"])

    def test_table_detection_failure_falls_back_to_text(self):
        self.save(lambda doc: doc.new_page().insert_text((50, 50), "Still readable"))
        with patch.object(pymupdf.Page, "find_tables", side_effect=RuntimeError("bad table")):
            with self.assertLogs("document.pdf_parser", level="WARNING"):
                self.assertEqual(pdf_parser(self.path)[0].text, "Still readable")


class LayoutParserTests(unittest.TestCase):
    def test_empty_and_whitespace_input(self):
        self.assertEqual(LayoutParser().parse([]), [])
        self.assertEqual(LayoutParser().parse([element("  \n")]), [])

    def test_missing_or_invalid_fonts_do_not_crash(self):
        for size in (None, 0, float("nan")):
            with self.subTest(size=size):
                blocks = LayoutParser().parse([element(size=size), element(y=65, size=size)])
                self.assertEqual(len(blocks), 1)
                self.assertEqual(blocks[0].block_type, "Paragraph")

    def test_parse_does_not_mutate_input_and_accepts_generator(self):
        elements = [element("Title", size=22), element("body " * 30, y=80)]
        original = [replace(e) for e in elements]
        parser = LayoutParser()
        self.assertEqual(parser.parse(iter(elements)), parser.parse(elements))
        self.assertEqual(elements, original)

    def test_relative_title_classification(self):
        for body_size in (8, 16):
            with self.subTest(body_size=body_size):
                blocks = LayoutParser().parse([
                    element("Title", size=body_size * 2),
                    element("Heading", y=80, size=body_size * 1.4),
                    element("body " * 40, y=110, size=body_size),
                ])
                self.assertEqual([b.block_type for b in blocks], ["Title", "Heading", "Paragraph"])

    def test_explicit_thresholds_and_preclassified_elements(self):
        blocks = LayoutParser(title_font_size=18, heading_font_size=13).parse([
            element("title", size=18), element("heading", y=80, size=13),
            element("upstream title", y=110, size=None, kind="Title"),
        ])
        self.assertEqual([b.block_type for b in blocks], ["Title", "Heading", "Title"])
        for options in ({"title_font_size": 0}, {"heading_font_size": float("nan")},
                        {"title_font_size": 12, "heading_font_size": 13}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                LayoutParser(**options)

    def test_sorting_and_complete_merged_bounds(self):
        blocks = LayoutParser().parse([
            element("second", x=45, y=65, width=120),
            element("first", x=50, y=50),
        ])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "first\nsecond")
        self.assertEqual((blocks[0].x0, blocks[0].y0, blocks[0].x1, blocks[0].y1),
                         (45, 50, 165, 77))

    def test_independent_paragraphs_are_not_merged(self):
        first = element(source_block=1)
        for second in (element(page=2, y=65), element(x=250, y=65),
                       element(y=100), element(y=75, source_block=2)):
            with self.subTest(second=second):
                self.assertEqual(len(LayoutParser().parse([first, second])), 2)

    def test_wrapped_body_can_cross_pdf_source_blocks(self):
        blocks = LayoutParser().parse([
            element("first line", source_block=1),
            element("second line", y=65, source_block=2),
        ])
        self.assertEqual([b.text for b in blocks], ["first line\nsecond line"])

    def test_wrapped_title_only_merges_within_original_block(self):
        blocks = LayoutParser().parse([
            element("first title line", kind="Title", source_block=1),
            element("second title line", kind="Title", y=65, source_block=1),
            element("next title", kind="Title", y=80, source_block=2),
        ])
        self.assertEqual([b.text for b in blocks], ["first title line\nsecond title line", "next title"])

    def test_tall_cell_does_not_join_separate_rows(self):
        cells = [element("tall", height=50), element("short", x=200),
                 element("next row", x=200, y=70)]
        self.assertEqual([len(row) for row in LayoutParser().cluster_rows(cells)], [2, 1])

    def test_native_table_geometry_does_not_absorb_footer_in_legacy_detector(self):
        # 原生表格行高 40，页脚距表格 20；旧推断会误认为它们连续。
        cells = [element("A", height=40, kind="TableText", table_id=0,
                         table_row=0, table_column=0),
                 element("B", x=200, height=40, kind="TableText", table_id=0,
                         table_row=0, table_column=1),
                 element("C", y=90, height=40, kind="TableText", table_id=0,
                         table_row=1, table_column=0),
                 element("D", x=200, y=90, height=40, kind="TableText", table_id=0,
                         table_row=1, table_column=1),
                 element("footer", y=150), element("page 1", x=200, y=150)]
        detected = TableDetector().detect(cells)
        self.assertEqual([e.element_type for e in detected[-2:]], ["Text", "Text"])

    def test_legacy_tables_do_not_merge_across_pages_or_large_gaps(self):
        blocks = LayoutParser().parse([
            element("A", kind="TableText"),
            element("B", page=2, kind="TableText"),
            element("C", page=2, y=200, kind="TableText"),
        ])
        self.assertEqual([b.text for b in blocks], ["A", "B", "C"])
        self.assertEqual([b.page_number for b in blocks], [1, 2, 2])

    def test_native_tables_are_atomic_and_distinguished_by_id(self):
        cells = [
            element("A", kind="TableText", table_id=0, table_row=0, table_column=0),
            element("B", y=90, kind="TableText", table_id=0, table_row=1, table_column=0),
            element("C", x=250, kind="TableText", table_id=1, table_row=0, table_column=0),
            element("side text", x=400, y=65),
        ]
        blocks = LayoutParser().parse(reversed(cells))
        self.assertEqual([b.text for b in blocks], ["A\nB", "C", "side text"])

    def test_geometry_tables_keep_rows_and_columns(self):
        cells = [element("A", kind="TableText"), element("B", x=200, kind="TableText"),
                 element("C", y=70, kind="TableText"),
                 element("D", x=200, y=70, kind="TableText")]
        self.assertEqual(LayoutParser().parse(cells)[0].text, "A | B\nC | D")

    def test_table_builder_rejects_empty_or_mixed_pages(self):
        for cells in ([], [element(kind="TableText"), element(page=2, kind="TableText")]):
            with self.subTest(cells=cells), self.assertRaises(ValueError):
                LayoutParser().build_table_block(cells)


if __name__ == "__main__":
    unittest.main()
