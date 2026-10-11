"""验证文档切片的内容完整性、结构边界和来源信息。"""

import unittest
from dataclasses import replace

from document.element import Block
from document.splitter import split_document


def block(text, *, kind="Paragraph", page=1):
    return Block(text, kind, page, 10, 20, 100, 200)


class DocumentSplitterTests(unittest.TestCase):
    def test_long_text_overlap_and_complete_coverage(self):
        source = "abcdefghijklmnopqrstuvwxyz"
        chunks = split_document([block(source)], chunk_size=10, chunk_overlap=3)
        self.assertEqual([c.text for c in chunks], [source[:10], source[7:17], source[14:24], source[21:]])
        restored = chunks[0].text + "".join(c.text[3:] for c in chunks[1:])
        self.assertEqual(restored, source)

    def test_sentence_and_paragraph_boundaries(self):
        for text, first in (("甲乙丙丁。戊己庚辛壬癸子丑", "甲乙丙丁。"),
                            ("abcdef\n\nghijklmnop", "abcdef\n\n"),
                            ("Hello. world again", "Hello. ")):
            with self.subTest(text=text):
                chunks = split_document([block(text)], chunk_size=10, chunk_overlap=0)
                self.assertEqual(chunks[0].text, first)
                self.assertEqual("".join(c.text for c in chunks), text)
                self.assertTrue(all(len(c.text) <= 10 for c in chunks))

    def test_structural_blocks_remain_complete(self):
        sources = [block("标题很长也保留完整", kind=kind) for kind in ("Title", "Heading", "Table")]
        chunks = split_document(sources, chunk_size=3, chunk_overlap=0)
        self.assertEqual(chunks, sources)
        self.assertTrue(all(c is not b for c, b in zip(chunks, sources)))

    def test_metadata_order_and_input_are_preserved(self):
        sources = [block("abcdefghijk", page=2), block("next", page=3)]
        original = [replace(b) for b in sources]
        chunks = split_document(iter(sources), chunk_size=6, chunk_overlap=2)
        self.assertEqual(sources, original)
        self.assertEqual([c.page_number for c in chunks], [2, 2, 2, 3])
        for chunk in chunks:
            self.assertEqual((chunk.block_type, chunk.x0, chunk.y0, chunk.x1, chunk.y1),
                             ("Paragraph", 10, 20, 100, 200))

    def test_empty_and_short_text(self):
        self.assertEqual(split_document([]), [])
        self.assertEqual(split_document([block(" \n\t")]), [])
        chunks = split_document([block(" short \n")])
        self.assertEqual(chunks[0].text, " short \n")

    def test_large_overlap_makes_progress_without_redundant_tail(self):
        text = "甲。乙。丙。丁。戊。己。庚。"
        chunks = split_document([block(text)], chunk_size=5, chunk_overlap=4)
        self.assertEqual(len(chunks), len(text) - 5 + 1)
        self.assertTrue(all(len(c.text) == 5 for c in chunks))
        self.assertEqual(chunks[0].text + "".join(c.text[4:] for c in chunks[1:]), text)
        self.assertEqual(len(split_document([block("12345")], chunk_size=5, chunk_overlap=4)), 1)

    def test_invalid_options(self):
        for options in ({"chunk_size": 0}, {"chunk_size": -1},
                        {"chunk_overlap": -1}, {"chunk_size": 5, "chunk_overlap": 5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                split_document([], **options)
        for options in ({"chunk_size": True}, {"chunk_size": 1.5},
                        {"chunk_overlap": False}, {"chunk_overlap": "1"}):
            with self.subTest(options=options), self.assertRaises(TypeError):
                split_document([], **options)


if __name__ == "__main__":
    unittest.main()
