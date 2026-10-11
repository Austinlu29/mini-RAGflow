"""云端全量回读校验必须拒绝遗漏、重复和错误来源。"""

from hashlib import sha256
import json
import unittest

from document.element import Block
from evaluation.verify_milvus import verify_rows


class CloudVerificationTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [Block("first", "Paragraph", 1, 0, 0, 0, 0),
                       Block("second", "Paragraph", 2, 0, 0, 0, 0)]
        self.source = "crud-rag:test"
        self.rows = [{"id": sha256(f"{self.source}\0{index}".encode()).hexdigest(),
                      "text": chunk.text, "source": self.source, "chunk_index": index,
                      "metadata": {"page_number": chunk.page_number}}
                     for index, chunk in enumerate(self.chunks)]

    def test_full_cloud_response_can_be_unordered(self):
        verify_rows(list(reversed(self.rows)), self.chunks, self.source)

    def test_rest_json_fields_can_be_serialized_strings(self):
        rows = [dict(row, metadata=json.dumps(row["metadata"])) for row in self.rows]
        verify_rows(rows, self.chunks, self.source)

    def test_missing_duplicate_or_changed_data_is_rejected(self):
        for rows in (self.rows[:1], self.rows + self.rows[:1],
                     [dict(self.rows[0], text="wrong"), self.rows[1]],
                     [dict(self.rows[0], source="other"), self.rows[1]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                verify_rows(rows, self.chunks, self.source)


if __name__ == "__main__":
    unittest.main()
