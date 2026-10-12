"""供离线存储与流程测试复用的内存 Milvus 客户端。"""

import copy
import json

from pymilvus import MilvusClient


class FakeIterator:
    def __init__(self, rows):
        self.rows = rows
        self.position = 0
        self.closed = False
        self.failure = None

    def next(self):
        if self.failure is not None:
            raise self.failure
        batch = self.rows[self.position:self.position + 1]
        self.position += len(batch)
        return batch

    def close(self):
        self.closed = True


class FakeMilvusClient:
    create_schema = staticmethod(MilvusClient.create_schema)
    prepare_index_params = staticmethod(MilvusClient.prepare_index_params)

    def __init__(self):
        self.schema = None
        self.indexes = {}
        self.rows = {}
        self.iterators = []
        self.upsert_calls = []
        self.delete_calls = []
        self.search_calls = []
        self.iterator_calls = []
        self.create_calls = []
        self.closed = False
        self.fail_upsert_at = None
        self.fail_iterator = False

    def has_collection(self, **kwargs):
        return self.schema is not None

    def create_collection(self, **kwargs):
        self.create_calls.append(kwargs)
        self.schema = kwargs["schema"].to_dict()
        self.indexes = {item.index_name: item.to_dict() for item in kwargs["index_params"]}

    def describe_collection(self, **kwargs):
        return copy.deepcopy(self.schema)

    def list_indexes(self, **kwargs):
        return list(self.indexes)

    def describe_index(self, collection_name, index_name, **kwargs):
        return copy.deepcopy(self.indexes[index_name])

    def load_collection(self, **kwargs):
        pass

    def get_load_state(self, **kwargs):
        return {"state": "Loaded"}

    def _filtered(self, expression=""):
        rows = list(self.rows.values())
        if expression:
            field, value = expression.split("==", 1)
            assert field.strip() == "source", expression
            source = json.loads(value.strip())
            rows = [row for row in rows if row["source"] == source]
        return rows

    def query_iterator(self, **kwargs):
        self.iterator_calls.append(kwargs)
        iterator = FakeIterator([{"id": row["id"]} for row in self._filtered(kwargs.get("filter", ""))])
        if self.fail_iterator:
            iterator.failure = RuntimeError("iterator unavailable")
        self.iterators.append(iterator)
        return iterator

    def upsert(self, **kwargs):
        self.upsert_calls.append(copy.deepcopy(kwargs))
        if len(self.upsert_calls) == self.fail_upsert_at:
            raise RuntimeError("upsert unavailable")
        for row in kwargs["data"]:
            self.rows[row["id"]] = copy.deepcopy(row)
        return {"upsert_count": len(kwargs["data"])}

    def delete(self, **kwargs):
        self.delete_calls.append(kwargs)
        for row_id in kwargs["ids"]:
            self.rows.pop(row_id, None)
        return {"delete_count": len(kwargs["ids"])}

    def query(self, **kwargs):
        rows = self._filtered(kwargs.get("filter", ""))
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": len(rows)}]
        return copy.deepcopy(rows)

    def get_collection_stats(self, **kwargs):
        return {"row_count": len(self.rows)}

    def search(self, **kwargs):
        self.search_calls.append(copy.deepcopy(kwargs))
        query = kwargs["data"][0]
        hits = []
        for row in self._filtered(kwargs.get("filter", "")):
            cosine = sum(a * b for a, b in zip(query, row["vector"]))
            hits.append({"id": row["id"], "distance": cosine, "entity": copy.deepcopy(row)})
        return [sorted(hits, key=lambda hit: hit["distance"], reverse=True)[:kwargs["limit"]]]

    def close(self):
        self.closed = True
