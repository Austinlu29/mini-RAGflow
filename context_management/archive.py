"""SQLite 保存 gzip 原始对话包与检索索引，仅访问本会话。"""

import gzip
import json
from pathlib import Path
import sqlite3

from .compression import search_terms


class LocalArchive:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS packages (
                id INTEGER PRIMARY KEY, session TEXT NOT NULL, kind TEXT NOT NULL,
                payload BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS packages_session ON packages(session, kind);
            CREATE VIRTUAL TABLE IF NOT EXISTS package_search USING fts5(terms);
            CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, state BLOB NOT NULL);
        """)

    @staticmethod
    def _pack(data):
        return gzip.compress(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    @staticmethod
    def _unpack(data):
        return json.loads(gzip.decompress(data).decode("utf-8"))

    def save_turn(self, session: str, question: str, answer: str, state: dict) -> int:
        # 原文与活动状态在同一事务提交，先归档成功再替换内存。
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO packages(session, kind, payload) VALUES (?, 'turn', ?)",
                (session, self._pack({"user": question, "assistant": answer})))
            turn_id = cursor.lastrowid
            self.connection.execute("INSERT INTO package_search(rowid, terms) VALUES (?, ?)",
                                    (turn_id, " ".join(search_terms(question + "\n" + answer))))
            state["recent"].append({"turn_id": turn_id, "user": question, "assistant": answer})
            self.save_state(session, state, commit=False)
        return turn_id

    def save_request(self, session: str, payload: dict):
        with self.connection:
            self.connection.execute("INSERT INTO packages(session, kind, payload) VALUES (?, 'request', ?)",
                                    (session, self._pack(payload)))

    def save_state(self, session: str, state: dict, *, commit=True):
        if commit:
            with self.connection:
                self.connection.execute("INSERT OR REPLACE INTO sessions(id, state) VALUES (?, ?)",
                                        (session, self._pack(state)))
        else:
            self.connection.execute("INSERT OR REPLACE INTO sessions(id, state) VALUES (?, ?)",
                                    (session, self._pack(state)))

    def load_state(self, session: str) -> dict | None:
        row = self.connection.execute("SELECT state FROM sessions WHERE id=?", (session,)).fetchone()
        return self._unpack(row[0]) if row else None

    def search(self, session: str, query: str, *, top_k=3, exclude=()) -> list[dict]:
        # 只取本会话归档。中文按二字组合索引，参数化查询避免 FTS 语法注入。
        terms = search_terms(query)[:64]
        excluded = set(exclude)
        rows = []
        if terms:
            expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
            rows = self.connection.execute("""
                SELECT p.id, p.payload FROM package_search
                JOIN packages p ON p.id=package_search.rowid
                WHERE package_search MATCH ? AND p.session=? AND p.kind='turn'
                ORDER BY bm25(package_search), p.id DESC LIMIT ?
            """, (expression, session, top_k + len(excluded))).fetchall()
        results = [{"turn_id": turn_id, **self._unpack(payload)} for turn_id, payload in rows
                   if turn_id not in excluded][:top_k]
        if not results:
            # 指代性追问没有匹配词时，从最近的已打包历史恢复。
            rows = self.connection.execute("SELECT id, payload FROM packages WHERE session=? AND kind='turn' "
                                           "ORDER BY id DESC LIMIT ?", (session, top_k + len(excluded))).fetchall()
            results = [{"turn_id": turn_id, **self._unpack(payload)} for turn_id, payload in rows
                       if turn_id not in excluded][:top_k]
        return results

    def count(self, session: str) -> int:
        return self.connection.execute("SELECT count(*) FROM packages WHERE session=? AND kind='turn'",
                                       (session,)).fetchone()[0]

    def close(self):
        self.connection.close()
