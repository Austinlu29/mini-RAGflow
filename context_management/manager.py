"""按模型上限管理活动上下文，压缩前保留原始内容，溢出后检索再压缩。"""

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from dotenv import load_dotenv

from .archive import LocalArchive
from .compression import LEVELS, compress_text, select_level
from .tokens import ContextBudgetError, TokenCounter


@dataclass(frozen=True)
class ContextSettings:
    max_tokens: int = 1048576
    output_reserve: int = 2048
    recent_turns: int = 6
    recall_top_k: int = 3
    archive_dir: Path = Path("data/context")

    def __post_init__(self):
        if type(self.max_tokens) is not int or self.max_tokens < 1024:
            raise ValueError("上下文上限必须是至少 1024 的整数")
        if type(self.output_reserve) is not int or not 1 <= self.output_reserve < self.max_tokens // 2:
            raise ValueError("输出预留必须为正整数且小于上下文上限的 50%")
        for name in ("recent_turns", "recall_top_k"):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 100:
                raise ValueError(f"{name} 必须是 1 到 100 的整数")


@dataclass(frozen=True)
class PreparedContext:
    user: str
    payload: dict
    tokens_before: int
    tokens_after: int
    level: str
    recalled_turn_ids: tuple[int, ...]


class ContextManager:
    def __init__(self, settings: ContextSettings | None = None, *, session_id: str | None = None):
        self.settings = settings or ContextSettings()
        self.session_id = session_id or uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.session_id):
            raise ValueError("会话 ID 仅支持 1 到 80 个字母、数字、下划线或连字符")
        self.archive = LocalArchive(Path(self.settings.archive_dir) / "history.sqlite3")
        try:
            state = self.archive.load_state(self.session_id) or {"summary": "", "recent": [], "level": -1}
        except Exception:
            self.archive.close()
            raise
        self.summary, self.recent, self.level = state["summary"], state["recent"], state["level"]
        if state.get("max_tokens", self.settings.max_tokens) != self.settings.max_tokens:
            self.level = -1
        self.last_report: PreparedContext | None = None

    @classmethod
    def from_env(cls, *, chat, session_id=None):
        root = Path(__file__).resolve().parents[1]
        load_dotenv(root / ".env")
        try:
            path = Path(os.getenv("CONTEXT_ARCHIVE_DIR", "data/context"))
            settings = ContextSettings(max_tokens=chat.context_window, output_reserve=chat.max_tokens,
                                       recent_turns=int(os.getenv("CONTEXT_RECENT_TURNS", "6")),
                                       recall_top_k=int(os.getenv("CONTEXT_RECALL_TOP_K", "3")),
                                       archive_dir=path if path.is_absolute() else root / path)
        except (ValueError, TypeError):
            raise ValueError("上下文配置无效，请检查模型容量、输出预留和 CONTEXT_* 配置") from None
        return cls(settings, session_id=session_id or os.getenv("CONTEXT_SESSION_ID") or None)

    @property
    def has_history(self):
        return bool(self.summary or self.recent)

    def _state(self):
        return {"summary": self.summary, "recent": deepcopy(self.recent), "level": self.level,
                "max_tokens": self.settings.max_tokens}

    def record(self, question: str, answer: str):
        if not question.strip() or not answer.strip():
            return
        state = self._state()
        self.archive.save_turn(self.session_id, question, answer, state)
        self.recent = state["recent"]

    @staticmethod
    def _encode(payload):
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _tokens(self, system, payload):
        return TokenCounter.messages(system, self._encode(payload)) + self.settings.output_reserve

    def prepare(self, *, system: str, payload: dict, query: str = "", include_history=True) -> PreparedContext:
        core = deepcopy(payload)
        conversation = {"summary": self.summary, "recent": deepcopy(self.recent), "recalled": []}
        if not include_history:
            conversation = {"summary": "", "recent": [], "recalled": []}
        full = {**core, "conversation": conversation} if include_history else core
        before = self._tokens(system, full)
        selected = select_level(before / self.settings.max_tokens)
        # 级别不倒退：一次强压缩后不在下一轮反复触发低级别压缩。
        should_compress = selected >= 0 and selected >= self.level
        overflow = before >= self.settings.max_tokens
        recalled = []
        if include_history and (overflow or self.level >= 0):
            recalled = self.archive.search(self.session_id, query, top_k=self.settings.recall_top_k,
                                           exclude=[] if overflow else [item["turn_id"] for item in self.recent
                                                                        if not item.get("compressed")])
            # 先按查询抽取恢复片段，不把已经压缩过的整包原文重新塞回上下文。
            per_turn = max(1, self.settings.max_tokens // 10 // max(1, len(recalled)))
            for item in recalled:
                item["user"] = compress_text(item["user"], per_turn // 3, query)
                item["assistant"] = compress_text(item["assistant"], per_turn * 2 // 3, query)
            conversation["recalled"] = recalled
            full["conversation"] = conversation
            expanded = self._tokens(system, full)
            selected = max(selected, select_level(expanded / self.settings.max_tokens))
            should_compress = selected >= 0 and selected >= self.level
            if expanded >= self.settings.max_tokens:
                overflow, selected, should_compress = True, len(LEVELS) - 1, True
        if should_compress:
            target = int(self.settings.max_tokens * LEVELS[selected].target)
        else:
            # 归档恢复同样必须有上限，避免添加历史后重新超过模型容量。
            target = self.settings.max_tokens - 1
        if overflow:
            selected = len(LEVELS) - 1
            target = self.settings.max_tokens // 2

        # 当前问题/证据自身过长时也要压缩，原始载荷先打包保存。
        if self._tokens(system, core) > target:
            self.archive.save_request(self.session_id, payload)
            self._fit_core(system, core, target, query)
        if include_history:
            if should_compress or overflow:
                older = conversation["recent"][:-self.settings.recent_turns]
                if older:
                    chunks = [conversation["summary"]] if conversation["summary"] else []
                    chunks += [f"用户：{item['user']}\n助手：{item['assistant']}" for item in older]
                    conversation["summary"] = "\n".join(chunks)
                    conversation["recent"] = conversation["recent"][-self.settings.recent_turns:]
            full = {**core, "conversation": conversation}
            self._fit_history(system, full, target, query)
            # recalled 只用于本次请求，原始历史仍留在本地库，不反复写入摘要。
            new_level = max(self.level, selected) if (should_compress or overflow) else self.level
            state = {"summary": conversation["summary"], "recent": conversation["recent"], "level": new_level,
                     "max_tokens": self.settings.max_tokens}
            self.archive.save_state(self.session_id, state)
            self.summary, self.recent, self.level = state["summary"], state["recent"], new_level
        else:
            full = core
        after = self._tokens(system, full)
        if after > target:
            raise ContextBudgetError("固定指令及输出预留超过目标容量，请提高模型容量或降低输出预留")
        level = LEVELS[selected].name if (should_compress or overflow) else "none"
        report = PreparedContext(self._encode(full), full, before, after, level,
                                 tuple(item["turn_id"] for item in recalled))
        self.last_report = report
        return report

    def _fit_core(self, system, payload, target, query):
        # 只改可压缩的文本，保留 JSON 协议、引用 ID、拒答候选原句和系统指令。
        while self._tokens(system, payload) > target:
            candidates = []
            if isinstance(payload.get("question"), str):
                candidates.append((payload, "question"))
            for ref in payload.get("references", []):
                if isinstance(ref, dict) and isinstance(ref.get("text"), str):
                    candidates.append((ref, "text"))
            candidates = [(obj, key) for obj, key in candidates if TokenCounter.text(obj[key]) > 8]
            if not candidates:
                if payload.get("references"):
                    payload["references"].pop()
                    continue
                raise ContextBudgetError("当前问题和固定协议无法装入压缩目标")
            obj, key = max(candidates, key=lambda pair: TokenCounter.text(pair[0][pair[1]]))
            excess = self._tokens(system, payload) - target
            obj[key] = compress_text(obj[key], max(8, TokenCounter.text(obj[key]) - excess - 16), query)

    def _fit_history(self, system, payload, target, query):
        history = payload["conversation"]
        while self._tokens(system, payload) > target:
            # 先缩旧摘要和归档恢复内容，最后缩最近对话；全部原文已经归档。
            candidates = [(history, "summary")]
            for item in history["recalled"] + history["recent"]:
                candidates.extend([(item, "user"), (item, "assistant")])
            candidates = [(obj, key) for obj, key in candidates if TokenCounter.text(obj[key]) > 16]
            if candidates:
                obj, key = candidates[0]
                excess = self._tokens(system, payload) - target
                obj[key] = compress_text(obj[key], max(16, TokenCounter.text(obj[key]) - excess - 16), query)
                if "turn_id" in obj:
                    obj["compressed"] = True
            elif history["recalled"]:
                history["recalled"].pop()
            elif history["recent"]:
                history["recent"].pop(0)
            elif history["summary"]:
                history["summary"] = ""
            else:
                self._fit_core(system, payload, target, query)
                if self._tokens(system, payload) > target:
                    raise ContextBudgetError("固定对话协议无法装入压缩目标")

    def close(self):
        self.archive.close()
