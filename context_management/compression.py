"""分级策略与不产生新事实的本地原文抽取压缩。"""

from dataclasses import dataclass
import re

from .tokens import TokenCounter


@dataclass(frozen=True)
class CompressionLevel:
    name: str
    threshold: float
    target: float


LEVELS = (CompressionLevel("light", 0.30, 0.25),
          CompressionLevel("medium", 0.50, 0.35),
          CompressionLevel("strong", 0.70, 0.40),
          CompressionLevel("critical", 0.90, 0.45),
          CompressionLevel("overflow", 1.00, 0.50))


def select_level(ratio: float) -> int:
    return max((index for index, level in enumerate(LEVELS) if ratio >= level.threshold), default=-1)


def search_terms(text: str) -> list[str]:
    terms = re.findall(r"[a-z0-9_]+", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.extend(run[i:i + 2] for i in range(len(run) - 1))
    return list(dict.fromkeys(terms))


def compress_text(text: str, budget: int, query: str = "") -> str:
    """优先保留查询相关句子、首尾句；最终保证 UTF-8 预算且不拼出新事实。"""
    if TokenCounter.text(text) <= budget:
        return text
    if budget <= 0:
        return ""
    terms = search_terms(query)
    sentences = [part for part in re.split(r"(?<=[。！？.!?\n])", text) if part.strip()]
    scored = sorted(range(len(sentences)), key=lambda i: (
        sum(term in sentences[i].lower() for term in terms), i == len(sentences) - 1, i == 0), reverse=True)
    selected, used = [], 0
    for index in scored:
        size = TokenCounter.text(sentences[index]) + 1
        if used + size <= budget:
            selected.append(index)
            used += size
    if selected:
        return TokenCounter.clip("\n".join(sentences[i] for i in sorted(selected)), budget)
    # 单句很长时保留开头及结尾，使用换行隔开，原文仍可在本地归档恢复。
    head_budget = budget // 2
    head = TokenCounter.clip(text, head_budget)
    tail_budget = max(0, budget - TokenCounter.text(head) - 1)
    tail = text.encode("utf-8")[-tail_budget:].decode("utf-8", errors="ignore") if tail_budget else ""
    return TokenCounter.clip(head + "\n" + tail, budget)
