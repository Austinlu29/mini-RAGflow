"""模型容量及保守的文本 token 预算。"""

import json


# 官方 /models 示例：容量包含输入与输出，不等于 max_tokens。
# https://api-docs.deepseek.com/api/list-models/
MODEL_CONTEXT_WINDOWS = {"deepseek-flash": 1048576, "deepseek-v4-pro": 1048576,
                         "deepseek-v4-flash": 1048576}


class ContextBudgetError(ValueError):
    """固定指令和输出预留使请求无法装入预算。"""


def model_context_window(model: str, override=None) -> int:
    if override is not None and str(override).strip():
        try:
            value = int(override)
        except (ValueError, TypeError):
            raise ValueError("DEEPSEEK_CONTEXT_WINDOW 必须是正整数 token 数") from None
    else:
        value = MODEL_CONTEXT_WINDOWS.get(model)
    if type(value) is not int or value < 1024:
        raise ValueError("未知模型或上下文容量无效，请配置 DEEPSEEK_CONTEXT_WINDOW（至少 1024）")
    return value


class TokenCounter:
    """以 UTF-8 字节数作为保守 token 估计；不是模型 tokenizer 的精确计数。

    不将字符数当 token 数。额外计入 JSON、消息封装和输出预留。
    在未提供官方 tokenizer 时倾向提前压缩，避免低估中文文本。
    """

    @staticmethod
    def text(value: str) -> int:
        return len(value.encode("utf-8"))

    @classmethod
    def messages(cls, system: str, user: str) -> int:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return cls.text(json.dumps(messages, ensure_ascii=False, separators=(",", ":"))) + 32

    @staticmethod
    def clip(value: str, budget: int) -> str:
        return value.encode("utf-8")[:max(0, budget)].decode("utf-8", errors="ignore")
