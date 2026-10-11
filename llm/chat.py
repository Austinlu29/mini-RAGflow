"""DeepSeek Chat Completions 客户端，只返回完整的 JSON，不流式输出。"""

import json
from math import isfinite
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv

from context_management.tokens import ContextBudgetError, TokenCounter, model_context_window


class DeepSeekChat:
    def __init__(self, *, api_key=None, base_url=None, model=None,
                 timeout=None, max_tokens=2048, context_window=None):
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        self._api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY", "")
        if not self._api_key.strip() or self._api_key.startswith("your-"):
            raise ValueError("请在 .env 或环境变量中配置 DEEPSEEK_API_KEY")
        base_url = (base_url if base_url is not None else
                    os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        parsed = urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username
                or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/v1")):
            raise ValueError("DEEPSEEK_BASE_URL 必须是 HTTPS 基础地址，可包含 /v1")
        self.endpoint = base_url + "/chat/completions"
        self.model = model if model is not None else os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        if not self.model.strip() or self.model == "deepseek-falsh":
            raise ValueError("请配置有效的 DEEPSEEK_MODEL，Flash 的名称为 deepseek-flash")
        try:
            self.timeout = float(timeout if timeout is not None else
                                 os.getenv("DEEPSEEK_TIMEOUT", "60"))
        except (TypeError, ValueError):
            raise ValueError("DEEPSEEK_TIMEOUT 必须是有限正数") from None
        if not isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("DEEPSEEK_TIMEOUT 必须是有限正数")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ValueError("max_tokens 必须是 1 到 8192 的整数")
        self.max_tokens = max_tokens
        self.context_window = model_context_window(
            self.model, context_window if context_window is not None else os.getenv("DEEPSEEK_CONTEXT_WINDOW"))
        if self.max_tokens >= self.context_window // 2:
            raise ValueError("输出预留 max_tokens 必须小于模型上下文容量的 50%")

    def complete_json(self, *, system: str, user: str) -> dict | None:
        """异常、截断或无效 JSON 不会成为可显示的回答；不记录凭证和响应正文。"""
        if TokenCounter.messages(system, user) + self.max_tokens >= self.context_window:
            raise ContextBudgetError("请求超过模型上下文容量，请先通过上下文管理器压缩")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self.model,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": user}],
                          "response_format": {"type": "json_object"},
                          "thinking": {"type": "disabled"},
                          "temperature": 0, "max_tokens": self.max_tokens,
                          "stream": False},
                )
        except httpx.HTTPError:
            raise RuntimeError("DeepSeek 请求失败，请检查网络连接") from None
        if response.status_code != 200:
            raise RuntimeError(f"DeepSeek 请求失败，HTTP {response.status_code}")
        try:
            choice = response.json()["choices"][0]
            if choice["finish_reason"] != "stop":
                return None
            message = choice["message"]
            if message.get("tool_calls") or message.get("role") != "assistant":
                return None
            result = json.loads(message["content"])
            return result if isinstance(result, dict) else None
        except (KeyError, IndexError, TypeError, ValueError):
            return None
