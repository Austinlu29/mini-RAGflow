"""DeepSeek Chat Completions 客户端，只返回完整的 JSON，不流式输出。"""

import json
from math import isfinite
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from context_management.tokens import ContextBudgetError, TokenCounter, model_context_window


class DeepSeekChat(BaseChatModel):
    """LangChain 聊天模型，保留 JSON 协议、预算限制和安全错误信息。"""

    endpoint: str
    model: str
    timeout: float
    max_tokens: int
    context_window: int
    _api_key: str = PrivateAttr()

    def __init__(self, *, api_key=None, base_url=None, model=None,
                 timeout=None, max_tokens=2048, context_window=None):
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY", "")
        if not key.strip() or key.startswith("your-"):
            raise ValueError("请在 .env 或环境变量中配置 DEEPSEEK_API_KEY")
        base_url = (base_url if base_url is not None else
                    os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        parsed = urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username
                or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/v1")):
            raise ValueError("DEEPSEEK_BASE_URL 必须是 HTTPS 基础地址，可包含 /v1")
        model_name = model if model is not None else os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        if not model_name.strip() or model_name == "deepseek-falsh":
            raise ValueError("请配置有效的 DEEPSEEK_MODEL，Flash 的名称为 deepseek-flash")
        try:
            request_timeout = float(timeout if timeout is not None else
                                    os.getenv("DEEPSEEK_TIMEOUT", "60"))
        except (TypeError, ValueError):
            raise ValueError("DEEPSEEK_TIMEOUT 必须是有限正数") from None
        if not isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("DEEPSEEK_TIMEOUT 必须是有限正数")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ValueError("max_tokens 必须是 1 到 8192 的整数")
        window = model_context_window(
            model_name, context_window if context_window is not None else os.getenv("DEEPSEEK_CONTEXT_WINDOW"))
        if max_tokens >= window // 2:
            raise ValueError("输出预留 max_tokens 必须小于模型上下文容量的 50%")
        super().__init__(endpoint=base_url + "/chat/completions", model=model_name,
                         timeout=request_timeout, max_tokens=max_tokens, context_window=window)
        self._api_key = key

    @property
    def _llm_type(self):
        return "miniragflow-deepseek-json"

    @property
    def _identifying_params(self):
        return {"model": self.model, "endpoint": self.endpoint,
                "max_tokens": self.max_tokens, "context_window": self.context_window}

    def complete_json(self, *, system: str, user: str) -> dict | None:
        """异常、截断或无效 JSON 不会成为可显示的回答；不记录凭证和响应正文。"""
        from .chains import StrictJsonOutputParser
        message = self.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return StrictJsonOutputParser().invoke(message)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if kwargs or stop is not None:
            raise ValueError("JSON 模型不支持额外生成参数或 stop 截断")
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        payload_messages = []
        for message in messages:
            if (message.type not in roles or not isinstance(message.content, str)
                    or getattr(message, "tool_calls", None)
                    or message.additional_kwargs.get("tool_calls")):
                raise ValueError("JSON 模型只支持不带工具调用的文本消息")
            payload_messages.append({"role": roles[message.type], "content": message.content})
        tokens = TokenCounter.text(json.dumps(payload_messages, ensure_ascii=False, separators=(",", ":"))) + 32
        if tokens + self.max_tokens >= self.context_window:
            raise ContextBudgetError("请求超过模型上下文容量，请先通过上下文管理器压缩")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self.model,
                          "messages": payload_messages,
                          "response_format": {"type": "json_object"},
                          "thinking": {"type": "disabled"},
                          "temperature": 0, "max_tokens": self.max_tokens,
                          "stream": False},
                )
        except httpx.HTTPError:
            raise RuntimeError("DeepSeek 请求失败，请检查网络连接") from None
        if response.status_code != 200:
            raise RuntimeError(f"DeepSeek 请求失败，HTTP {response.status_code}")
        content, finish = "", "invalid_response"
        try:
            choice = response.json()["choices"][0]
            finish = choice["finish_reason"]
            message = choice["message"]
            if (finish == "stop" and message.get("role") == "assistant"
                    and not message.get("tool_calls") and isinstance(message.get("content"), str)):
                content = message["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content=content, response_metadata={"finish_reason": finish}))])
