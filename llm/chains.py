"""可组合的提示词、聊天模型和严格 JSON 解析链。"""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import ChatGeneration
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


class StrictJsonOutputParser(BaseOutputParser[dict | None]):
    """只接受完整 JSON 对象；不修复截断 JSON，也不移除 Markdown。"""

    def parse(self, text: str) -> dict | None:
        try:
            value = json.loads(text)
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def parse_result(self, result, *, partial=False) -> dict | None:
        if partial or not result:
            return None
        generation = result[0]
        if isinstance(generation, ChatGeneration):
            message = generation.message
            if (message.response_metadata.get("finish_reason", "stop") != "stop"
                    or message.additional_kwargs.get("tool_calls") or getattr(message, "tool_calls", None)):
                return None
        return self.parse(generation.text)

    @property
    def _type(self):
        return "strict_json_object"


def create_json_chain(chat, system: str):
    """原生 BaseChatModel 走提示词和解析器；保留旧客户端的注入接口。"""
    # 使用变量传递系统指令，避免将 JSON 示例中的花括号当作模板变量。
    prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{user}")]).partial(system=system)
    if isinstance(chat, BaseChatModel):
        model = chat
    else:
        def complete(prompt_value):
            messages = prompt_value.to_messages()
            result = chat.complete_json(system=messages[0].content, user=messages[1].content)
            return json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else ""
        model = RunnableLambda(complete, name="legacy_json_client")
    return prompt | model | StrictJsonOutputParser()
