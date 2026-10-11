"""调用 embedding API，将文本或文档片段转换为归一化向量。"""

from collections.abc import Iterable
from math import isfinite, sqrt
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from document.element import Block


class DashScopeEmbedder(Embeddings):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_host: str | None = None,
        batch_size: int = 25,
        timeout: float = 60.0,
    ):
        """使用 text-embedding-v2，每批最多 25 条，输出 1536 维向量。

        从项目 .env 或环境变量读取 DASHSCOPE_API_KEY、DASHSCOPE_API_HOST。
        默认使用北京地域的公共 DashScope 地址，可配置专属 API Host。超长文本由 API
        报错，不静默截断；单条输入的模型上限为 2048 Token。
        """
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        self._api_key = api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY", "")
        if not self._api_key.strip():
            raise ValueError("请在 .env 或环境变量中配置 DASHSCOPE_API_KEY")
        host = api_host if api_host is not None else os.getenv("DASHSCOPE_API_HOST", "")
        host = (host or "https://dashscope.aliyuncs.com").rstrip("/")
        parsed = urlsplit(host)
        if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("DASHSCOPE_API_HOST 必须是有效的 HTTPS 地址")
        path = parsed.path.rstrip("/")
        service_path = "/services/embeddings/text-embedding/text-embedding"
        if path in ("", "/compatible-mode/v1"):
            self.endpoint = f"https://{parsed.netloc}/api/v1{service_path}"
        elif path == "/api/v1":
            self.endpoint = host + service_path
        elif path == "/api/v1" + service_path:
            self.endpoint = host
        else:
            raise ValueError("请提供 API Host、/api/v1 基础地址或 DashScope 文本向量接口地址")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size 必须是整数")
        if not 1 <= batch_size <= 25:
            raise ValueError("text-embedding-v2 的 batch_size 必须在 1 到 25 之间")
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout 必须是有限正数")
        self.batch_size = batch_size
        self.timeout = timeout
        self.model = "text-embedding-v2"
        self.embedding_id = f"dashscope:{self.endpoint}:{self.model}"

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        """生成底库文本的向量，输出顺序与输入顺序一致。"""
        return self._embed(texts, text_type="document")

    def embed_query(self, text: str) -> list[float]:
        """生成查询向量，使用与底库相同的模型和归一化方式。"""
        return self._embed([text], text_type="query")[0]

    def _embed(self, texts: Iterable[str], *, text_type: str) -> list[list[float]]:
        if isinstance(texts, str):
            raise TypeError("texts 必须是字符串序列，单条查询请使用 embed_query")
        texts = list(texts)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("每条文本必须是字符串")
        if any(not text.strip() for text in texts):
            raise ValueError("不能向量化空白文本")
        if not texts:
            return []
        embeddings = []
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                response = client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self.model, "input": {"texts": batch},
                          "parameters": {"text_type": text_type}},
                )
                if response.status_code != 200:
                    # 不输出响应正文，避免将服务端回显的凭证带进日志。
                    raise RuntimeError(f"DashScope 向量化失败，HTTP {response.status_code}")
                try:
                    items = response.json()["output"]["embeddings"]
                    if len(items) != len(batch):
                        raise ValueError
                    indices = [item["text_index"] for item in items]
                    if any(type(index) is not int for index in indices):
                        raise ValueError
                    if sorted(indices) != list(range(len(batch))):
                        raise ValueError
                    vectors = [item["embedding"] for item in sorted(items, key=lambda item: item["text_index"])]
                    vectors = normalize_vectors(vectors)
                    if any(len(vector) != 1536 for vector in vectors):
                        raise ValueError
                except (KeyError, TypeError, ValueError, OverflowError):
                    raise ValueError("DashScope 返回的向量数量、索引或 1536 维数据无效") from None
                embeddings.extend(vectors)
        return embeddings


def embed_chunks(
    chunks: Iterable[Block],
    *,
    embedder: DashScopeEmbedder | None = None,
) -> list[list[float]]:
    """将 split_document 返回的 Block 片段向量化，不修改输入。"""
    chunks = list(chunks)
    if not chunks:
        return []
    model = embedder if embedder is not None else DashScopeEmbedder()
    return model.embed_documents([chunk.text for chunk in chunks])


def normalize_vectors(vectors: Iterable[Iterable[float]]) -> list[list[float]]:
    """执行 L2 归一化，拒绝维度不一致、非有限值和零向量。"""
    result = []
    dimension = None
    for vector in vectors:
        values = [float(value) for value in vector]
        if not values or any(not isfinite(value) for value in values):
            raise ValueError("向量必须包含有限数值且不能为空")
        if dimension is None:
            dimension = len(values)
        if len(values) != dimension:
            raise ValueError("向量维度必须一致")
        norm = sqrt(sum(value * value for value in values))
        if norm == 0 or not isfinite(norm):
            raise ValueError("向量长度必须是有限正数")
        result.append([value / norm for value in values])
    return result
