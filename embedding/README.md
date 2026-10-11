# 切片向量化与云端 Milvus

使用阿里云 `text-embedding-v2` 生成 1536 维向量。模型负责向量化，
项目默认连接 Zilliz Cloud 的 Free 集群，通过 Milvus 执行余弦检索。
返回结果统一提供 `similarity` 和 `distance = 1 - similarity`，
分数越高越相似，不能视为正确率或概率。

`DashScopeEmbedder` 现在实现 LangChain `Embeddings` 接口，原来的方法继续可用。
`vectorstore.langchain.ChunkVectorStore` 将两种后端接入 `VectorStore`，
`EvidenceRetriever` 直接返回带页码和来源的 `Document`，主问答链按原始余弦阈值过滤。
完整组件用法与分数尺度见 [LangChain 架构](../README.md#langchain-架构)。

安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

在项目 `.env` 中设置 `DASHSCOPE_API_KEY` 和控制台提供的
`DASHSCOPE_API_HOST`，格式参见 `.env.example`。不要把真实密钥写到源码。
未配置 Host 时使用北京地域的公共地址 `https://dashscope.aliyuncs.com`，
也可指定业务空间专属 Host。新版 `sk-ws` 密钥的调用能力与旧版相同。
环境变量优先于 `.env`，显式构造参数优先于两者。

在 `.env` 中设置 `VECTOR_STORE=milvus`、`MILVUS_URI`、`MILVUS_TOKEN`、
`MILVUS_COLLECTION`。URI 使用控制台复制的完整公共 Endpoint；Token 可用
Zilliz API Key 或已有集群的 `username:password`，不复用阿里云密钥。

直接将现有切片向量化并入库：

```python
from embedding.embed import DashScopeEmbedder, embed_chunks
from vectorstore.factory import create_vector_store

embedder = DashScopeEmbedder()
vectors = embed_chunks(chunks, embedder=embedder)
store = create_vector_store(embedding_model=embedder.embedding_id)
try:
    ids = store.add_chunks(chunks, vectors, source="my-document.pdf")
    results = store.search(embedder.embed_query("库存准确率是多少？"), top_k=3)
    for result in results:
        print(result["similarity"], result["metadata"]["page_number"], result["text"])
finally:
    store.close()
```

也可以直接跑完整 PDF 调用链：

```python
from RAG.pipeline import index_pdf

ids = index_pdf("data/test.pdf")
print(f"已保存 {len(ids)} 个片段")
```

向量、文本、页码、类型、原始坐标、来源和片段序号保存于云端集合。
重复导入同一 source 时更新已有 ID，并在写入成功后清除旧的多余片段。
空输入不清空原文档，数据库分批写入不具备事务性。
集合校验模型信息、维度、字段和 COSINE 索引，防止混用向量空间。

命令行用法：

```powershell
# 仅验证连接，不调用 embedding API、不上传文档
.\.venv\Scripts\python.exe main.py --check
# 显式入库默认样本，也可指定 --pdf 路径
.\.venv\Scripts\python.exe main.py --index
# 只查询已有片段，不重新向量化文档
.\.venv\Scripts\python.exe main.py --query "库存准确率是多少？"
# 复用本地 Chroma 已有向量迁移到云端，不调用 embedding API
.\.venv\Scripts\python.exe -m vectorstore.migrate_chroma
```

同一文件重复入库会更新固定 ID，全部写入成功后清理旧多余片段。入库命令
仍会重新调用 embedding API；迁移和仅查询命令不会重新向量化文档。
Milvus 查询使用 Strong 一致性验证新数据。分批更新没有跨请求事务保证，
中途网络失败时可重试同一文件。文档切片减少时会删除该文档的旧多余片段。

原有 Chroma 数据仍在 `data/chroma`，可以通过 `--backend chroma` 或
`VECTOR_STORE=chroma` 继续使用。切换后端时，请关闭旧连接，使用新后端
对应的集合名称。Milvus 默认集合来自 `MILVUS_COLLECTION`，Chroma 默认
集合仍是 `documents`。

每次 API 调用最多 25 条文本。单条上限为 2048 Token，字符数不等于
Token 数。当前切片器保留完整表格，过长表格可能触发 API 限制，此时
异常会直接抛出，不截断文本、不写入部分 API 结果。

接口与距离定义参考
[阿里云文本向量 API](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)
和 [Zilliz Python SDK](https://docs.zilliz.com.cn/docs/install-sdks)。
