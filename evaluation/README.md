# CRUD-RAG 30 问向量检索评测

本模块将 [IAAR-Shanghai/CRUD_RAG](https://github.com/IAAR-Shanghai/CRUD_RAG) 的官方数据接入 MiniRAGFlow，复用项目已有 `split_document`、`DashScopeEmbedder` 和 Milvus/Chroma 存储接口。

## 数据与抽样

- 上游版本固定为 `1aace383994e1f68efa12cf2a8e2dadfb4102ceb`。
- 原始数据：`data/crud_split/split_merged.json`，SHA256 存于 `benchmark.json`。
- 抽样种子 `20261010`，`questanswer_1doc`、`questanswer_2docs`、`questanswer_3docs` 各 10 问，覆盖 30 个不同事件。保留官方问题和参考答案原文。这些是基于真实新闻构建的基准问题，并非真实用户日志。
- 相关文档直接取这些记录的 `news1/news2/news3`。正文按去除空白后的内容去重，共 59 篇参考新闻。
- 从其他事件的官方问答记录中提取新闻上下文作为干扰语料；排除与参考新闻相同或有包含关系的正文，固定组成 1000 篇新闻（59 参考 + 941 干扰）。每篇保留 task、记录索引、事件 ID 和字段来源。
- 只索引 `documents[].text`，不索引 questions、answers、thoughts 或 event。数据文件中保留参考答案用于审阅，但不会将其额外写入向量库。
- 此评测使用官方 QA 的新闻上下文子集，未导入约 8 万篇完整检索语料，成绩不可直接对比 CRUD-RAG 论文完整基准。

本地生成的 `data/crud_rag/benchmark.json` 保存完整子集和相关文档标签；`data/crud_rag/questions.md` 是方便人工阅读的 30 问及参考答案清单。两者均不随仓库发布，使用下方命令生成。

## 已有实测结果

2026-10-10 的本地评测使用真实 DashScope 向量服务，对 30 问、1000 篇新闻、2420 个片段执行检索。汇总文件见 [results_summary.json](results_summary.json)。

| 后端 | Hit@1 | 文档 Recall@5 | 文档 Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| Chroma | 96.67% | 94.44% | 100% | 0.9833 |
| Milvus | 96.67% | 94.44% | 100% | 0.9833 |

这些指标只描述固定小规模子集的文档召回，不代表完整 CRUD-RAG 基准成绩或问答正确率。仓库仅保存聚合指标；新闻原文、逐问答案、检索片段和本地报告不随仓库发布。

## 复现

在项目根目录运行，依赖使用现有 `requirements.txt`，无需安装上游的 LLM/评测框架。

```powershell
# 下载固定版本原始文件；存在时复用本地文件
.\.venv\Scripts\python.exe -X utf8 -m evaluation.download_crud

# 从原始文件生成固定 30 问、1000 篇新闻及来源标签
.\.venv\Scripts\python.exe -X utf8 -m evaluation.crud_dataset

# 调用真实 DashScope API，并使用 .env 配置的默认向量库（当前为 Milvus）
.\.venv\Scripts\python.exe -X utf8 -m evaluation.retrieval

# 可选择真实本地 Chroma；仍使用真实 DashScope embedding
.\.venv\Scripts\python.exe -X utf8 -m evaluation.retrieval --backend chroma --output output/evaluation/crud30_chroma

# 无网络的实现验证（含本地 Chroma 集成测试）
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 云端上传完成后，通过独立 HTTPS 接口回读并核对全部片段
.\.venv\Scripts\python.exe -X utf8 -m evaluation.verify_milvus
```

默认切片为 300 字符，重叠 50 字符；本语料共 2420 个片段。`--chunk-size`、`--chunk-overlap` 和 `--ks` 可用于后续实验。改变切片参数会建立不同评测集合，问题和语料固定。

首次评测时 Milvus 在 gRPC/TLS 握手阶段连接失败，因此 `output/evaluation/crud30/` 保存了真实 DashScope + 本地 Chroma 的基线。随后已成功将相同 1000 篇新闻、2420 个片段上传到云 Milvus 集合 `crud30_fa0f5a158c3251c51b43`，并重新执行全部 30 问。云端结果单独保存在 `output/evaluation/crud30_milvus/`，复现命令如下：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m evaluation.retrieval --backend milvus --output output/evaluation/crud30_milvus
```

向量化会使用 `.env` 中的 DashScope 凭证。默认模型为项目的 `text-embedding-v2`，1536 维，COSINE 检索。Milvus 使用单独的 `crud30_<fingerprint>` 集合；指纹包括数据、切片参数和 embedding 模型标识。文本新闻没有 PDF 页码，评测集合中的 `page_number` 表示文档序号，真实文档映射保存在报告目录的 `chunk_manifest.json`。

文档和问题向量按“模型标识 + document/query 类型 + 文本 SHA256”缓存到 `data/crud_rag/cache/embeddings.sqlite3`，避免重复付费并支持中断后续跑。短暂的网络错误、HTTP 429/5xx 最多重试三次；永久性鉴权/输入错误直接失败。缓存和下载文件均被 Git 忽略。重复运行仍会执行真实向量库检索。

## 输出与指标

默认输出到 `output/evaluation/crud30/`：

- `report.md`：总指标、分任务结果，以及全部 30 问、参考答案、遗漏文档和 Top-10 片段预览。
- `results.json`：配置、数据版本、逐问指标、完整片段文本、相似度、检索耗时和缓存状态。
- `chunk_manifest.json`：实际写入的 chunk ID 与参考/干扰文档的映射。

云端评测输出使用 `output/evaluation/crud30_milvus/`。`evaluation.verify_milvus` 只读核对云端集合结构、总数以及全部片段的 ID、文本、来源与文档序号，成功后保存 `verification.json`；认证信息不会写入该文件。

K 表示返回的 **chunk 数**，不是去重后的文档数。默认 K=1、3、5、10。

- **Hit@K**：前 K 个片段中至少一个来自参考文档的问题比例。
- **document Recall@K**：每个问题召回的不同参考文档数 / 该问题参考文档总数，然后对问题宏平均。相同文档的多个片段不会重复计数。
- **all evidence@K**：前 K 个片段覆盖所有参考文档的问题比例。
- **MRR@10**：前 10 个片段中第一个参考文档片段的排名倒数，未命中记 0，再宏平均。

相关性采用上游参考新闻的文档级银标。命中参考文档中的任意片段算命中文档，不能证明该片段包含完整答案；遗漏上游标签的其他相关报道可能使指标低估。参考答案不参与自动打分。当前不评测 LLM 生成质量，也不使用 BM25、reranker 或官方 RAGQuestEval。

## 来源说明

固定版本的上游目录未发现 LICENSE 文件；本地保留来源、版本及原始文件摘要，便于追溯。仓库提供下载和构建脚本，不包含上游新闻原文、问题和参考答案。

论文：Lyu et al., *CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models*, [DOI: 10.1145/3701228](https://doi.org/10.1145/3701228)。
