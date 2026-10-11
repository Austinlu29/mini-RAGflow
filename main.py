"""默认启动 DeepSeek RAG 问答；也支持显式入库、检索和连接检查。"""

import argparse
from pathlib import Path
import sys

from context_management import ContextManager
from RAG.pipeline import index_pdf
from RAG.qa import QASettings, RAGQuestionAnswerer
from embedding.embed import DashScopeEmbedder
from llm.chat import DeepSeekChat
from vectorstore.factory import create_vector_store


def main():
    parser = argparse.ArgumentParser(description="MiniRAGFlow DeepSeek RAG 问答客户端")
    parser.add_argument("--pdf", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "test.pdf")
    parser.add_argument("--backend", choices=("milvus", "chroma"), default=None)
    parser.add_argument("--session", help="上下文会话 ID，使用相同 ID 恢复本地历史")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--query", help="只查询已有向量，不重新解析或向量化 PDF")
    actions.add_argument("--index", action="store_true", help="显式将 PDF 向量化并上传到配置的向量库")
    actions.add_argument("--ask", help="单次 RAG 问答；没有依据时由 AI 生成无答案说明")
    actions.add_argument("--check", action="store_true", help="只检查向量库连接与片段数量")
    args = parser.parse_args()
    if args.query:
        embedder = DashScopeEmbedder()
        store = create_vector_store(embedding_model=embedder.embedding_id, backend=args.backend)
        try:
            results = store.search(embedder.embed_query(args.query))
            for result in results:
                metadata = result["metadata"]
                print(f"相似度 {result['similarity']:.4f} | 第 {metadata['page_number']} 页 | {metadata['source']}")
                print(result["text"])
                print()
            if not results:
                print("没有找到匹配片段")
        finally:
            if hasattr(store, "close"):
                store.close()
    elif args.index:
        ids = index_pdf(args.pdf, backend=args.backend)
        print(f"已向量化并保存 {len(ids)} 个片段到配置的向量库")
    elif args.check:
        embedder = DashScopeEmbedder()
        store = create_vector_store(embedding_model=embedder.embedding_id, backend=args.backend)
        try:
            count = store.count() if hasattr(store, "count") else store.collection.count()
            print(f"已连接向量库，当前 {count} 个片段；未上传文档")
        finally:
            if hasattr(store, "close"):
                store.close()
    else:
        settings = QASettings.from_env()
        chat = DeepSeekChat()
        embedder = DashScopeEmbedder()
        store = create_vector_store(embedding_model=embedder.embedding_id, backend=args.backend,
                                    collection_name=settings.collection_name,
                                    persist_directory=Path(__file__).resolve().parent / "data" / "chroma")
        context = None
        try:
            context = ContextManager.from_env(chat=chat, session_id=args.session)
            qa = RAGQuestionAnswerer(embedder=embedder, store=store, chat=chat, settings=settings, context=context)
            if args.ask is not None:
                answer = qa.ask(args.ask)
                if answer:
                    print(f"AI：{answer}", flush=True)
            else:
                # 提示和回答使用同一输出流，避免 IDE 将下一轮提示排在回答前。
                print("RAG 问答已启动；输入 exit / quit / 退出 结束。",
                      flush=True)
                if context is not None:
                    print(f"会话 ID：{context.session_id}（可用 --session 恢复）；输入 /context 查看上下文。", flush=True)
                while True:
                    print("你：", end="", flush=True)
                    try:
                        question = input()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if question.strip().lower() in ("exit", "quit", "退出"):
                        break
                    if not question.strip():
                        continue
                    if question.strip() == "/context" and context is not None:
                        report = context.last_report
                        if report:
                            print(f"上下文预算估计：{report.tokens_after}/{context.settings.max_tokens} tokens；"
                                  f"最近压缩级别：{report.level}；本地归档 {context.archive.count(context.session_id)} 轮。",
                                  flush=True)
                        else:
                            print("还没有模型请求；对话原文将打包保存在本地。", flush=True)
                        continue
                    try:
                        answer = qa.ask(question)
                    except Exception:
                        print("问答请求失败，请检查网络、API 配置和向量库连接。", file=sys.stderr)
                        continue
                    if answer:
                        print(f"AI：{answer}\n", flush=True)
        finally:
            try:
                if context is not None:
                    context.close()
            finally:
                if hasattr(store, "close"):
                    store.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception:
        print("客户端运行失败，请检查 .env 配置、依赖、网络和向量库连接。", file=sys.stderr)
        raise SystemExit(1) from None
