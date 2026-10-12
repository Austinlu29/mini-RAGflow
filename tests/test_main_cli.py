"""验证问答、连接检查，以及显式入库、查询的边界，不访问真实 API。"""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import main
from vectorstore.factory import create_vector_store
from RAG.qa import NO_ANSWER_NOTICES, NO_ANSWER_SUGGESTIONS, QASettings


class MainCliTests(unittest.TestCase):
    def setUp(self):
        self.index = self.enterContext(patch("main.index_pdf"))
        self.model_constructor = self.enterContext(patch("main.DashScopeEmbedder"))
        self.factory = self.enterContext(patch("main.create_vector_store"))
        self.chat_constructor = self.enterContext(patch("main.DeepSeekChat"))
        self.context_constructor = self.enterContext(patch("main.ContextManager.from_env", return_value=None))
        self.enterContext(patch("main.QASettings.from_env", return_value=QASettings()))
        self.model = self.model_constructor.return_value
        self.model.embedding_id = "test-model"
        self.model.embed_query.return_value = [1.0, 0.0]
        self.store = self.factory.return_value
        self.store.count.return_value = 38
        self.output = self.enterContext(redirect_stdout(io.StringIO()))
        self.errors = self.enterContext(redirect_stderr(io.StringIO()))

    def run_cli(self, *args):
        with patch("sys.argv", ["main.py", *args]):
            main.main()

    def test_check_only_connects_counts_and_closes(self):
        self.run_cli("--check")
        self.factory.assert_called_once_with(embedding_model="test-model", backend=None)
        self.store.count.assert_called_once_with()
        self.store.close.assert_called_once_with()
        self.store.search.assert_not_called()
        self.index.assert_not_called()
        self.model.embed_query.assert_not_called()
        self.model.embed_documents.assert_not_called()
        self.assertIn("38", self.output.getvalue())
        self.assertIn("未上传文档", self.output.getvalue())

    def test_index_is_explicit_and_passes_pdf_and_backend(self):
        self.index.return_value = ["chunk-1", "chunk-2"]
        self.run_cli("--index", "--pdf", "custom.pdf", "--backend", "milvus")
        self.index.assert_called_once_with(Path("custom.pdf"), backend="milvus")
        self.factory.assert_not_called()
        self.model_constructor.assert_not_called()
        self.assertIn("2", self.output.getvalue())

    def test_query_embeds_question_and_searches_without_indexing(self):
        self.store.search.return_value = [{
            "text": "a matching paragraph", "similarity": 0.75,
            "metadata": {"source": "source.pdf", "page_number": 2},
        }]
        self.run_cli("--query", "文档讲了什么", "--backend", "milvus")
        self.factory.assert_called_once_with(embedding_model="test-model", backend="milvus")
        self.model.embed_query.assert_called_once_with("文档讲了什么")
        self.store.search.assert_called_once_with([1.0, 0.0])
        self.store.close.assert_called_once_with()
        self.store.count.assert_not_called()
        self.index.assert_not_called()
        self.model.embed_documents.assert_not_called()
        self.assertIn("a matching paragraph", self.output.getvalue())

    def test_index_and_query_are_mutually_exclusive_before_any_api_call(self):
        with self.assertRaises(SystemExit) as error:
            self.run_cli("--index", "--query", "question")
        self.assertEqual(error.exception.code, 2)
        self.index.assert_not_called()
        self.factory.assert_not_called()
        self.model_constructor.assert_not_called()

    def test_connection_count_failure_closes_client_without_uploading(self):
        self.store.count.side_effect = RuntimeError("database unavailable")
        with self.assertRaises(RuntimeError):
            self.run_cli("--check")
        self.store.close.assert_called_once_with()
        self.index.assert_not_called()
        self.model.embed_query.assert_not_called()
        self.model.embed_documents.assert_not_called()

    def test_query_failure_closes_client(self):
        self.store.search.side_effect = RuntimeError("search unavailable")
        with self.assertRaises(RuntimeError):
            self.run_cli("--query", "question")
        self.store.close.assert_called_once_with()
        self.index.assert_not_called()

    def test_removed_backend_is_rejected_before_any_api_call(self):
        with self.assertRaises(SystemExit) as error:
            self.run_cli("--check", "--backend", "chroma")
        self.assertEqual(error.exception.code, 2)
        self.factory.assert_not_called()
        self.index.assert_not_called()
        self.model_constructor.assert_not_called()

    def matching_hit(self):
        return {"text": "MiniRAGFlow 支持 PDF 知识库问答。", "similarity": 0.85,
                "metadata": {"source": "guide.pdf", "page_number": 2}}

    def test_default_starts_qa_and_exit_closes_without_indexing(self):
        self.store.search.return_value = [self.matching_hit()]
        self.chat_constructor.return_value.complete_json.return_value = {
            "answerable": True, "quotes": [{"reference_id": "R1",
                                            "text": "MiniRAGFlow 支持 PDF 知识库问答。"},
                                           {"reference_id": "R1", "text": "PDF 知识库问答"}]}
        with patch("builtins.input", side_effect=["有哪些功能？", "exit"]):
            self.run_cli()
        self.model.embed_query.assert_called_once_with("有哪些功能？")
        self.store.search.assert_called_once_with([1.0, 0.0], top_k=5)
        self.chat_constructor.return_value.complete_json.assert_called_once()
        self.assertIn("guide.pdf", self.output.getvalue())
        output = self.output.getvalue()
        self.assertEqual(output.count("AI："), 1)
        self.assertEqual(output.count("[来源："), 1)
        self.assertLess(output.index("你："), output.index("AI："))
        self.assertLess(output.index("[来源："), output.rindex("你："))
        self.store.close.assert_called_once_with()
        self.index.assert_not_called()

    def test_ask_without_evidence_prints_ai_generated_no_answer(self):
        self.store.search.return_value = []
        self.chat_constructor.return_value.complete_json.return_value = {
            "notice": NO_ANSWER_NOTICES[1], "suggestion": NO_ANSWER_SUGGESTIONS[1]}
        self.run_cli("--ask", "库外问题")
        self.assertEqual(self.output.getvalue(), f"AI：{NO_ANSWER_NOTICES[1]}{NO_ANSWER_SUGGESTIONS[1]}\n")
        self.assertEqual(self.errors.getvalue(), "")
        self.chat_constructor.return_value.complete_json.assert_called_once()
        self.store.close.assert_called_once_with()

    def test_qa_collection_is_forwarded_without_changing_management_modes(self):
        self.store.search.return_value = []
        with patch("main.QASettings.from_env", return_value=QASettings(collection_name="qa_news")):
            self.run_cli("--ask", "问题")
        self.assertEqual(self.factory.call_args.kwargs["collection_name"], "qa_news")

    def test_interactive_invalid_answer_prints_no_answer_then_accepts_next_question(self):
        self.store.search.return_value = [self.matching_hit()]
        self.chat_constructor.return_value.complete_json.side_effect = [
            {"answerable": True, "quotes": [{"reference_id": "R1", "text": "模型编造内容"}]},
            {"notice": NO_ANSWER_NOTICES[0], "suggestion": NO_ANSWER_SUGGESTIONS[0]},
            {"answerable": True, "quotes": [{"reference_id": "R1",
                                             "text": "MiniRAGFlow 支持 PDF 知识库问答。"}]},
        ]
        with patch("builtins.input", side_effect=["问题1", "问题2", "退出"]):
            self.run_cli()
        self.assertNotIn("模型编造内容", self.output.getvalue())
        self.assertIn(NO_ANSWER_NOTICES[0], self.output.getvalue())
        self.assertEqual(self.output.getvalue().count("AI："), 2)
        self.assertEqual(self.output.getvalue().count("[来源："), 1)

    def test_session_error_does_not_print_sensitive_exception_and_can_exit(self):
        self.store.search.side_effect = RuntimeError("secret-token")
        with patch("builtins.input", side_effect=["问题", "quit"]):
            self.run_cli()
        self.assertNotIn("AI：", self.output.getvalue())
        self.assertNotIn("secret-token", self.errors.getvalue())
        self.assertIn("问答请求失败", self.errors.getvalue())
        self.store.close.assert_called_once_with()

    def test_eof_interrupt_and_blank_questions_do_not_call_apis(self):
        for ending in (EOFError, KeyboardInterrupt):
            with self.subTest(ending=ending), patch("builtins.input", side_effect=[" ", ending]):
                self.run_cli()
        self.model.embed_query.assert_not_called()
        self.chat_constructor.return_value.complete_json.assert_not_called()
        self.assertEqual(self.store.close.call_count, 2)

    def test_one_shot_failure_closes_and_never_prints_answer(self):
        self.store.search.side_effect = RuntimeError("search failed")
        with self.assertRaises(RuntimeError):
            self.run_cli("--ask", "问题")
        self.assertEqual(self.output.getvalue(), "")
        self.store.close.assert_called_once_with()

    def test_session_argument_is_forwarded_and_context_closes_on_error(self):
        context = Mock()
        context.has_history = False
        self.context_constructor.return_value = context
        self.store.search.side_effect = RuntimeError("search failed")
        with self.assertRaises(RuntimeError):
            self.run_cli("--session", "work1", "--ask", "问题")
        self.context_constructor.assert_called_once_with(chat=self.chat_constructor.return_value, session_id="work1")
        context.record.assert_not_called()
        context.close.assert_called_once_with()
        self.store.close.assert_called_once_with()

    def test_ask_and_other_actions_are_mutually_exclusive(self):
        for args in (("--ask", "问题", "--index"), ("--ask", "问题", "--check"),
                     ("--ask", "问题", "--query", "问题")):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.run_cli(*args)
        self.factory.assert_not_called()
        self.chat_constructor.assert_not_called()


class VectorStoreFactoryTests(unittest.TestCase):
    def setUp(self):
        # 保持配置测试完全独立，不读取项目 .env 或连接已有数据库。
        self.enterContext(patch("vectorstore.factory.load_dotenv"))
        self.milvus = self.enterContext(patch("vectorstore.milvus.MilvusStore"))
        self.enterContext(patch.dict("os.environ", {}, clear=True))

    def test_default_selects_cloud_store(self):
        result = create_vector_store(embedding_model="test-model")
        self.assertIs(result, self.milvus.return_value)
        self.milvus.assert_called_once_with(embedding_model="test-model", collection_name=None)

    def test_explicit_cloud_backend_overrides_environment(self):
        with patch.dict("os.environ", {"VECTOR_STORE": "unsupported"}):
            result = create_vector_store(embedding_model="test-model", backend="milvus",
                                         collection_name="cloud_chunks")
        self.assertIs(result, self.milvus.return_value)
        self.milvus.assert_called_once_with(embedding_model="test-model", collection_name="cloud_chunks")

    def test_environment_is_used_when_backend_is_omitted(self):
        with patch.dict("os.environ", {"VECTOR_STORE": "MILVUS"}):
            result = create_vector_store(embedding_model="test-model")
        self.assertIs(result, self.milvus.return_value)
        self.milvus.assert_called_once_with(embedding_model="test-model", collection_name=None)

    def test_invalid_backend_is_rejected_without_opening_store(self):
        for backend in ("chroma", "unsupported", ""):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                create_vector_store(embedding_model="test-model", backend=backend)
            with patch.dict("os.environ", {"VECTOR_STORE": backend}), self.assertRaises(ValueError):
                create_vector_store(embedding_model="test-model")
        self.milvus.assert_not_called()


if __name__ == "__main__":
    unittest.main()
