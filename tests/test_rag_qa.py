"""严格检索问答的依据校验与 DeepSeek 协议测试，不访问真实 API。"""

import json
import unittest
from unittest.mock import Mock, patch

import httpx

from llm.chat import DeepSeekChat
from RAG.qa import (DEFAULT_NO_ANSWER_REPLY, NO_ANSWER_NOTICES, NO_ANSWER_SUGGESTIONS,
                    QASettings, RAGQuestionAnswerer)


class RAGQATests(unittest.TestCase):
    def setUp(self):
        self.embedder, self.store, self.chat = Mock(), Mock(), Mock()
        self.embedder.embed_query.return_value = [1.0, 0.0]
        self.hit = {"text": "产品支持 PDF 问答。试用期限为 30 天。", "similarity": 0.8,
                    "metadata": {"source": "manual.pdf", "page_number": 3}}
        self.store.search.return_value = [self.hit]
        self.qa = RAGQuestionAnswerer(embedder=self.embedder, store=self.store,
                                     chat=self.chat, settings=QASettings())

    def answer(self, text="试用期限为 30 天。", ref="R1"):
        return {"answerable": True, "quotes": [{"reference_id": ref, "text": text}]}

    def test_success_only_outputs_exact_quote_and_database_citation(self):
        self.chat.complete_json.return_value = self.answer()
        self.assertEqual(self.qa.ask("  试用多久？  "),
                         "试用期限为 30 天。\n[来源：manual.pdf；第 3 页]")
        self.embedder.embed_query.assert_called_once_with("试用多久？")
        prompt = json.loads(self.chat.complete_json.call_args.kwargs["user"])
        self.assertEqual(prompt["references"], [{"reference_id": "R1", "text": self.hit["text"]}])
        self.assertNotIn("manual.pdf", self.chat.complete_json.call_args.kwargs["user"])

    def test_no_matches_or_low_invalid_scores_call_ai_for_no_answer(self):
        for hits in ([], [dict(self.hit, similarity=0.64)],
                     [dict(self.hit, similarity=float("nan"))],
                     [dict(self.hit, similarity=float("inf"))],
                     [dict(self.hit, similarity=1.5)], [dict(self.hit, similarity=True)],
                     [dict(self.hit, similarity="0.9")]):
            with self.subTest(hits=hits):
                self.store.search.return_value = hits
                self.chat.complete_json.reset_mock()
                self.chat.complete_json.return_value = {"notice": NO_ANSWER_NOTICES[1],
                                                        "suggestion": NO_ANSWER_SUGGESTIONS[1]}
                self.assertEqual(self.qa.ask("问题"), NO_ANSWER_NOTICES[1] + NO_ANSWER_SUGGESTIONS[1])
                self.chat.complete_json.assert_called_once()
                payload = json.loads(self.chat.complete_json.call_args.kwargs["user"])
                self.assertEqual(payload["question"], "问题")
                self.assertNotIn("references", payload)

    def test_invalid_source_empty_text_and_missing_page_are_not_evidence(self):
        for hit in (dict(self.hit, text=" "), dict(self.hit, metadata={}),
                    dict(self.hit, metadata={"source": "manual.pdf", "page_number": 0}),
                    dict(self.hit, metadata={"source": "", "page_number": 1}), None):
            with self.subTest(hit=hit):
                self.store.search.return_value = [hit]
                self.chat.complete_json.reset_mock()
                self.assertEqual(self.qa.ask("问题"), DEFAULT_NO_ANSWER_REPLY)
                self.chat.complete_json.assert_called_once()

    def test_refusal_bad_schema_and_fabrications_generate_no_answer_reply(self):
        cases = [None, {}, {"answerable": False, "quotes": []},
                 {"answerable": "true", "quotes": self.answer()["quotes"]},
                 {"answerable": True, "quotes": []},
                 self.answer("试用期限为 60 天。"), self.answer(ref="R99"),
                 self.answer(" "), self.answer(text=None), self.answer(ref=[]),
                 dict(self.answer(), extra="模型添加解释"),
                 {"answerable": True, "quotes": [{"reference_id": "R1", "text": "试用期限为 30 天。",
                                                   "source": "伪造来源.pdf"}]},
                 {"answerable": True, "quotes": "原文"},
                 {"answerable": True, "quotes": [None]},
                 {"answerable": True, "quotes": self.answer()["quotes"] * 9}]
        for result in cases:
            with self.subTest(result=result):
                self.chat.complete_json.reset_mock()
                self.chat.complete_json.side_effect = [result, {
                    "notice": NO_ANSWER_NOTICES[2], "suggestion": NO_ANSWER_SUGGESTIONS[0]}]
                self.assertEqual(self.qa.ask("问题"), NO_ANSWER_NOTICES[2] + NO_ANSWER_SUGGESTIONS[0])
                self.assertEqual(self.chat.complete_json.call_count, 2)

    def test_one_fabricated_quote_discards_all_content_without_partial_answer(self):
        result = self.answer()
        result["quotes"].append({"reference_id": "R1", "text": "库外补充"})
        self.chat.complete_json.return_value = result
        self.assertEqual(self.qa.ask("问题"), DEFAULT_NO_ANSWER_REPLY)

    def test_multiple_sources_duplicate_quotes_and_similarity_boundary(self):
        self.store.search.return_value = [dict(self.hit, similarity=0.65),
            {"text": "咨询电话是 12345。", "similarity": 0.9,
             "metadata": {"source": "contact.pdf", "page_number": 7}}]
        result = self.answer()
        result["quotes"].extend([result["quotes"][0],
                                {"reference_id": "R2", "text": "咨询电话是 12345。"}])
        self.chat.complete_json.return_value = result
        answer = self.qa.ask("试用期限和咨询电话？")
        self.assertEqual(answer.count("试用期限为 30 天。"), 1)
        self.assertIn("contact.pdf；第 7 页", answer)

    def test_context_budget_excludes_unseen_text(self):
        self.qa.settings = QASettings(max_context_chars=10)
        self.chat.complete_json.return_value = self.answer()
        self.assertEqual(self.qa.ask("试用多久？"), DEFAULT_NO_ANSWER_REPLY)
        prompt = json.loads(self.chat.complete_json.call_args_list[0].kwargs["user"])
        self.assertEqual(prompt["references"][0]["text"], self.hit["text"][:10])

    def test_blank_input_and_errors_do_not_return_a_fallback(self):
        self.assertEqual(self.qa.ask(" "), "")
        self.embedder.embed_query.assert_not_called()
        self.store.search.side_effect = RuntimeError("db error")
        with self.assertRaises(RuntimeError):
            self.qa.ask("问题")
        self.chat.complete_json.assert_not_called()

    def test_no_answer_generation_failure_or_fabrication_uses_safe_notice(self):
        self.store.search.return_value = []
        for result in (None, {}, {"notice": "当前知识库检索结果中未找到这个问题的答案。真实答案为 60 天。",
                                  "suggestion": NO_ANSWER_SUGGESTIONS[0]},
                       {"notice": NO_ANSWER_NOTICES[0], "suggestion": "伪造来源.pdf"},
                       {"notice": NO_ANSWER_NOTICES[0], "suggestion": NO_ANSWER_SUGGESTIONS[0],
                        "answer": "60 天"}):
            with self.subTest(result=result):
                self.chat.complete_json.return_value = result
                self.assertEqual(self.qa.ask("试用多久？"), DEFAULT_NO_ANSWER_REPLY)
        self.chat.complete_json.side_effect = RuntimeError("secret-api-error")
        self.assertEqual(self.qa.ask("试用多久？"), DEFAULT_NO_ANSWER_REPLY)

    def test_settings_reject_bad_ranges_and_environment(self):
        for kwargs in ({"top_k": 0}, {"top_k": True}, {"min_similarity": -0.1},
                       {"min_similarity": float("nan")}, {"max_context_chars": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QASettings(**kwargs)
        with patch("RAG.qa.load_dotenv"), patch.dict("os.environ", {
                "RAG_TOP_K": "4", "RAG_MIN_SIMILARITY": "0.75", "RAG_MAX_CONTEXT_CHARS": "2000",
                "RAG_COLLECTION": "qa_news"}):
            self.assertEqual(QASettings.from_env(), QASettings(4, 0.75, 2000, "qa_news"))
        with patch("RAG.qa.load_dotenv"), patch.dict("os.environ", {"RAG_TOP_K": "bad"}), self.assertRaises(ValueError):
            QASettings.from_env()

    def test_news_sources_use_document_numbers_instead_of_pdf_pages(self):
        self.hit["metadata"]["source"] = "crud-rag:news123"
        self.chat.complete_json.return_value = self.answer()
        self.assertIn("文档序号 3", self.qa.ask("试用多久？"))


class DeepSeekChatTests(unittest.TestCase):
    def model(self, **kwargs):
        defaults = {"api_key": "test-only-key", "base_url": "https://api.deepseek.com", "model": "deepseek-flash"}
        return DeepSeekChat(**(defaults | kwargs))

    def complete(self, body=None, status=200, finish="stop"):
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
            self.assertEqual(payload["model"], "deepseek-flash")
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            self.assertFalse(payload["stream"])
            self.assertEqual(request.headers["Authorization"], "Bearer test-only-key")
            return httpx.Response(status, json={"choices": [{"finish_reason": finish,
                "message": {"role": "assistant", "content": body}}]})
        client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("llm.chat.httpx.Client", return_value=client):
            return self.model().complete_json(system="Return JSON.", user="问题")

    def test_valid_json_protocol(self):
        self.assertEqual(self.complete('{"answerable": false, "quotes": []}'),
                         {"answerable": False, "quotes": []})

    def test_empty_invalid_or_truncated_output_is_never_accepted(self):
        for body in (None, "", "not json", "```json\n{}\n```", "[]"):
            with self.subTest(body=body):
                self.assertIsNone(self.complete(body))
        self.assertIsNone(self.complete("{}", finish="length"))
        self.assertIsNone(self.complete("{}", finish="content_filter"))

    def test_failure_does_not_expose_response_body(self):
        with self.assertRaises(RuntimeError) as error:
            self.complete("sensitive-key-echo", status=401)
        self.assertIn("401", str(error.exception))
        self.assertNotIn("sensitive-key-echo", str(error.exception))

    def test_timeout_has_safe_error(self):
        def handler(request):
            raise httpx.ReadTimeout("secret", request=request)
        client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("llm.chat.httpx.Client", return_value=client), self.assertRaises(RuntimeError) as error:
            self.model().complete_json(system="JSON", user="问题")
        self.assertNotIn("secret", str(error.exception))

    def test_environment_config_and_validation(self):
        with patch("llm.chat.load_dotenv"), patch.dict("os.environ", {
                "DEEPSEEK_API_KEY": "env-key", "DEEPSEEK_MODEL": "deepseek-flash",
                "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1", "DEEPSEEK_TIMEOUT": "12"}):
            model = DeepSeekChat()
            self.assertEqual(model.endpoint, "https://api.deepseek.com/v1/chat/completions")
            self.assertEqual(model.timeout, 12)
        for kwargs in ({"api_key": ""}, {"api_key": "your-api-key"},
                       {"base_url": "http://api.deepseek.com"},
                       {"base_url": "https://user:password@api.deepseek.com"},
                       {"base_url": "https://api.deepseek.com?key=secret"},
                       {"model": "deepseek-falsh"}, {"timeout": "nan"},
                       {"timeout": 0}, {"max_tokens": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model(**kwargs)


if __name__ == "__main__":
    unittest.main()
