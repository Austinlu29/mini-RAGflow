"""验证压缩阈值、整个请求预算、本地原文恢复和多轮 RAG 证据边界。"""

import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from context_management import ContextBudgetError, ContextManager, ContextSettings, TokenCounter
from context_management.compression import compress_text, select_level
from context_management.tokens import model_context_window
from llm.chat import DeepSeekChat
from RAG.qa import DEFAULT_NO_ANSWER_REPLY, QASettings, RAGQuestionAnswerer


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.folder = self.enterContext(tempfile.TemporaryDirectory())

    def manager(self, *, limit=10000, reserve=128, session="session1", recent=2):
        manager = ContextManager(ContextSettings(max_tokens=limit, output_reserve=reserve,
                                 recent_turns=recent, archive_dir=Path(self.folder)), session_id=session)
        self.addCleanup(manager.close)
        return manager

    def test_no_compression_below_thirty_percent_keeps_history(self):
        manager = self.manager()
        manager.record("论坛是谁主办的？", "协会和省委联合主办。")
        result = manager.prepare(system="规则", payload={"question": "它在哪里举办？"}, query="论坛")
        self.assertEqual(result.level, "none")
        self.assertEqual(result.payload["conversation"]["recent"][0]["assistant"], "协会和省委联合主办。")
        self.assertLess(result.tokens_after, 3000)
        self.assertEqual(manager.archive.count("session1"), 1)

    def test_all_thresholds_progressively_compact_without_losing_archived_original(self):
        for ratio, level, target in ((0.31, "light", 0.25), (0.51, "medium", 0.35),
                                      (0.71, "strong", 0.40), (0.91, "critical", 0.45),
                                      (1.10, "overflow", 0.50)):
            with self.subTest(level=level):
                manager = self.manager(session=level)
                original = "历史实体：山西安委办。" + "x" * int(10000 * ratio)
                manager.record("历史实体是什么？", original)
                result = manager.prepare(system="规则", payload={"question": "历史实体？"}, query="山西安委办")
                self.assertEqual(result.level, level)
                self.assertLessEqual(result.tokens_after, int(10000 * target))
                self.assertEqual(result.tokens_after, TokenCounter.messages("规则", result.user) + 128)
                restored = manager.archive.search(level, "山西安委办")
                self.assertEqual(restored[0]["assistant"], original)
                if level == "overflow":
                    self.assertTrue(result.recalled_turn_ids)

    def test_level_does_not_regress_after_strong_compression(self):
        manager = self.manager()
        manager.record("历史实体？", "a" * 7100)
        first = manager.prepare(system="规则", payload={"question": "问题"})
        self.assertEqual(first.level, "strong")
        second = manager.prepare(system="规则", payload={"question": "问题"})
        self.assertEqual(second.level, "none")
        self.assertEqual(manager.level, 2)

    def test_large_current_payload_is_archived_and_whole_request_is_below_half(self):
        manager = self.manager(limit=4000)
        payload = {"question": "a" * 9000, "references": [{"reference_id": "R1", "text": "b" * 9000}]}
        result = manager.prepare(system="不可更改的系统规则", payload=payload, query="a")
        self.assertEqual(result.level, "overflow")
        self.assertLessEqual(result.tokens_after, 2000)
        self.assertEqual(payload["question"], "a" * 9000)
        self.assertEqual(result.payload["references"][0]["reference_id"], "R1")
        blob = manager.archive.connection.execute("SELECT payload FROM packages WHERE kind='request'").fetchone()[0]
        self.assertEqual(json.loads(gzip.decompress(blob)), payload)

    def test_query_recovers_older_pack_and_never_reads_another_session(self):
        manager = self.manager(recent=1)
        manager.record("紫藤工程项目名称", "紫藤工程编号 8472。" + "a" * 3300)
        manager.prepare(system="规则", payload={"question": "另一个问题"})
        manager.record("新话题", "短回答")
        manager.prepare(system="规则", payload={"question": "新问题"})
        other = self.manager(session="other")
        other.record("紫藤工程", "另一个会话的私有内容")
        result = manager.prepare(system="规则", payload={"question": "紫藤工程编号？"}, query="紫藤工程")
        recalled = result.payload["conversation"]["recalled"]
        self.assertTrue(any("8472" in item["assistant"] for item in recalled))
        self.assertNotIn("私有内容", result.user)
        self.assertLess(result.tokens_after, manager.settings.max_tokens)

    def test_resuming_session_restores_state_and_does_not_duplicate_archives(self):
        settings = ContextSettings(max_tokens=10000, output_reserve=128, archive_dir=Path(self.folder))
        manager = ContextManager(settings, session_id="resume")
        manager.record("之前的问题", "原始回答" + "a" * 3500)
        manager.prepare(system="规则", payload={"question": "当前问题"})
        saved = manager._state()
        manager.close()
        resumed = self.manager(session="resume")
        self.assertEqual(resumed._state(), saved)
        self.assertEqual(resumed.archive.count("resume"), 1)
        new_session = self.manager(session="empty")
        self.assertFalse(new_session.has_history)

    def test_fixed_instruction_over_budget_fails_instead_of_sending_oversized_request(self):
        manager = self.manager(limit=1024)
        with self.assertRaises(ContextBudgetError):
            manager.prepare(system="a" * 1500, payload={"question": "问题"})

    def test_no_history_mode_does_not_disclose_previous_dialogue(self):
        manager = self.manager()
        manager.record("上一轮内容", "不应出现在无答案提示中")
        result = manager.prepare(system="规则", payload={"question": "新问题"}, include_history=False)
        self.assertNotIn("conversation", result.payload)
        self.assertNotIn("上一轮内容", result.user)

    def test_failed_local_write_does_not_change_memory(self):
        manager = self.manager()
        with patch.object(manager.archive, "save_turn", side_effect=OSError("disk failed")), self.assertRaises(OSError):
            manager.record("问题", "答案")
        self.assertFalse(manager.has_history)

    def test_configuration_and_model_capacity_validation(self):
        self.assertEqual(model_context_window("deepseek-flash"), 1048576)
        self.assertEqual(model_context_window("custom", "8192"), 8192)
        with self.assertRaises(ValueError):
            model_context_window("unknown")
        with self.assertRaises(ValueError):
            ContextSettings(max_tokens=1024, output_reserve=512)
        with self.assertRaises(ValueError):
            ContextManager(ContextSettings(archive_dir=Path(self.folder)), session_id="../bad")
        self.assertEqual([select_level(x) for x in (.29, .30, .50, .70, .90, 1.01)], [-1, 0, 1, 2, 3, 4])

    def test_chinese_compression_never_exceeds_budget_or_breaks_encoding(self):
        original = "论坛主办单位是协会。论坛地点是郑州。举办日期是八月八日。"
        compressed = compress_text(original, 40, "郑州")
        self.assertLessEqual(TokenCounter.text(compressed), 40)
        self.assertIn("郑州", compressed)
        self.assertNotIn("�", compressed)


class ConversationIntegrationTests(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        self.context = ContextManager(ContextSettings(max_tokens=20000, output_reserve=128,
                                      archive_dir=Path(folder)), session_id="rag")
        self.addCleanup(self.context.close)
        self.embedder, self.store, self.chat = Mock(), Mock(), Mock()
        self.store.search.return_value = [{"text": "论坛在郑州举办。", "similarity": 0.9,
                                          "metadata": {"source": "news.pdf", "page_number": 1}}]
        self.qa = RAGQuestionAnswerer(embedder=self.embedder, store=self.store, chat=self.chat,
                                     settings=QASettings(), context=self.context)

    def test_followup_rewrites_query_then_validates_current_rag_and_archives_original_question(self):
        self.context.record("哪个论坛由青年企业家协会主办？", "青年创新创业论坛。")
        self.chat.complete_json.side_effect = [
            {"question": "青年创新创业论坛在哪里举办？"},
            {"answerable": True, "quotes": [{"reference_id": "R1", "text": "论坛在郑州举办。"}]}]
        answer = self.qa.ask("它在哪里举办？")
        self.embedder.embed_query.assert_called_once_with("青年创新创业论坛在哪里举办？")
        self.assertIn("郑州", answer)
        first = json.loads(self.chat.complete_json.call_args_list[0].kwargs["user"])
        self.assertIn("青年创新创业论坛", first["conversation"]["recent"][0]["assistant"])
        self.assertEqual(self.context.recent[-1]["user"], "它在哪里举办？")
        self.assertEqual(self.context.archive.count("rag"), 2)

    def test_history_answer_is_never_accepted_as_rag_evidence(self):
        self.context.record("地点？", "论坛在上海举办。")
        self.chat.complete_json.side_effect = [
            {"question": "论坛在哪里举办？"},
            {"answerable": True, "quotes": [{"reference_id": "R1", "text": "论坛在上海举办。"}]},
            None]
        self.assertEqual(self.qa.ask("它的地点？"), DEFAULT_NO_ANSWER_REPLY)

    def test_quote_removed_from_sent_payload_is_rejected_even_if_present_in_original_rag(self):
        self.context.record("上一轮", "历史回答")
        original_prepare = self.context.prepare
        def prepare(**kwargs):
            result = original_prepare(**kwargs)
            if "references" in result.payload:
                result.payload["references"][0]["text"] = "论坛"
            return result
        self.chat.complete_json.side_effect = [
            {"question": "论坛在哪里举办？"},
            {"answerable": True, "quotes": [{"reference_id": "R1", "text": "论坛在郑州举办。"}]}, None]
        with patch.object(self.context, "prepare", side_effect=prepare):
            self.assertEqual(self.qa.ask("地点？"), DEFAULT_NO_ANSWER_REPLY)

    def test_chat_preflight_blocks_oversized_http_request(self):
        chat = DeepSeekChat(api_key="test", model="deepseek-flash", context_window=1024, max_tokens=64)
        with patch("llm.chat.httpx.Client") as http, self.assertRaises(ContextBudgetError):
            chat.complete_json(system="规则", user="x" * 1500)
        http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
