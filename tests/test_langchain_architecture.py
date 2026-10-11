"""验证 LangChain 原生模型、LCEL 和既有存储的集成，不调用真实 API。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pymupdf
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableSequence

from context_management import ContextManager, ContextSettings
from document.element import Block
from document.langchain import LayoutDocumentSplitter, LayoutPDFLoader, block_to_document
from embedding.embed import DashScopeEmbedder
from llm.chains import StrictJsonOutputParser, create_json_chain
from llm.chat import DeepSeekChat
from RAG.pipeline import create_indexing_chain
from RAG.qa import DEFAULT_NO_ANSWER_REPLY, QASettings, RAGQuestionAnswerer
from vectorstore.chromadb import ChromaStore
from vectorstore.langchain import ChunkVectorStore, EvidenceRetriever


class TestEmbeddings(Embeddings):
    embedding_id = "langchain-test"

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class Events(BaseCallbackHandler):
    def __init__(self):
        self.events = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.events.append(("chat", kwargs.get("tags", [])))

    def on_retriever_start(self, serialized, query, **kwargs):
        self.events.append(("retriever", kwargs.get("tags", [])))


class LangChainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.embeddings = TestEmbeddings()
        self.store = ChromaStore(embedding_model=self.embeddings.embedding_id,
                                 persist_directory=str(Path(self.temp.name) / "chroma"))
        self.addCleanup(self.store.client._system.stop)
        self.vectorstore = ChunkVectorStore(store=self.store, embeddings=self.embeddings)

    @staticmethod
    def document(text, source="manual.pdf", page=2, block_type="Paragraph"):
        return block_to_document(Block(text, block_type, page, 10, 20, 100, 200), source=source)

    def model(self):
        return DeepSeekChat(api_key="test-only-key", base_url="https://api.deepseek.com",
                            model="deepseek-flash")

    def transport(self, responses, requests=None):
        responses = iter(responses)

        def handler(request):
            if requests is not None:
                requests.append(json.loads(request.content))
            body, finish = next(responses)
            return httpx.Response(200, json={"choices": [{"finish_reason": finish,
                "message": {"role": "assistant", "content": body}}]})

        # 每次请求使用独立客户端，与真实运行中的生命周期一致。
        client_class = httpx.Client
        return patch("llm.chat.httpx.Client", side_effect=lambda **kwargs:
                     client_class(transport=httpx.MockTransport(handler), **kwargs))

    def test_layout_splitter_keeps_table_title_and_metadata(self):
        documents = [self.document("123456789"), self.document("long title", block_type="Title"),
                     self.document("cell1 | cell2\nvalue1 | value2", block_type="Table")]
        documents[0].metadata["custom"] = "retained"
        splitter = LayoutDocumentSplitter(chunk_size=6, chunk_overlap=2)
        result = splitter.transform_documents(documents)
        self.assertEqual([doc.page_content for doc in result],
                         ["123456", "56789", documents[1].page_content, documents[2].page_content])
        self.assertEqual(result[1].metadata, documents[0].metadata)
        result[0].metadata["x0"] = -1
        self.assertEqual(documents[0].metadata["x0"], 10)
        self.assertEqual(documents[0].page_content, "123456789")

    def test_real_pdf_loader_preserves_page_and_absolute_source(self):
        path = Path(self.temp.name) / "layout.pdf"
        with pymupdf.open() as pdf:
            for text in ("First page paragraph.", "Second page paragraph."):
                pdf.new_page().insert_text((72, 72), text)
            pdf.save(path)
        documents = LayoutPDFLoader(path).load()
        self.assertEqual({doc.metadata["page_number"] for doc in documents}, {1, 2})
        self.assertTrue(all(doc.metadata["source"] == str(path.resolve()) for doc in documents))
        self.assertTrue(all("x0" in doc.metadata and "block_type" in doc.metadata for doc in documents))

    def test_adapter_reads_existing_vectors_and_preserves_cosine_threshold(self):
        ids = self.store.add_chunks([Block("matching", "Paragraph", 2, 0, 0, 1, 1),
                                     Block("opposite", "Paragraph", 2, 0, 0, 1, 1)],
                                    [[1, 0], [-1, 0]], source="existing.pdf")
        documents = EvidenceRetriever(vectorstore=self.vectorstore).invoke("question")
        self.assertEqual([doc.id for doc in documents], [ids[0]])
        self.assertEqual(documents[0].metadata["source"], "existing.pdf")
        scores = self.vectorstore.similarity_search_with_relevance_scores("question")
        self.assertEqual([score for _, score in scores], [1.0, 0.0])
        self.assertEqual(self.vectorstore.as_retriever(search_kwargs={"k": 1}).invoke("question")[0].id, ids[0])

    def test_add_documents_replaces_source_without_touching_other_sources(self):
        ids = self.vectorstore.add_documents([self.document("old"), self.document("stale")])
        self.vectorstore.add_documents([self.document("other", source="other.pdf")])
        updated = self.vectorstore.add_documents([self.document("new")])
        self.assertEqual(updated, ids[:1])
        self.assertEqual(self.store.collection.count(), 2)
        self.assertEqual(self.vectorstore.similarity_search("question", source="manual.pdf")[0].page_content, "new")

    def test_mixed_sources_and_missing_pages_fail_before_embedding_or_write(self):
        embeddings = Mock()
        adapter = ChunkVectorStore(store=self.store, embeddings=embeddings)
        for documents in ([self.document("one"), self.document("two", source="other.pdf")],
                          [Document(page_content="no page", metadata={"source": "manual.pdf"})]):
            with self.subTest(documents=documents), self.assertRaises(ValueError):
                adapter.add_documents(documents)
        embeddings.embed_documents.assert_not_called()
        self.assertEqual(self.store.collection.count(), 0)

    def test_native_model_prompt_parser_and_secret_exclusion(self):
        model = self.model()
        self.assertNotIn("test-only-key", repr(model))
        self.assertNotIn("test-only-key", json.dumps(model.model_dump(), default=str))
        self.assertTrue(issubclass(DashScopeEmbedder, Embeddings))
        chain = create_json_chain(model, 'Return JSON: {"question": "..."}')
        self.assertIsInstance(chain, RunnableSequence)
        requests = []
        with self.transport([('{"question": "hello"}', "stop")], requests):
            result = chain.invoke({"user": "hello"})
        self.assertEqual(result, {"question": "hello"})
        self.assertEqual(requests[0]["messages"][0]["content"], 'Return JSON: {"question": "..."}')

    def test_parser_rejects_markdown_partial_json_and_completed_but_truncated_object(self):
        parser = StrictJsonOutputParser()
        for content in ('```json\n{}\n```', '{"key":', "[]", "null"):
            with self.subTest(content=content):
                self.assertIsNone(parser.invoke(content))
        self.assertIsNone(parser.invoke(AIMessage(content='{"question": "hello"}',
                                                  response_metadata={"finish_reason": "length"})))

    def test_native_rag_chain_emits_callbacks_and_only_archives_validated_answer(self):
        self.vectorstore.add_documents([self.document("Trial lasts 30 days.")])
        context = ContextManager(ContextSettings(archive_dir=Path(self.temp.name) / "history"), session_id="native")
        self.addCleanup(context.close)
        qa = RAGQuestionAnswerer(embedder=self.embeddings, store=self.store, chat=self.model(),
                                 settings=QASettings(), context=context)
        events = Events()
        quote = json.dumps({"answerable": True, "quotes": [{"reference_id": "R1", "text": "Trial lasts 30 days."}]})
        with self.transport([(quote, "stop")]):
            result = qa.invoke("How long is the trial?", config={"callbacks": [events], "tags": ["test-run"]})
        self.assertEqual(result, "Trial lasts 30 days.\n[来源：manual.pdf；第 2 页]")
        self.assertEqual(context.archive.count("native"), 1)
        self.assertEqual({name for name, _ in events.events}, {"chat", "retriever"})
        self.assertTrue(all("test-run" in tags for _, tags in events.events))

    def test_native_chain_rejects_fabrication_and_uses_no_answer_branch(self):
        self.vectorstore.add_documents([self.document("Trial lasts 30 days.")])
        qa = RAGQuestionAnswerer(embedder=self.embeddings, store=self.store, chat=self.model(), settings=QASettings())
        forged = json.dumps({"answerable": True, "quotes": [{"reference_id": "R1", "text": "Trial lasts 60 days."}]})
        with self.transport([(forged, "stop"), ("{}", "stop")]):
            self.assertEqual(qa.chain.invoke("How long is the trial?"), DEFAULT_NO_ANSWER_REPLY)

    def test_indexing_chain_skips_empty_pdf_and_closes_store_on_write_failure(self):
        model = Mock(embedding_id="test-model")
        chain = create_indexing_chain(embedder=model)
        with patch("RAG.pipeline.LayoutPDFLoader") as loader, patch("RAG.pipeline.create_vector_store") as factory:
            loader.return_value.load.return_value = []
            self.assertEqual(chain.invoke("empty.pdf"), [])
            model.embed_documents.assert_not_called()
            factory.assert_not_called()
            loader.return_value.load.return_value = [self.document("body")]
            model.embed_documents.return_value = [[1, 0]]
            factory.return_value.add_chunks.side_effect = RuntimeError("write failed")
            with self.assertRaises(RuntimeError):
                chain.invoke("body.pdf")
            factory.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
