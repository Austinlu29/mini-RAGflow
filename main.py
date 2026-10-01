from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import models
from models import *
dotenv.load_dotenv()

deepseek = models.client

loader = Docx2txtLoader(r"E:\example.docx")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=[
        '\n\n\n\n',
        '\n\n\n',
        '\n\n',
        '\n',
        ' ',
        '',
    ]
)
documents = text_splitter.split_documents(docs)

embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

db = Chroma.from_documents(
    collection_name='demo',
    documents=documents,
    embedding=embeddings,
)

#检索
docs_find = RunnableLambda(db.similarity_search).bind(k=2)


message = """
仅使用提供的上下文回答下面的问题
{question}
上下文：
{context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("human",message)
])

chain = {
    "question":RunnablePassthrough(),
    "context":docs_find,} | prompt_template | deepseek

response = chain.invoke("快速品应用场景有哪些")
print(response.content)