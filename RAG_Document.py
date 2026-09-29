import os
# from pandas.io.sas.sas_constants import page_count_length
# from langchain_community.document_loaders import WebBaseLoader
# import bs4
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from models import *
from langchain_chroma import Chroma
from langchain_core.documents import Document
import dotenv

dotenv.load_dotenv()

# web_path = "https://www.chinanews.com.cn/cj/2026/09-29/10705466.shtml"
#
#
# loader = WebBaseLoader(
#     web_path = [web_path],
#     bs_kwargs = dict(parse_only = bs4.SoupStrainer(class_=("content","title")))
# )
# docs = loader.load()
#
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 500,    #文档块大小
#     chunk_overlap=50,    #允许重叠大小
#     separators=[
#         '\n\n',
#         '\n',
#         ' ',
#         ''
#     ]                    #分割符
# )
#  documents = splitter.split_documents(docs)

embeddings = DashScopeEmbeddings(
    model = "text-embedding-v2",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
)
#评分方式
score_measures = [
    "default",  #default = 'l2'
    "cosine",   #cosine  #余弦相似度
    "l2",       #l2      #欧式距离
    "ip"        #ip      #点积
]

db = Chroma(
    collection_name="collection_name",
    embedding_function=embeddings,
    persist_directory="./chroma_db1",
)

documents = [
    Document(page_content = "这个苹果手机很好用"),
    Document(page_content = "我国山东盛产苹果")
]

print('-' * 100)
db.add_documents(documents, ids=["apple-phone", "shandong-apple"])

results = db.similarity_search_with_score("我想买个手机")
for doc,score in results:
    print(doc.page_content,end='\t')
    print(f"score: {score}")

