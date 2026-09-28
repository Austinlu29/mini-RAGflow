from langchain_community.document_loaders import WebBaseLoader
import bs4
# from langchain_text_splitters import  RecursiveCharacterTextSplitter


web_path = "https://www.nju.edu.cn/ndgk/ndjj.htm"

loader = WebBaseLoader(
    web_path = [web_path],
    bs_kwargs = dict(parse_only = bs4.SoupStrainer(class_=("ndjj","title")))
)
docs = loader.load()
print(docs)
exit()