import os
import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

client = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-v4.1-flash"
)