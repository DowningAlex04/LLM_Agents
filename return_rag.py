import json
import os

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.tools import tool
GEMINI_KEY = os.environ.get('GEMINI_API_KEY')
return_policy_document = 'return_policy_document.md'

def create_retriever():

    with open(return_policy_document) as file:
        policy_document = file.read()

    headers_to_split_on = [('#', 'Header 1'), ('##','Header 2')]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
