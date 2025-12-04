import json
import os

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

GEMINI_KEY = os.environ.get('GEMINI_API_KEY')

#tip: print the key
#print(GEMINI_KEY)

plant_database_file = 'plants.json'

def plant_json_to_text(plant):

    return f"""
    Plant name: {plant['name']} 
    Description: {plant['description']} 
    Scientific name: {plant['scientific_name']}
    Care, light levels: {plant['care']['light']}
    Care, water needs: {plant['care']['water']}
    Care, soil: {plant['care']['soil']}
    Care, prefered temperature and humidity: {plant['care']['temperature_and_humidity']}
    Care, tips: {plant['care']['tips']}
    """

def create_retriver():
    data = json.load(open(plant_database_file))
    texts=[plant_json_to_text(plant) for plant in data]
    ids= [plant['id'] for plant in data]

    documents = [Document(page_content = text, id=id) for text, id in zip (texts, ids)]
    gemini_embeddigs = GoogleGenerativeAIEmbeddings(model = 'models /embedding-001', google_api_key=GEMINI_KEY)
    vector_store = Chroma.from_documents(
        documents=documents, 
        embedding=gemini_embeddigs, # how to create embeddings for each document
        persist_directory='./plant_embeddings'

    )

    query = 'What is a plant with intresting leaves?'

    results = vector_store.max_marginal_relevance_search(query=query, k=2)
    for res in results:
        print(res.page_content)
        
    retriever = vector_store.as_retriever(search_type = 'mmr',search_kwargs={'k':5,'fetch_k':15})
    return retriever

def create_rag_chain(retreiver):
    rag_prompt = PromptTemplate(
        template="""
    Answer the customer's question using the context provided.
    Reply 'I dont know' if the context cant be used to answer the customers question.
    if the content doesnt directly answer the question, reply with whatever you do know
    from the context that may be related

    use this context to answer the question:
    {context} 

    Customer question:
    {question}

    Response:
    """, 
    input_variables=['context', 'question']
    )

    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',temperature=0.5)

    plant_rag_chain =(
        {'context': retreiver, 'question': RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return plant_rag_chain

@tool
def plant_rag(customer_query):
    """
    Answer general questions about plant care and plant attributes for plants we sell
    and, find plants that match customer requirements, for example, care levels, size, ammount of sun/light, water, soil
    """
    return plant_rag_chain.invoke(customer_query)

document_retriver = create_retriver()
plant_rag_chain = create_rag_chain(document_retriver)

# customer_query = 'Does a spider plant like humidity?'
# rag_result = plant_rag.invoke(customer_query)
# print(rag_result)

