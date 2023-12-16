import pandas as pd
from langchain import OpenAI, FAISS, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv() 

def create_vector_db_from_pdf(pdf_url: str) -> FAISS:
    loader = PyPDFLoader(pdf_url)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)

    vector_db = FAISS.from_documents(loader.load_and_split(text_splitter), OpenAIEmbeddings())

    return vector_db

# url = 'https://hr.umich.edu/sites/default/files/uofm_cb_ppo_baag_2023.pdf'
# print(create_vector_db_from_pdf(url))

def get_response_from_query(db, query):
    model_name = "gpt-3.5-turbo-instruct"
    llm = OpenAI(model = model_name, temperature = 0)

    prompt_template = """You are an insurance consultant. You are tasked with answering basic questions on insurance forms.
    You are provided with a question and a summary benefits coverage form that defines the benefits on a specific medical plan offered through an employer. Be polite with your responses.

    Context: {context}

    Question: {question}

    Your answer:
    """

    prompt = PromptTemplate(template = prompt_template, input_variables = ["question", "context"])
    chain = RetrievalQA.from_llm(llm = llm, retriever=db.as_retriever(), prompt=prompt)

    return chain.run({"query": query})

