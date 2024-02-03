from openai import OpenAI
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import retriever
from langchain_community.llms import Ollama

llm = Ollama(model="mistral")

raw_documents = TextLoader("data/classschedule.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

embeddings = OllamaEmbeddings(model="mistral")
db = Chroma.from_documents(documents, embeddings)
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

st.title("🦜🔗 Langchain Quickstart App")


def generate_response(input_text):
    st.info(chain.run(input_text))


with st.form("my_form"):
    text = st.text_area("Enter text:", "What are 3 key advice for learning how to code?")
    submitted = st.form_submit_button("Submit")

    if submitted:
        generate_response(text)
