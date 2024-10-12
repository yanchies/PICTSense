__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.text_splitter import RecursiveJsonSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import json
import streamlit as st
import shutil

splitter = RecursiveJsonSplitter(max_chunk_size=500)

def init_split(json_file_path):
    """Initialize document splitting from JSON file."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    documents = splitter.create_documents([data])
    
    # Check if the documents were created successfully
    if documents:
        st.write(f"Total documents created: {len(documents)}")
    return documents

embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

def create_vector_store(file, force_create=False):
    documents = init_split(file)

    # to delete the old vector store
    if db.check_persist_directory("./chroma_langchain_db"):
        shutil.rmtree("./chroma_langchain_db")
    
    # Create new vector store from documents
    st.write("Creating new vector store...")
    vector_store = Chroma.from_documents(
        collection_name="pictsense_store",
        documents=documents,
        embedding=embeddings_model,
        persist_directory="./chroma_langchain_db"  # Local persistence
    )
    st.write("Vector store created.")
  
    # Store the vector store in session_state
    st.session_state['vector_store'] = vector_store
    return vector_store

# load the vector store
db = Chroma("pictsense_store",
    embedding_function=embeddings_model,
    persist_directory= "./chroma_langchain_db")

qa_chain = RetrievalQA.from_chain_type(
        ChatOpenAI(model='gpt-4o-mini'),
        retriever=db.as_retriever(k=5),
        return_source_documents=True,
        chain_type="stuff"
    )
