__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.text_splitter import RecursiveJsonSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import json
import streamlit as st

# """  
# This file contains the RAG-based functions.   
# """ 

splitter = RecursiveJsonSplitter(max_chunk_size=1000)

def init_split(json_file_path):
    """Initialize document splitting from JSON file."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    print(data)
    documents = splitter.create_documents(data)
    return documents

embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

def create_vector_store(file):
    chroma_db = Chroma(persist_directory="./chroma_langchain_db",
        embedding_function=embeddings_model,
        collection_name="pictsense_store")
    collection = chroma_db.get()
    
    # look for existing collection and delete if it exists
    try:
        if len(collection['ids']) > 0:
            chroma_db.delete_collection()
            st.write("Collection deleted.")
    except Exception as e:
        st.write(f"An error occurred while trying to delete the collection: {e}")
    
    st.write("Creating vector store...")
    documents = init_split(file)

    # Create new vector store from documents
    chroma_db = Chroma.from_documents(
        collection_name="pictsense_store",
        documents=documents,
        embedding=embeddings_model,
        persist_directory="./chroma_langchain_db"  # Local persistence
    )
    chroma_db.persist()
    st.write("Vector store created.")
    st.session_state['vector_store'] = chroma_db
  
    return chroma_db




