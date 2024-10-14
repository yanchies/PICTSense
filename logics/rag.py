__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.text_splitter import RecursiveJsonSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import json
import streamlit as st
from chromadb import PersistentClient

splitter = RecursiveJsonSplitter(max_chunk_size=500)

def init_split(json_file_path):
    """Initialize document splitting from JSON file."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    print(data)
    documents = splitter.create_documents(data)
    return documents

embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

def create_vector_store(file):
    st.write("Creating vector store...")
    # look for existing collection and delete if it exists
    collection_name = "pictsense_store"
    chroma_client = PersistentClient(path="./chroma_langchain_db")
    
    try:
        # Check if the collection exists
        if chroma_client.path.joinpath(collection_name).exists():
            chroma_client.delete_collection(collection_name)
            st.write(f"Collection '{collection_name}' deleted successfully.")
        else:
            st.write(f"No existing collection named '{collection_name}' found. No deletion needed.")
    except FileNotFoundError:
        st.write(f"No collection found at {chroma_client.path}. No deletion needed.")
    except Exception as e:
        st.write(f"An error occurred while trying to delete the collection: {e}")
        # Optionally re-raise if you want to stop execution here
        # raise Exception(f"Failed to delete collection: {e}")
    
    documents = init_split(file)

    # Create new vector store from documents
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
