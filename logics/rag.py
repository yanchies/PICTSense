from langchain.text_splitter import RecursiveJsonSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import json
import streamlit as st

splitter = RecursiveJsonSplitter(max_chunk_size=200)

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

def create_vector_store(file):
    # Initialize the vector store only if it's not already in session_state
    if 'vector_store' not in st.session_state:
        # Split the documents
        documents = init_split(file)
        
        # Create the vector store from documents
        vector_store = Chroma.from_documents(
            collection_name="pictsense_store",
            documents=documents,
            embedding=embeddings_model,
            persist_directory="./chroma_langchain_db"  # Local persistence
        )
        
        # Store the vector store in session_state
        st.session_state['vector_store'] = vector_store
        st.write("Vector store created and stored in session state.")
    else:
        # Retrieve from session_state if it already exists
        vector_store = st.session_state['vector_store']
        st.write("Vector store loaded from session state.")
    
    return vector_store

if not "./chroma_langchain_db":
    st.write("Creating vector store... Please upload a .csv file in the Main page.")

else:
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
