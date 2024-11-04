import streamlit as st
import pandas as pd
from logics.rag import create_vector_store
import json
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

st.title("Query Page")

if 'file' not in st.session_state:
    st.warning("Please upload a .csv file in the Main page.")

else:
    st.subheader("Dataframe")
    json_file_path = st.session_state['json_file_path']
    st.dataframe(pd.read_json(json_file_path))

    st.subheader("Response Query Interface")
    if 'vector_store' not in st.session_state:
        create_vector_store(json_file_path)
    else:
        st.write("Loading vector store from session state...")
    
    chroma_db = st.session_state['vector_store']
    st.write(f"There are {len(chroma_db.get()['documents'])} documents in the database.")
    
    # use RetrievalQA to enable Document Query
    qa_chain = RetrievalQA.from_chain_type(
        ChatOpenAI(model='gpt-4o-mini'),
        retriever=chroma_db.as_retriever(k=5),
        return_source_documents=True,
        chain_type="stuff"
    )

    user_query = st.text_input("Enter your query:")

    if user_query:
        result = qa_chain.invoke(user_query)
        st.write(f"**Query:** {user_query}")
        st.subheader("**Answer:**")
        st.write(f"{result['result']}")
        st.subheader(f"**Sources:** ")

        if result['source_documents']:
            for source in result['source_documents']:
                data = json.loads(source.page_content)
                reponse_id = data["response_id"]
                response = data["response"]
                sentiment = data["sentiment"]
                topic = data["topic"]
                st.write(f"**Response ID:** {reponse_id}")
                st.write(f"**Response:** {response}")
                st.write(f"**Sentiment Score:** {sentiment}")
                st.write(f"**Topic:** {topic}")
                st.divider()

        else:
            st.write("No relevant sources found.")






