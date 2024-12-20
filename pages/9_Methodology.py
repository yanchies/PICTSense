import streamlit as st
st.title('Methodology')
st.write("""
         The application contains two key use cases: (1) Sentiment Analysis & Topic Identification; and (2) Response Query.
         See below for the consolidated process flows of the two use cases.   
         """)
st.write("""In summary, the application utilises various LLM-based techniques such as Prompt Engineering, use of embeddings, 
         Retrieval-Augment Generation (RAG), RetrievalQA, and summarisation to enable its features.""")

st.subheader("Process Flow")
st.image("./helper_functions/pictsense.drawio.png")

