import streamlit as st
st.title('Methodology')
st.write("""
         The application contains two key use cases: (1) Sentiment Analysis & Topic Identification; and (2) Response Query.
         See below for the process flows of the two use cases.   
         """)
st.write("""In summary, the application utilises various LLM-based techniques such as Prompt Engineering, use of embeddings, 
         Retrieval-Augment Generation (RAG), and RetrievalQA to enable its features.""")

st.subheader("Process Flow for Sentiment Analysis & Topic Identification")
# pass in csv file
# convert to json
# pass json to LLM for sentiment analysis and embedding for topic identification
# collate new inputs and update json
# convert json to dataframe for visualisation and analytics

st.divider()
st.subheader("Process Flow for Response Query")
# use RecursiveJsonSplitter to split the json file into documents
# create vector database
# use RetrievalQA to enable querying
# display results and relevant source documents