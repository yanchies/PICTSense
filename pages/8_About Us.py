import streamlit as st
st.title('About this App')

st.write("This is a Streamlit App that utilises LLM-based methods to analyse open-ended survey responses.")

st.subheader("Project Scope & Objectives")
st.write("""
         The project aims to support analysis of large volumes of open-ended survey responses through sentiment analysis
         and topic identification to reduce manual overheads.
         """)

with st.expander("Features"):
    st.write("Sentiment Analysis: Individual responses will be passed into a prompt to generate a sentiment score from 1 to 10.")
    st.write("Topic Identification: Individual responses will be embedded to identify the most relevant topic from a list of 9 topics.")
    st.write("Responses Query: The responses can be queried to generate a response based on RAG.")

with st.expander("How to use this App", False):
    st.write("1. Upload a .csv (utf-8) file containing survey responses.")
    st.write("2. Wait for the app to input sentiment scores and a topic for each response and provide an overview of the results")
    st.write("3. Query the new responses to extract relevant insights.")