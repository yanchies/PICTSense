import streamlit as st
st.title('About this App')

st.write("This is a Streamlit App that utilises LLM-based approaches to analyse open-ended survey responses.")

with st.expander("How to use this App", False):
    st.write("1. Upload a .csv file containing survey responses.")
    st.write("2. Wait for the app to input sentiment scores and a topic for each response.")
    st.write("3. Query the new document to extract relevant insights.")