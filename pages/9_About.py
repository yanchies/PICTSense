import streamlit as st
st.title('About this App')

st.write("This is a Streamlit App that demonstrates how to use the OpenAI\
          API to generate text completions.")

with st.expander("How to use this App", False):
    st.write("1. Enter your prompt in the text area.")
    st.write("2. Click the 'Submit' button.")
    st.write("3. The app will generate a text completion based on your prompt.")