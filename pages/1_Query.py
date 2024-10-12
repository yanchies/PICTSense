import streamlit as st
import pandas as pd
from logics.functions import get_df
from logics.rag import qa_chain, db, create_vector_store
import json
from Main import json_file_path

st.title("Query Page")

if 'file' not in st.session_state:
    st.warning("Please upload a .csv file in the Main page.")

# Check if the file exists in session state
else:
    st.subheader("Dataframe")
    # Load the JSON content
    st.dataframe(pd.read_json(json_file_path))

    st.subheader("Document Query Interface")
    
    if 'vector_store' not in st.session_state:
        st.write("Creating vector store...")
        create_vector_store(json_file_path)
    else:
        st.write("Loading vector store from session state...")
        
    st.write(f"There are {db._collection.count()} documents in the database.")
    user_query = st.text_input("Enter your query:")

    if user_query:
        # Perform query against the documents
        try:
            result = qa_chain.invoke(user_query)
            st.write(f"**Query:** {user_query}")
            st.subheader("**Answer:**")
            st.write(f"{result['result']}")
            st.subheader(f"**Sources:** ")
            if result['result'] != "I don't know.":
                ids = []
                for i, doc in enumerate(result["source_documents"]):
                    content = json.loads(doc.page_content)
                    
                    for id, response_data in content.items():
                        if not response_data.get("response"):
                            continue
                        if id in ids:
                            continue
                        ids.append(id)
                        st.write(f"**Source {len(ids)}:**")
                        response_text = response_data["response"]
                        sentiment = response_data["sentiment"]
                        topic = response_data.get("topic", "NA")
                        
                        st.write(f"**{id}:** {response_text}")
                        st.write(f"**Sentiment Score:** {sentiment}")
                        st.write(f"**Topic:** {topic}")
                        st.write(f"**Raw:** {response_data}")
                        st.divider()
                    
        except Exception as e:
            st.error(f"Error retrieving answer: {e}")






