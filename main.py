# Set up and run this Streamlit App
import streamlit as st
import pandas as pd
from logics.file_uploader import file_uploader
from logics.functions import process_responses, visualise

def main():
    # region <--------- Streamlit App Configuration --------->
    st.set_page_config(
        layout="centered",
        page_title="PICTSense"
    )
    # endregion <--------- Streamlit App Configuration --------->

    st.title("PICTSense - Analytical Tool for Open-Ended Responses")

    st.expander("""
    IMPORTANT NOTICE: This web application is developed as a proof-of-concept prototype. The information provided here is NOT intended for actual usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.
    Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.
    Always consult with qualified professionals for accurate and personalized advice.
    """)
       
    if 'file' not in st.session_state:      
        file = st.file_uploader(label="Upload a .csv file", type=['csv'])
        if file is not None:
            csv_file_path, json_file_path = file_uploader(file)
            st.session_state['file'] = json_file_path
            st.success("File uploaded successfully!")
            st.divider()
            
            st.subheader("Dataframe")
            raw_df = pd.read_csv(csv_file_path)  # Load your CSV file
            json_file_path = process_responses(raw_df, json_file_path)
            st.session_state['json_file_path'] = json_file_path
            final_df = pd.read_json(json_file_path)
            
            # display dataframe
            st.dataframe(final_df)

            # display overview information
            st.subheader("Overview")
            visualise(final_df)

    else:
        final_df = pd.read_json(st.session_state['json_file_path'])
        st.warning("File already uploaded.")
        st.warning("Please refresh the app if you wish to upload a new file.")
        st.subheader("Dataframe")
        st.dataframe(final_df)
        st.divider()

        # display overview information
        st.subheader("Overview")
        visualise(final_df)
        
        




if __name__ == "__main__":
    main()