# Set up and run this Streamlit App
import streamlit as st
import pandas as pd
from logics.file_uploader import file_uploader
from logics.functions import process_responses, get_df
__import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def main():
    # region <--------- Streamlit App Configuration --------->
    st.set_page_config(
        layout="centered",
        page_title="PICTSense"
    )
    # endregion <--------- Streamlit App Configuration --------->

    st.title("PICTSense - Analytical Tool for Open-Ended Responses")
    # file uploader widget
    file = st.file_uploader(label="Upload a .csv file", type=['csv'])
    
    if file is not None:
        csv_file_path, json_file_path = file_uploader(file)
        if 'file' not in st.session_state:
            st.session_state['file'] = json_file_path
        
        st.success("File uploaded successfully!")
        st.divider()
        df = pd.read_csv(csv_file_path)  # Load your CSV file
        json_file_path = process_responses(df, json_file_path)

        st.subheader("Dataframe")
        st.dataframe(get_df(json_file_path))        

        # result = run_crew(file_path)
        # #st.subheader("Task Output 1")
        # #st.write(result.tasks_output[0].raw)
        # #st.write(result.tasks_output[0])
        # st.subheader("Task Output 2")
        # st.write(result.tasks_output[1].raw)
        # st.write(result.tasks_output[1])
        # st.subheader("Task Output 3")
        # st.write(result.tasks_output[2].raw)
        # st.write(result.tasks_output[2])
        # st.subheader("Task Output 4")
        # st.write(result.tasks_output[3].raw)
    
    else:
        # Message to show if no file is uploaded
        st.warning("Please upload a CSV file to continue.")



if __name__ == "__main__":
    main()