# Set up and run this Streamlit App
import streamlit as st
import pandas as pd
from logics.file_uploader import file_uploader
from logics.functions import process_responses, get_df

def main():
    # region <--------- Streamlit App Configuration --------->
    st.set_page_config(
        layout="centered",
        page_title="PICTSense"
    )
    # endregion <--------- Streamlit App Configuration --------->

    st.title("PICTSense - Analytical Tool for Open-Ended Responses")
    # file uploader widget
    
    # file = st.file_uploader(label="Upload a .csv file", type=['csv'])
    
    if 'file' not in st.session_state:
        file = st.file_uploader(label="Upload a .csv file", type=['csv'])
        
        if file is not None:
            csv_file_path, json_file_path = file_uploader(file)
            st.success("File uploaded successfully!")
            st.session_state['file'] = json_file_path
            st.divider()
            
            st.subheader("Dataframe")
            df = pd.read_csv(csv_file_path)  # Load your CSV file
            json_file_path = process_responses(df, json_file_path)
            st.session_state['json_file_path'] = json_file_path
            # df = get_df(json_file_path)
            st.dataframe(pd.read_json(json_file_path))
    else:
        # Message to show if no file is uploaded
        st.warning("File already uploaded.")
        st.write("Please refresh the app if you wish to upload a new file.")




if __name__ == "__main__":
    main()