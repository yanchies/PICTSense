# Set up and run this Streamlit App
import streamlit as st
import pandas as pd
from logics.file_uploader import file_uploader
from logics.functions import process_responses

def main():
    # region <--------- Streamlit App Configuration --------->
    st.set_page_config(
        layout="centered",
        page_title="PICTSense"
    )
    # endregion <--------- Streamlit App Configuration --------->

    st.title("PICTSense - Analytical Tool for Open-Ended Responses")
       
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
            
            sentiment_bar = final_df["sentiment"].value_counts().sort_index()
            st.write("Sentiment Scores:")
            st.bar_chart(data=sentiment_bar, x_label= "Count", y_label="Sentiment Score", horizontal=True)
            
            topic_bar = final_df["topic"].value_counts()
            st.write("Topics:")
            st.bar_chart(data=topic_bar, x_label="Topic", y_label="Count")

            neg_issues = final_df[final_df['sentiment'].between(1, 4)]['topic'].value_counts()
            st.write("Top Negative Issues:")
            st.bar_chart(data=neg_issues.head(3), x_label="Topic", y_label="Count")

            pos_issues = final_df[final_df['sentiment'].between(6, 10)]['topic'].value_counts()
            st.write("Top Positive Issues:")
            st.bar_chart(data=pos_issues.head(3), x_label="Topic", y_label="Count")

    else:
        final_df = pd.read_json(st.session_state['json_file_path'])
        st.warning("File already uploaded.")
        st.write("Please refresh the app if you wish to upload a new file.")
        st.subheader("Dataframe")
        st.dataframe(final_df)

        # display overview information
        st.subheader("Overview")
        
        sentiment_bar = final_df["sentiment"].value_counts().sort_index()
        st.write("Sentiment Scores:")
        st.bar_chart(data=sentiment_bar, x_label= "Sentiment Score", y_label="Count", horizontal=True)
        
        topic_bar = final_df["topic"].value_counts()
        st.write("Topics:")
        st.bar_chart(data=topic_bar, x_label="Topic", y_label="Count")

        neg_issues = final_df[final_df['sentiment'].between(1, 4)]['topic'].value_counts()
        st.write("Top Negative Issues:")
        st.bar_chart(data=neg_issues.head(3), x_label="Topic", y_label="Count")

        pos_issues = final_df[final_df['sentiment'].between(6, 10)]['topic'].value_counts()
        st.write("Top Positive Issues:")
        st.bar_chart(data=pos_issues.head(3), x_label="Topic", y_label="Count")




if __name__ == "__main__":
    main()