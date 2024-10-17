import streamlit as st  
import random  
import hmac  

# """  
# This file contains the common components used in the Streamlit App.  
# This includes the file uploader and the password check.  
# """ 

def file_uploader(file):
    csv_file_path = file.name
    # Save uploaded file directly
    with open(csv_file_path, "wb") as f:
        f.write(file.getbuffer())

    # Return the path for processing
    json_file_path = csv_file_path.replace('.csv', '.json')
    return csv_file_path, json_file_path
    
def check_password():  
    """Returns `True` if the user had the correct password."""  
    def password_entered():  
        """Checks whether a password entered by the user is correct."""  
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):  
            st.session_state["password_correct"] = True  
            del st.session_state["password"]  # Don't store the password.  
        else:  
            st.session_state["password_correct"] = False  
    # Return True if the passward is validated.  
    if st.session_state.get("password_correct", False):  
        return True  
    # Show input for password.  
    st.text_input(  
        "Password", type="password", on_change=password_entered, key="password"  
    )  
    if "password_correct" in st.session_state:  
        st.error("😕 Password incorrect")  
    return False