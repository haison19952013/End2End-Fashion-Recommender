import streamlit as st
from src.web_service.sidebar import login, upload_image
from src.web_service.main_page import main

st.set_page_config(
    page_title="Fashion Recommender System",
    layout="wide",
    page_icon=":womans_clothes:",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
login()
if st.session_state.logged_in:
    uploaded_file, num_recommendations, make_recommendation = upload_image()
    main(uploaded_file, num_recommendations, make_recommendation)

else:
    st.warning("Please login to continue")
    print('This is for testing Jenkins')
