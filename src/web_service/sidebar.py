
import streamlit as st
def login():
    with st.sidebar:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login")
        
        if login_button:
            if not st.session_state.logged_in:
                if username == "admin" and password == "admin":
                    st.success(f"Welcome, {username}!")
                    st.sidebar.success("You are logged in.")
                    st.session_state.logged_in = True
                else:
                    st.error("Invalid username or password")
                    st.sidebar.error("Login failed")
            else:
                st.sidebar.info("You are already logged in.")

def upload_image():
    uploaded_file = None
    num_recommendations = None
    make_recommendation = None
    
    with st.sidebar:
        uploaded_file = st.file_uploader(label = "Upload your image....", type=["jpg", "jpeg", "png"], help = "A full body image will be more helpful")
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            num_recommendations = st.number_input("Number of recommendations", min_value=1, value=10)
            make_recommendation = st.button("Get Recommendations", use_container_width = True)
            return uploaded_file, num_recommendations, make_recommendation
    return uploaded_file, num_recommendations, make_recommendation
        