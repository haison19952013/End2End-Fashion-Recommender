import streamlit as st
import requests

def get_html_content(image_urls, scores):
    # Create a grid layout using HTML
    grid_html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;">'

    # Ensure the number of captions matches the number of images
    if len(image_urls) != len(scores):
        raise ValueError("The number of image URLs must match the number of captions.")

    # Add images and captions to the grid
    for url, score in zip(image_urls, scores):
        start = url.find('src="') + len('src="')
        end = url.find('"', start)
        url = url[start:end]
        caption = f"Score: {score:.4f}"
        grid_html += (
            f'<div style="text-align: center;">'
            f'<img src="{url}" style="height: 200px; width: 100%; object-fit: contain; border-radius: 8px;" />'
            f'<p style="margin-top: 5px; font-size: 14px;">{caption}</p>'
            f'</div>'
        )

    grid_html += '</div>'
    return grid_html


def main(uploaded_file, num_recommendations, make_recommendation):
    # Title of the app
    st.title("Welcome to Fashion Recommender App")

    if uploaded_file is not None:
        # Create a button to upload the file and get recommendations
        if make_recommendation:
            # Prepare the file and number of recommendations for upload
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            data = {"num_recommendations": num_recommendations}
            
            # Make a POST request to the FastAPI endpoint
            try:
                response = requests.post("http://my_api:8000/recsys", files=files, data=data) # Change the URL to the FastAPI endpoint
                # response = requests.post("http://localhost:8000/recsys", files=files, data=data) # For testing 
 
                # Check the response status and display the result
                if response.status_code == 200:
                    result = response.json()
                    recommendation = result["recommendation"]["recommendation"]
                    score = result["recommendation"]["score"]
                    if result["success"]:
                        st.success("Recommendations retrieved successfully!")
                        html_content = get_html_content(recommendation, score)
                        # print(html_content)
                        st.markdown(html_content, unsafe_allow_html=True)
                    else:
                        st.error("Error: " + result["message"])
                else:
                    st.error("File upload failed! " + response.text)
            except Exception as e:
                st.error(f"An error occurred: {e}")
