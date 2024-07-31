import streamlit as st
import requests

st.title("Mental Health Chatbot")

# RAG
st.header("Retrieval-Augmented Generation")
rag_input = st.text_input("Describe your mental health issue:")
if st.button("Get Help"):
    try:
        response = requests.get("http://localhost:8000/rag", params={"prompt": rag_input})
        response.raise_for_status()
        result = response.json()
        st.write("Chatbot Response:", result["response"])
        st.write("Relevant Article:", result["relevant_article"])
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        st.error("Make sure the FastAPI server is running on localhost:8000")

# Classification
st.header("Mental Health Classification")
classification_input = st.text_input("Enter text for classification:")
if st.button("Classify"):
    try:
        response = requests.get("http://localhost:8000/classification", params={"text": classification_input})
        response.raise_for_status()
        category = response.json()['category']
        st.write(f"Category: {category}")
        
        if category == "Suicidal":
            st.error("IMPORTANT: If you're having suicidal thoughts, please seek immediate professional help. Contact a suicide prevention hotline or emergency services.")
        elif category in ["Anxiety", "Depression"]:
            st.warning("It seems you might be experiencing some mental health challenges. Consider speaking with a mental health professional for support.")
        
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        st.error("Make sure the FastAPI server is running on localhost:8000")

st.info("Note: This is a demonstration and should not be used for actual mental health diagnosis. Always consult with a qualified mental health professional for accurate assessment and support.")