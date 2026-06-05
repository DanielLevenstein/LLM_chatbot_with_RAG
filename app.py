import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
os.environ.setdefault("PYTHONPATH", ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
import streamlit as st
from chatbot.chatbot import ChatBot


st.title("Streamlit Chatbot")
st.write("Welcome to the RAG-based chatbot! You can ask questions based on the context of provided documents.")

@st.cache_resource
def get_chatbot():
    return ChatBot()

# Input field for user queries
user_input = st.text_input("Enter your question:")

# Button to submit the query
if st.button("Ask"):
    if user_input.strip():
        with st.spinner('Processing...'):
            try:
                chatbot = get_chatbot()
                response = chatbot.ask_question_using_rag(user_input)
                st.write("Chatbot's response:")
                st.write(response)
            except Exception as e:
                st.error(f"Request failed: {e}")
    else:
        st.write("Please enter a valid question.")
