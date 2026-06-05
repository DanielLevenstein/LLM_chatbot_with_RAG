import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import streamlit as st
from huggingface_hub import hf_hub_download
from chatbot.chatbot import ChatBot


st.title("Streamlit Chatbot")
st.write("Welcome to the RAG-based chatbot! You can ask questions based on the context of provided documents.")

# Create an instance of the ChatBot class
chatbot = ChatBot()

# Input field for user queries
user_input = st.text_input("Enter your question:")

# Button to submit the query
if st.button("Ask"):
    if user_input.strip():
        with st.spinner('Processing...'):
            response = chatbot.ask_question_using_rag(user_input)
        st.write("Chatbot's response:")
        st.write(response)
    else:
        st.write("Please enter a valid question.")
