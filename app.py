import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
os.environ.setdefault("PYTHONPATH", ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import streamlit as st
from chatbot.chatbot import ChatBot


st.title("Streamlit Chatbot")
st.write("Welcome to the RAG-based chatbot! You can ask questions based on the context of provided documents.")

# Cache the ChatBot initialization - this ensures the model is only loaded once per session
@st.cache_resource
def initialize_chatbot():
    chatbot = ChatBot()
    chatbot._get_llm()  # Pre-initialize the LLM client
    return chatbot

# Initialize session state for tracking user action
if "load_clicked" not in st.session_state:
    st.session_state.load_clicked = False

# Load model button
if st.button("Load Model"):
    st.session_state.load_clicked = True

# Only initialize chatbot if user has clicked load
if st.session_state.load_clicked:
    with st.spinner("Loading model... This may take a minute."):
        try:
            chatbot = initialize_chatbot()
            st.success("✓ Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.session_state.load_clicked = False

# Input field for user queries (disabled until model is loaded)
user_input = st.text_input(
    "Enter your question:",
    disabled=not st.session_state.load_clicked,
    placeholder="Load model first..." if not st.session_state.load_clicked else ""
)

# Button to submit the query (disabled until model is loaded)
if st.button("Ask", disabled=not st.session_state.load_clicked):
    if user_input.strip():
        # Create placeholder for status updates
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        try:
            status_placeholder.info("🔍 Retrieving relevant context from knowledge base...")
            chatbot = initialize_chatbot()
            
            status_placeholder.info("⏳ Generating response from model...")
            response = chatbot.ask_question_using_rag(user_input)
            
            status_placeholder.empty()
            response_placeholder.write("Chatbot's response:")
            response_placeholder.write(response)
        except Exception as e:
            status_placeholder.empty()
            response_placeholder.error(f"Request failed: {e}")
            import traceback
            st.error(f"Error details: {traceback.format_exc()}")
    else:
        st.write("Please enter a valid question.")
