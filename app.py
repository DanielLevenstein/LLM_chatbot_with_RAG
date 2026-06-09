import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
os.environ.setdefault("PYTHONPATH", ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import streamlit as st
from chatbot.chatbot import ChatBot


st.title("Streamlit Chatbot")
st.write("Welcome to the RAG-based chatbot! You can ask questions based on the context of provided documents.")


def log_timing(label, start_time, **fields):
    elapsed = time.perf_counter() - start_time
    extra_fields = " ".join(f"{key}={value}" for key, value in fields.items())
    message = f"TIMING {label} seconds={elapsed:.3f}"
    if extra_fields:
        message = f"{message} {extra_fields}"
    print(message, flush=True)
    return elapsed


@st.cache_resource
def get_chatbot():
    return ChatBot()

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model_status" not in st.session_state:
    st.session_state.model_status = None


# Input field for user queries
user_input = st.text_input(
    "Enter your question:",
    disabled=not st.session_state.model_loaded,
    placeholder="Load model first..." if not st.session_state.model_loaded else "",
)

load_model_column, ask_column  = st.columns([1, 1])

with load_model_column:
    load_model_clicked = st.button("Load Model")

with ask_column:
    ask_clicked = st.button("Ask Question", disabled=not st.session_state.model_loaded)

if load_model_clicked:
    with st.spinner("Loading model..."):
        try:
            load_start = time.perf_counter()
            chatbot = get_chatbot()
            chatbot.load_model()
            log_timing("app.load_model", load_start)
            st.session_state.model_loaded = True
            st.session_state.model_status = "Model loaded."
            st.rerun()
        except Exception as e:
            st.session_state.model_loaded = False
            st.session_state.model_status = None
            st.error(f"Model load failed: {e}")

if st.session_state.model_status:
    st.success(st.session_state.model_status)

# Button to submit the query
if ask_clicked:
    if user_input.strip():
        with st.spinner('Processing...'):
            try:
                request_start = time.perf_counter()
                chatbot_start = time.perf_counter()
                chatbot = get_chatbot()
                log_timing("app.get_chatbot", chatbot_start)
                rag_start = time.perf_counter()
                response = chatbot.ask_question_using_rag(user_input)
                log_timing("app.ask_question_using_rag", rag_start, question_chars=len(user_input))
                log_timing("app.request.total", request_start, question_chars=len(user_input))
                st.write("Chatbot's response:")
                st.write(response)
            except Exception as e:
                st.error(f"Request failed: {e}")
    else:
        st.write("Please enter a valid question.")
