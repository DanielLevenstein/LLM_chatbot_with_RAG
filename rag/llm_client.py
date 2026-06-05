from datetime import datetime
import pickle
import faiss
import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer

from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError


CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50

SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_PATH = "index/index.faiss"
CHUNKS_PATH = "index/chunks.pkl"

TOP_K = 4
# Model Options for MODEL_PATH = "unsloth/Qwen3-4B-GGUF"
# Qwen3-4B-BF16.gguf, 8.05 GB
# Qwen3-4B-Q8_0.gguf, 4.28 GB, 53 s
# Qwen3-4B-Q6_K.gguf, 3.31 GB
# Qwen3-4B-Q5_K_M.gguf, 2.85 GB

MODEL_PATH = "unsloth/Qwen3-4B-GGUF"
MODEL_FILENAME = "Qwen3-4B-Q5_K_M.gguf"
_client = None
_index = None
_chunks = None
_embed_model = None


def get_llm_client():
    global _client
    if _client is None:
        _client = create_llm(MODEL_PATH, MODEL_FILENAME)
    return _client


def download_model(model_name_or_path, model_basename, max_retries=3, initial_backoff=5):
    for attempt in range(1, max_retries + 1):
        try:
            return hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
        except HfHubHTTPError as e:
            status = getattr(e, "status_code", None)
            if attempt == max_retries or status != 502:
                raise
            print(f"Hugging Face download failed with status {status}. Retrying {attempt}/{max_retries}...")
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"Model download failed: {e}. Retrying {attempt}/{max_retries}...")
        time.sleep(initial_backoff * attempt)


def load_rag_assets():
    global _index, _chunks, _embed_model
    if _index is None or _chunks is None or _embed_model is None:
        print("CWD:", os.getcwd())
        _index = faiss.read_index(INDEX_PATH)

        with open(CHUNKS_PATH, "rb") as f:
            _chunks = pickle.load(f)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embed_model = SentenceTransformer(SENTENCE_TRANSFORMER, device=device)

    return _index, _chunks, _embed_model


def create_llm(model_name_or_path, model_basename):
    print(f"Creating model: {MODEL_FILENAME}")
    start_time = datetime.now()
    # Using hf_hub_download to download a model from the Hugging Face model hub
    # The repo_id parameter specifies the model name or path in the Hugging Face repository
    # The filename parameter specifies the name of the file to download
    model_path = download_model(
        model_name_or_path,
        model_basename,
    )
    print(f"Model path: {model_path}")
    print(os.path.exists(model_path))
    print(os.path.getsize(model_path))

    llm = Llama(
        model_path=model_path,
        n_threads=2,  # CPU cores
        n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_gpu_layers=43,  # Change this value based on your model and your GPU VRAM pool.
        n_ctx=4096,  # Context window
    )
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Model download to {model_path}\n Download time: {elapsed_time}")
    return llm


# Working.
def generate_response_without_context(llm, instruction: str, question: str) -> str:
    start_time = datetime.now()
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        max_tokens=512,
        temperature=0.0,
        top_p=0.95,
        repeat_penalty=1.2,
    )

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"generate_response_without_context -Elapsed time: {elapsed_time}")
    return trim_response(response["choices"][0]["message"]["content"])


def generate_response_with_context(llm, instruction: str, context: str, question: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "system",
                "content": f"Retrieved context:\n{context}",
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        max_tokens=512,
        temperature=0.0,
        top_p=0.95,
        repeat_penalty=1.2,
    )
    return trim_response(response["choices"][0]["message"]["content"])


def generate_response_using_rag(llm, instruction: str, question: str) -> str:
    index, chunks, model = load_rag_assets()
    context = retrieve(question, index, chunks, model)
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "system",
                "content": f"Retrieved context:\n{context}",
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        max_tokens=512,
        temperature=0.0,
        top_p=0.95,
        repeat_penalty=1.2,
    )
    return trim_response(response["choices"][0]["message"]["content"])


def trim_response(response_text):
    marker = "</think>"
    response_text = response_text
    if marker in response_text:
        return response_text.split(marker, 1)[1].lstrip()
    return response_text


def retrieve(query: str, index, chunks, model, k=TOP_K):
    query_embedding = model.encode(query)
    query_embedding = np.array([query_embedding]).astype("float32")  # shape (1, dim)

    scores, indices = index.search(query_embedding, k)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks = []
    text_len = len(text)
    start = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start += chunk_size - overlap

    return chunks
