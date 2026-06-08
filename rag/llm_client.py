import atexit
import gc
import json
from datetime import datetime
import pickle
import os
import time
from threading import RLock


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MODEL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "model_default.json")


DEFAULT_MODEL_CONFIG = {
    "model_path": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    "model_filename": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
    "top_p": 0.95,
    "max_tokens": 500,
    "temperature": 0.0,
    "repeat_penalty": 1.2,
    "n_gpu_layers": 50,
    "n_batches": 256,
    "n_ctx": 2048,
    "n_threads": None,
    "sentence_transformer": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k": 4,
    "index_path": "index/index.faiss",
    "chunks_path": "index/chunks.pkl",
}

def load_model_config(overrides=None):
    config = DEFAULT_MODEL_CONFIG.copy()
    try:
        with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
            config.update(json.load(f))
    except FileNotFoundError:
        print(f"Model config not found at {MODEL_CONFIG_PATH}, using defaults.")
    except json.JSONDecodeError as e:
        print(f"Failed to parse model config: {e}. Using defaults.")

    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def apply_model_config(config):
    global MODEL_SETTINGS, MODEL_PATH, MODEL_FILENAME, TOP_P, MAX_TOKENS, TEMPERATURE
    global REPEAT_PENALTY, N_GPU_LAYERS, N_BATCHES, N_CTX, N_THREADS
    global SENTENCE_TRANSFORMER, INDEX_PATH, CHUNKS_PATH, TOP_K

    MODEL_SETTINGS = config
    MODEL_PATH = config["model_path"]
    MODEL_FILENAME = config["model_filename"]
    TOP_P = config["top_p"]
    MAX_TOKENS = config["max_tokens"]
    TEMPERATURE = config["temperature"]
    REPEAT_PENALTY = config["repeat_penalty"]
    N_GPU_LAYERS = config["n_gpu_layers"]
    N_BATCHES = config["n_batches"]
    N_CTX = config["n_ctx"]
    N_THREADS = config["n_threads"]
    SENTENCE_TRANSFORMER = config["sentence_transformer"]
    TOP_K = config["top_k"]
    INDEX_PATH = config["index_path"]
    CHUNKS_PATH = config["chunks_path"]


def override_model_config(overrides):
    close_all_resources()
    apply_model_config(load_model_config(overrides))


MODEL_SETTINGS = load_model_config()
MODEL_PATH = None
MODEL_FILENAME = None
TOP_P = None
MAX_TOKENS = None
TEMPERATURE = None
REPEAT_PENALTY = None
N_GPU_LAYERS = None
N_BATCHES = None
N_CTX = None
N_THREADS = None
SENTENCE_TRANSFORMER = None
TOP_K = None
INDEX_PATH = None
CHUNKS_PATH = None
_client = None
_index = None
_chunks = None
_embed_model = None
_client_lock = RLock()
_assets_lock = RLock()
apply_model_config(MODEL_SETTINGS)


def get_llm_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = create_llm(MODEL_PATH, MODEL_FILENAME)
    return _client


def _close_resource(resource):
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def close_llm_client():
    global _client
    with _client_lock:
        client = _client
        _client = None

    if client is not None:
        _close_resource(client)
        gc.collect()


def close_rag_assets():
    global _index, _chunks, _embed_model
    with _assets_lock:
        _index = None
        _chunks = None
        _embed_model = None

    gc.collect()


def close_all_resources():
    close_llm_client()
    close_rag_assets()


atexit.register(close_all_resources)


def download_model(model_name_or_path, model_basename, max_retries=3, initial_backoff=5):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

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
        with _assets_lock:
            if _index is None or _chunks is None or _embed_model is None:
                import faiss

                print("CWD:", os.getcwd())
                index = faiss.read_index(INDEX_PATH)

                with open(CHUNKS_PATH, "rb") as f:
                    chunks = pickle.load(f)

                from sentence_transformers import SentenceTransformer

                embed_model = SentenceTransformer(SENTENCE_TRANSFORMER)
                _index = index
                _chunks = chunks
                _embed_model = embed_model

    return _index, _chunks, _embed_model


def create_llm(model_name_or_path, model_basename):
    from llama_cpp import Llama

    print(f"Creating model: {model_basename}")
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

    # Detect available CPU cores
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    n_threads = N_THREADS or max(4, cpu_count - 1)  # Use most cores, at least 4

    llm = Llama(
        model_path=model_path,
        n_threads=n_threads,  # Use detected CPU cores for better performance
        n_batch=N_BATCHES,  # Increase batch size for faster inference
        n_gpu_layers=N_GPU_LAYERS,  # Use CPU-only model execution if GPU is not required
        n_ctx=N_CTX,  # Reduced context window for faster processing
        verbose=False,  # Reduce console output
    )
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Model download to {model_path}\n Download time: {elapsed_time}")
    print(f"LLM initialized with {n_threads} threads")
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
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repeat_penalty=REPEAT_PENALTY,
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
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repeat_penalty=REPEAT_PENALTY,
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
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repeat_penalty=REPEAT_PENALTY,
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
    try:
        import numpy as np

        query_embedding = np.array([query_embedding]).astype("float32")  # shape (1, dim)
    except ImportError:
        query_embedding = [query_embedding]

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
