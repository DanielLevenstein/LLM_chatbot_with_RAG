FROM python:3.11-slim

WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    XDG_CACHE_HOME=/app/.cache

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --no-compile --prefer-binary llama-cpp-python

RUN pip install --no-cache-dir --no-compile --index-url https://download.pytorch.org/whl/cpu torch==2.7.1+cpu

COPY requirements.txt ./
RUN pip3 install --no-cache-dir --no-compile -r requirements.txt

COPY config/ ./config/
COPY docker/warm_hf_cache.py ./docker/warm_hf_cache.py
RUN python docker/warm_hf_cache.py

RUN rm -rf chatbot rag config
COPY index ./index
COPY app.py app.py
COPY chatbot/ ./chatbot/
COPY rag/ ./rag/
COPY config/ ./config/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
