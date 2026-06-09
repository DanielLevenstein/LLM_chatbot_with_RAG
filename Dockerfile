FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --prefer-binary llama-cpp-python

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

RUN rm -rf chatbot rag config
COPY index ./index
COPY app.py app.py
COPY chatbot/ ./chatbot/
COPY rag/ ./rag/
COPY config/ ./config/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
