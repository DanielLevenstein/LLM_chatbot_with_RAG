# AWS Documentation RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) chatbot built for answering technical AWS questions using 
live documentation embeddings and semantic search.

The system crawls selected AWS documentation pages, processes and chunks the content, generates vector embeddings, 
and stores them in a FAISS index for fast retrieval. When a user submits a question, the chatbot retrieves the 
most relevant documentation fragments and uses them as grounded context for response generation.

## Current AWS Coverage

```
["cli", "cloudformation", "cloudwatch", "dynamodb", "elasticloadbalancing",
            "ec2", "ecs", "eks", "iam", "lambda", "rds", "s3",
            "sagemaker", "vpc", "xray" ]

```


## 1. Install Dependencies

Make sure your virtual environment is active and install required packages:

```bash
pip install -r requirements.txt
```

## Running Locally
My AWS rag includes a local streamlit app for testing. For ease of testing, the `extract.py` and `ingest.py` scripts 
have been run and the chunks and indexes created have been committed to source control.  

To run the streamlit app run, install the dependencies and run the following command locally. 

```bash
pip install -r requirements.txt
streamlit run app.py
```
Or run through docker.
```bash
docker build --tag aws-rag .
docker run -d -p 8501:8501 aws-rag
```

Then open `http://localhost:8501` on the host machine.

Note: inside a Docker container, `localhost` means that same container. If another container needs to reach this app,
put both containers on the same Docker network and use the app container name, or use `host.docker.internal:8501` to
reach a service running on the host.

## Example Queries

- "How do I configure an Application Load Balancer for ECS?"
- "What permissions are required for Lambda to access S3?"
- "How do I troubleshoot DynamoDB throttling?"
- "How do I deploy a SageMaker endpoint?"

# Architecture Overview

1. Documentation Crawling

   - Scrapes AWS documentation pages
   - Limits recursive traversal depth to control corpus size
   - Filters and normalizes extracted content
2. Document Processing

   - Cleans HTML and removes navigation noise
   - Splits documentation into searchable chunks
   - Preserves contextual structure where possible
3. Embedding & Indexing

   - Generates vector embeddings for document chunks
   - Stores embeddings in a FAISS vector index
   - Enables low-latency semantic similarity search
4. Retrieval-Augmented Generation

   - Retrieves relevant documentation chunks for each query
   - Injects retrieved context into the LLM prompt
   - Produces grounded technical responses based on AWS documentation

## Goals

- Reduce hallucinations in AWS-related answers
- Provide context-aware technical support
- Enable fast semantic search across AWS services
- Serve as a lightweight proof-of-concept RAG architecture

## Tech Stack

- Python
- BeautifulSoup
- Requests
- FAISS
- Embedding Models
- Large Language Models (LLMs)


## Status

This project is currently a proof of concept focused on validating:

- constrained documentation crawling
- semantic retrieval quality
- FAISS-based vector search
- AWS-focused RAG workflows
### Notes

- Make sure the embedding model used during ingestion matches the model used for querying (default: `all-MiniLM-L6-v2`).
- FAISS indexes and chunks must remain aligned — do not modify the pickled chunks file manually.
- All scripts have to be run from the root directory of the project.

# Creating New Indexes

## Extract Documentation from AWS

- Copy `config/features_default.json` to `config/features_current.json`
- Run `extract.py` to extract documentation
- Run `ingest.py` to create new indexes

This process will:

1. Scrapes AWS documentation for features listed in features_current.json
2. Save raw file content to data directory. 
3. Build a FAISS index from download data. 
4. Save the chunks and index in the `index/` folder

### Output

- `index/index.faiss` – FAISS vector index
- `index/chunks.pkl` – Pickled text chunks
