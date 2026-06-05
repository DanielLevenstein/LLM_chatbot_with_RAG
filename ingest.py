import os
import pickle
import faiss

from rag.llm_client import chunk_text, SENTENCE_TRANSFORMER

DATA_DIR = "data"
OUTPUT_INDEX_PATH = "index/index.faiss"
OUTPUT_CHUNKS_PATH = "index/chunks.pkl"


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def ingest_folder(data_dir: str):
    all_chunks: list[str] = []

    for filename in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, filename)

        if not os.path.isfile(path):
            continue

        print(f"Processing: {filename}")

        try:
            if filename.lower().endswith(".txt") or filename.lower().endswith(".md"):
                raw_text = read_txt(path)

            else:
                print(f"Skipping unsupported file: {filename}")
                continue

            chunks = chunk_text(raw_text)
            all_chunks.extend(chunks)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    return all_chunks

def embed_chunks(chunks):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(SENTENCE_TRANSFORMER)
    embeddings = model.encode(chunks, normalize_embeddings=True)
    print("Embeddings Chunks Please be patient.")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, OUTPUT_INDEX_PATH)

    with open(OUTPUT_CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Ingested {len(chunks)} chunks into FAISS index")


if __name__ == "__main__":
    os.makedirs("index", exist_ok=True)

    # 1 Load and chunk all files
    chunks = ingest_folder(DATA_DIR)

    if not chunks:
        raise RuntimeError("No chunks were generated. Check your data folder.")

    print(f"Total chunks created: {len(chunks)}")
    embed_chunks(chunks)
