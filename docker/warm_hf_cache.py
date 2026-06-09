import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer


CONFIG_PATH = Path("config/model_default.json")


def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    hf_hub_download(
        repo_id=config["model_path"],
        filename=config["model_filename"],
    )
    SentenceTransformer(config["sentence_transformer"])


if __name__ == "__main__":
    main()
