import csv
import os
import sys
import time
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chatbot.chatbot import ChatBot
from rag import llm_client


SAMPLE_QUESTIONS_PATH = ROOT_DIR / "perftest" / "fixtures" / "sample_questions.csv"
PERF_RESULTS_PATH = ROOT_DIR / "perftest" / "results" / "single_question_perf_results.csv"
MODEL_OVERRIDE_ENV_VARS = {
    "model_path": ("MODEL_PATH", str),
    "model_filename": ("MODEL_FILENAME", str),
    "top_p": ("MODEL_TOP_P", float),
    "max_tokens": ("MODEL_MAX_TOKENS", int),
    "temperature": ("MODEL_TEMPERATURE", float),
    "repeat_penalty": ("MODEL_REPEAT_PENALTY", float),
    "n_gpu_layers": ("MODEL_N_GPU_LAYERS", int),
    "n_batches": ("MODEL_N_BATCHES", int),
    "n_ctx": ("MODEL_N_CTX", int),
    "n_threads": ("MODEL_N_THREADS", int),
    "sentence_transformer": ("MODEL_SENTENCE_TRANSFORMER", str),
    "top_k": ("MODEL_TOP_K", int),
    "index_path": ("MODEL_INDEX_PATH", str),
    "chunks_path": ("MODEL_CHUNKS_PATH", str),
}


def _load_sample_questions():
    with open(SAMPLE_QUESTIONS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["question"] for row in reader if row.get("question")]


def _write_perf_result(elapsed_seconds, query):
    file_exists = os.path.exists(PERF_RESULTS_PATH)

    PERF_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PERF_RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "query", "version", "change"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "time": f"{elapsed_seconds:.3f}",
                "query": query,
                "version": os.getenv("VERSION", ""),
                "change": os.getenv("CHANGE", ""),
            }
        )


def _load_model_overrides_from_env():
    overrides = {}
    for config_key, (env_var, caster) in MODEL_OVERRIDE_ENV_VARS.items():
        value = os.getenv(env_var)
        if value not in (None, ""):
            overrides[config_key] = caster(value)
    return overrides


@unittest.skipUnless(
    os.getenv("RUN_PERF_TESTS") == "1",
    "Set RUN_PERF_TESTS=1 to run real RAG performance tests.",
)
class ChatBotPerfTest(unittest.TestCase):
    def test_rag_response_time_sequence(self):
        overrides = _load_model_overrides_from_env()
        if overrides:
            llm_client.override_model_config(overrides)

        chatbot = ChatBot()
        questions = _load_sample_questions()
        self.assertGreater(len(questions), 0)

        try:
            question = "How do I create a S3 bucker in AWS?"
            with self.subTest(question=question):
                start = time.perf_counter()
                response = chatbot.ask_question_using_rag(question)
                elapsed_seconds = time.perf_counter() - start

                print(f"query: {question}")
                print(f"ask_question_using_rag response time: {elapsed_seconds:.3f}s")
                print(f"response length: {len(response)} characters")
                _write_perf_result(elapsed_seconds, question)

                self.assertIsInstance(response, str)
                self.assertGreater(len(response.strip()), 0)
        finally:
            chatbot.close_model()


if __name__ == "__main__":
    unittest.main()
