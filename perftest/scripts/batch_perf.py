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


SAMPLE_QUESTIONS_PATH = ROOT_DIR / "perftest" / "fixtures" / "sample_questions.csv"
PERF_RESULTS_PATH = ROOT_DIR / "perftest" / "results" / "batch_perf_results.csv"


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


@unittest.skipUnless(
    os.getenv("RUN_PERF_TESTS") == "1",
    "Set RUN_PERF_TESTS=1 to run real RAG performance tests.",
)
class ChatBotPerfTest(unittest.TestCase):
    def test_rag_response_time_sequence(self):
        chatbot = ChatBot()
        questions = _load_sample_questions()
        self.assertGreater(len(questions), 0)

        try:
            for question in questions:
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
