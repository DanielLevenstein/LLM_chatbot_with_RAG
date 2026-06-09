import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from rag.llm_client import (
    close_llm_client,
    trim_response,
    retrieve,
    chunk_text,
    format_retrieved_context,
)
from rag import llm_client


class LlmClientTest(unittest.TestCase):
    def tearDown(self):
        llm_client._client = None
        llm_client.apply_model_config(llm_client.load_model_config())

    def test_trim_response(self):
        self.assertEqual("value", trim_response("<think></think>value"))

    def test_close_llm_client_releases_singleton(self):
        mock_llm = Mock()
        llm_client._client = mock_llm

        close_llm_client()

        mock_llm.close.assert_called_once()
        self.assertIsNone(llm_client._client)

    def test_get_llm_client_only_creates_one_client_for_concurrent_callers(self):
        mock_llm = Mock()

        with patch("rag.llm_client.create_llm", return_value=mock_llm) as mock_create:
            with ThreadPoolExecutor(max_workers=8) as executor:
                clients = list(executor.map(lambda _: llm_client.get_llm_client(), range(8)))

        self.assertEqual([mock_llm] * 8, clients)
        mock_create.assert_called_once_with(llm_client.MODEL_PATH, llm_client.MODEL_FILENAME)

    def test_load_model_config_applies_overrides(self):
        config = llm_client.load_model_config({"max_tokens": 123, "n_ctx": 456})

        self.assertEqual(123, config["max_tokens"])
        self.assertEqual(456, config["n_ctx"])

    def test_format_retrieved_context_trims_to_configured_limit(self):
        llm_client.apply_model_config(llm_client.load_model_config({"max_context_chars": 20}))

        context = format_retrieved_context(["a" * 15, "b" * 20])

        self.assertLessEqual(len(context), 20)

    def test_import_does_not_eagerly_load_torch(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import rag.llm_client; print('torch' in sys.modules)",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual("False", result.stdout.strip())

    def test_retrieve_chunks(self):
        model = Mock()
        model.encode.return_value = [0.1, 0.2, 0.3]
        index = Mock()
        index.search.return_value = ([0.9, 0.8], [[2, 0]])
        chunks = ["first chunk", "second chunk", "blood pressure chunk"]

        context = retrieve("Blood Pressure", index, chunks, model, k=2)

        self.assertEqual(["blood pressure chunk", "first chunk"], context)
        model.encode.assert_called_once_with("Blood Pressure")
        index.search.assert_called_once()

    def test_chunk_text_exact_boundary(self):
        text = "a" * 500
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 500)

    def test_chunk_text_just_over_boundary(self):
        text = "a" * 501
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 500)
        self.assertEqual(len(chunks[1]), 51)
        self.assertEqual(chunks[0][-50:], chunks[1][:50])

    def test_chunk_text_multiple_boundaries(self):
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertEqual(len(chunks), 3)
        self.assertEqual([len(chunk) for chunk in chunks], [500, 500, 100])
        self.assertEqual(chunks[0][-50:], chunks[1][:50])
        self.assertEqual(chunks[1][-50:], chunks[2][:50])

    def test_chunk_text_very_long_text(self):
        text = "a" * 10000
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))
        self.assertEqual(chunks[0][-50:], chunks[1][:50])

    def test_chunk_text_varying_chunk_size(self):
        text = "a" * 1200
        chunks = chunk_text(text, chunk_size=1024, overlap=128)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 1024)
        self.assertEqual(len(chunks[1]), 304)
        self.assertEqual(chunks[0][-128:], chunks[1][:128])


if __name__ == '__main__':
    unittest.main()
