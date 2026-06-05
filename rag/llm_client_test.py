import unittest

from rag.llm_client import (
    trim_response,
    retrieve,
    chunk_text,
    INDEX_PATH,
    SENTENCE_TRANSFORMER,
    CHUNKS_PATH,
)

from sentence_transformers import SentenceTransformer

import pickle
import faiss


class LlmClientTest(unittest.TestCase):
    def test_trim_response(self):
        self.assertEqual("value", trim_response("<think></think>value"))

    def test_retrieve_chunks(self):
        index = faiss.read_index(INDEX_PATH)
        model = SentenceTransformer(SENTENCE_TRANSFORMER)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        context = retrieve("Blood Pressure", index, chunks, model)
        self.assertNotEqual(len(context[0]), 1, "Chunks should be more than 1 character")

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
