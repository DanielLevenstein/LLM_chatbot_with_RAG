import unittest

from ingest import context_aware_chunk_text


class IngestTest(unittest.TestCase):
    def test_context_aware_chunk_text_preserves_small_sections(self):
        chunks = context_aware_chunk_text("first paragraph\n\nsecond paragraph", max_chunk_size=50)

        self.assertEqual(["first paragraph", "second paragraph"], chunks)

    def test_context_aware_chunk_text_splits_oversized_single_newline_section(self):
        raw_text = "\n".join(["a" * 100 for _ in range(20)])

        chunks = context_aware_chunk_text(raw_text, max_chunk_size=250, overlap=25)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 250 for chunk in chunks))

    def test_context_aware_chunk_text_applies_overlap_to_oversized_sections(self):
        raw_text = "a" * 501

        chunks = context_aware_chunk_text(raw_text, max_chunk_size=500, overlap=50)

        self.assertEqual(2, len(chunks))
        self.assertEqual(chunks[0][-50:], chunks[1][:50])


if __name__ == "__main__":
    unittest.main()
