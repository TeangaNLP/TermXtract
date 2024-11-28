import unittest
from teanga import Corpus
from termxtract.term_extractor import TermExtractor
from termxtract.utils import ATEResults


class TestTFIDFExample(unittest.TestCase):

    def test_tfidf_extraction_teanga(self):
        # Step 1: Create a Teanga corpus and add layers
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        # Step 2: Add documents with word offsets
        doc1 = corpus.add_doc("This is an example document.")
        doc1.words = [(0, 4), (5, 7), (8, 10), (11, 18), (19, 28)]

        doc2 = corpus.add_doc("TFIDF extraction with unigrams, bigrams, and trigrams.")
        doc2.words = [(0, 5), (6, 16), (17, 21), (22, 30), (31, 39), (40, 44), (45, 53)]

        # Step 3: Initialize TermExtractor with TF-IDF, n=3 for 1-gram, 2-gram, and 3-gram extraction
        extractor = TermExtractor(method="tfidf", threshold=0.1, n=3)

        # Step 4: Perform term extraction on the corpus
        results = extractor.extract(corpus)

        # Step 5: Assertions for ATEResults structure
        self.assertIsInstance(results, ATEResults, "Expected results to be an instance of ATEResults.")
        self.assertEqual(results.corpus, corpus, "Expected the corpus in ATEResults to match the input corpus.")
        
        # Validate terms field
        self.assertIsInstance(results.terms, list, "Expected 'terms' to be a list.")
        for item in results.terms:
            self.assertIn("doc_id", item, "Each term entry should have a 'doc_id'.")
            self.assertIn("terms", item, "Each term entry should have a 'terms' field.")
            self.assertIsInstance(item["terms"], list, "The 'terms' field should be a list.")

        # Validate offsets field
        self.assertIsInstance(results.offsets, list, "Expected 'offsets' to be a list.")
        for item in results.offsets:
            self.assertIn("doc_id", item, "Each offset entry should have a 'doc_id'.")
            self.assertIn("offsets", item, "Each offset entry should have an 'offsets' field.")
            self.assertIsInstance(item["offsets"], list, "The 'offsets' field should be a list.")

    def test_tfidf_extraction_list_of_strings(self):
        # Input: List of strings (non-Teanga corpus)
        text_corpus = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?"
        ]

        # Step 1: Initialize TermExtractor
        extractor = TermExtractor(method="tfidf", threshold=0.1, n=2)

        # Step 2: Perform term extraction on the list of strings
        results = extractor.extract(text_corpus)

        # Step 3: Assertions for ATEResults structure
        self.assertIsInstance(results, ATEResults, "Expected results to be an instance of ATEResults.")
        self.assertIsInstance(results.corpus, list, "Expected the corpus to be a list for non-Teanga input.")
        for doc in results.corpus:
            self.assertIn("doc_id", doc, "Each document in the corpus should have a 'doc_id'.")
            self.assertIn("text", doc, "Each document in the corpus should have a 'text' field.")
            self.assertIsInstance(doc["text"], str, "The 'text' field should be a string.")

        # Validate terms field
        self.assertIsInstance(results.terms, list, "Expected 'terms' to be a list.")
        for item in results.terms:
            self.assertIn("doc_id", item, "Each term entry should have a 'doc_id'.")
            self.assertIn("terms", item, "Each term entry should have a 'terms' field.")
            self.assertIsInstance(item["terms"], list, "The 'terms' field should be a list.")

        # Validate offsets field (should be None for list of strings)
        self.assertIsNone(results.offsets, "Expected 'offsets' to be None for non-Teanga input.")


if __name__ == '__main__':
    unittest.main()
