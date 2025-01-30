import unittest
from teanga import Corpus
from termxtract.unsupervised.term_extractor import TermExtractor


class TestRIDFExtraction(unittest.TestCase):

    def test_ridf_extraction_teanga(self):
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        doc1 = corpus.add_doc("This is an example document.")
        doc1.words = [(0, 4), (5, 7), (8, 10), (11, 18), (19, 28)]

        doc2 = corpus.add_doc("RIDF extraction with unigrams, bigrams, and trigrams.")
        doc2.words = [(0, 4), (5, 15), (16, 19), (20, 28), (29, 37), (38, 41), (42, 50)]

        extractor = TermExtractor(method="ridf", threshold=0.1, n=3)
        results = extractor.extract(corpus)

        self.assertIsNotNone(results.terms, "RIDF terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")

    def test_ridf_extraction_strings(self):
        text_corpus = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?"
        ]

        extractor = TermExtractor(method="ridf", threshold=0.1, n=2)
        results = extractor.extract(text_corpus)

        self.assertIsNotNone(results.terms, "RIDF terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")


if __name__ == "__main__":
    unittest.main()
