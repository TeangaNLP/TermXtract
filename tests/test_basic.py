import unittest
from teanga import Corpus
from termxtract.term_extractor import TermExtractor


class TestBasicExtraction(unittest.TestCase):
    """Test cases for Basic term extraction method."""

    def test_basic_extraction_teanga(self):
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        doc1 = corpus.add_doc("Basic method promotes nested terms.")
        doc1.words = [(0, 5), (6, 12), (13, 21), (22, 29), (30, 36)]

        extractor = TermExtractor(method="basic", alpha=0.5, threshold=0.1, n=2)
        results = extractor.extract(corpus)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

    def test_basic_extraction_strings(self):
        text_corpus = ["Basic method promotes nested terms."]
        extractor = TermExtractor(method="basic", alpha=0.5, threshold=0.1, n=2)
        results = extractor.extract(text_corpus)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")
