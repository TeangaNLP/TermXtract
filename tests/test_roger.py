import unittest
from teanga import Corpus
from termxtract.supervised.term_extractor import SupervisedTermExtractor


class TestRogerExtraction(unittest.TestCase):
    """Test cases for Roger term extraction method."""

    def test_roger_extraction_teanga(self):
        """Test term extraction from a Teanga corpus using RogerTermExtractor."""
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        doc1 = corpus.add_doc("Genetic algorithms optimize ranking functions.")
        doc1.words = [(0, 7), (8, 18), (19, 29), (30, 38), (39, 50)]

        doc2 = corpus.add_doc("Machine learning and term extraction.")
        doc2.words = [(0, 7), (8, 16), (17, 20), (21, 25), (26, 36)]

        labels = {
            "genetic algorithms": 1,
            "machine learning": 1,
            "term extraction": 1,
            "ranking functions": 0,
        }

        extractor = SupervisedTermExtractor(method="roger", population_size=30, generations=50, threshold=0.1, n=2)
        results = extractor.extract(corpus, labels)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

    def test_roger_extraction_strings(self):
        """Test term extraction from a list of strings using RogerTermExtractor."""
        text_corpus = [
            "Genetic algorithms optimize ranking functions.",
            "Machine learning and term extraction.",
        ]

        labels = {
            "genetic algorithms": 1,
            "machine learning": 1,
            "term extraction": 1,
            "ranking functions": 0,
        }

        extractor = SupervisedTermExtractor(method="roger", population_size=30, generations=50, threshold=0.1, n=2)
        results = extractor.extract(text_corpus, labels)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")
