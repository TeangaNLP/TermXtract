import unittest
from teanga import Corpus
from termxtract.unsupervised.term_extractor import TermExtractor


class TestNMFExtraction(unittest.TestCase):
    """Test cases for NMF-based term extraction method."""

    def test_nmf_extraction_teanga(self):
        """Test NMF extraction using a Teanga corpus."""
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        doc1 = corpus.add_doc("Natural language processing is an important field of AI.")
        doc2 = corpus.add_doc("Machine learning techniques are applied to NLP tasks.")
        doc1.words = [(0, 7), (8, 16), (17, 27), (28, 35), (36, 38), (39, 48)]
        doc2.words = [(0, 7), (8, 16), (17, 27), (28, 35), (36, 38), (39, 48)]

        extractor = TermExtractor(method="nmf", threshold=0.1, n_topics=2, max_features=500, n=2)
        results = extractor.extract(corpus)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

    def test_nmf_extraction_strings(self):
        """Test NMF extraction using a list of strings."""
        text_corpus = [
            "Natural language processing is an important field of AI.",
            "Machine learning techniques are applied to NLP tasks.",
        ]
        extractor = TermExtractor(method="nmf", threshold=0.1, n_topics=2, max_features=500, n=2)
        results = extractor.extract(text_corpus)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

if __name__ == "__main__":
    unittest.main()
