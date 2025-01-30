import unittest
from teanga import Corpus
from termxtract.supervised.term_extractor import SupervisedTermExtractor


class TestAdaBoostExtraction(unittest.TestCase):
    """Test cases for AdaBoost term extraction method."""

    def setUp(self):
        """Set up a small test corpus and labels."""
        self.corpus = Corpus()
        self.corpus.add_layer_meta("text")
        self.corpus.add_layer_meta("words", layer_type="span", base="text")

        doc1 = self.corpus.add_doc("Machine learning is widely used in AI research.")
        doc1.words = [(0, 7), (8, 16), (17, 19), (20, 25), (26, 30), (31, 33), (34, 42)]

        self.text_corpus = ["Machine learning is widely used in AI research."]

        # Labels (Supervised Learning)
        self.labels = {
            "machine learning": 1,
            "AI research": 1,
            "widely used": 0
        }

    def test_adaboost_extraction_teanga(self):
        """Test AdaBoost term extraction on a Teanga corpus."""
        extractor = SupervisedTermExtractor(method="adaboost", threshold=0.1, n=2)
        results = extractor.extract(self.corpus, self.labels)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

    def test_adaboost_extraction_strings(self):
        """Test AdaBoost term extraction on a list of strings."""
        extractor = SupervisedTermExtractor(method="adaboost", threshold=0.1, n=2)
        results = extractor.extract(self.text_corpus, self.labels)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")


if __name__ == "__main__":
    unittest.main()

