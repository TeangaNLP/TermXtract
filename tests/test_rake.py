import unittest
from teanga import Corpus
from termxtract.term_extractor import TermExtractor


class TestRAKEExtraction(unittest.TestCase):
    """Test cases for RAKE term extraction method."""

    def test_rake_extraction_teanga(self):
        """Test RAKE extraction on a Teanga corpus."""
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        doc1 = corpus.add_doc("RAKE method identifies key phrases effectively.")
        doc1.words = [(0, 4), (5, 11), (12, 23), (24, 27), (28, 35), (36, 46)]

        stoplist = ["method", "identifies"]
        extractor = TermExtractor(method="rake", stoplist=stoplist, threshold=0.1, n=3)
        results = extractor.extract(corpus)

        # Validate that results are returned correctly
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

    def test_rake_extraction_strings(self):
        """Test RAKE extraction on a list of strings."""
        text_corpus = ["RAKE method identifies key phrases effectively."]
        stoplist = ["method", "identifies"]

        extractor = TermExtractor(method="rake", stoplist=stoplist, threshold=0.1, n=3)
        results = extractor.extract(text_corpus)

        # Validate that results are returned correctly
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

    def test_rake_extraction_empty_string(self):
        """Test RAKE extraction with an empty string."""
        text_corpus = [""]
        stoplist = ["method", "identifies"]

        extractor = TermExtractor(method="rake", stoplist=stoplist, threshold=0.1, n=3)
        results = extractor.extract(text_corpus)

        # Validate that no terms are extracted
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        for term_entry in results.terms:
            self.assertEqual(len(term_entry["terms"]), 0, "No terms should be extracted from an empty string.")

    def test_rake_extraction_no_stoplist(self):
        """Test RAKE extraction with no stoplist provided."""
        text_corpus = ["RAKE method identifies key phrases effectively."]
        stoplist = ["method", "identifies"]

        extractor = TermExtractor(method="rake", stoplist=stoplist, threshold=0.1, n=3)
        results = extractor.extract(text_corpus)

        # Validate that results are returned correctly
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

    def test_rake_extraction_with_threshold(self):
        """Test RAKE extraction with a high threshold, ensuring filtering works."""
        text_corpus = ["RAKE method identifies key phrases effectively."]
        stoplist = ["method", "identifies"]

        extractor = TermExtractor(method="rake", stoplist=stoplist, threshold=10, n=3)  # Set a high threshold
        results = extractor.extract(text_corpus)

        # Validate that no terms are extracted due to the high threshold
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        for term_entry in results.terms:
            self.assertEqual(len(term_entry["terms"]), 0, "No terms should be extracted with a high threshold.")


if __name__ == "__main__":
    unittest.main()

