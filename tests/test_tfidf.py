import unittest
from termxtract.term_extractor import TermExtractor

class TestTFIDFTermExtractor(unittest.TestCase):
    def setUp(self):
        """Set up a sample corpus for testing."""
        self.corpus = [
            "This is a sample document.",
            "This document is another example sample."
        ]

    def test_extract_terms_without_threshold(self):
        """Test TF-IDF extraction without a threshold."""
        extractor = TermExtractor(method="tfidf")
        results = extractor.extract(self.corpus)

        # Verify that results are returned for each document
        self.assertEqual(len(results), len(self.corpus))
        
        # Check that the results contain expected words
        for doc_tfidf in results:
            self.assertIn("sample", doc_tfidf)
            self.assertIn("document", doc_tfidf)
        
        # Confirm that no threshold filtering occurred
        self.assertGreater(len(results[0]), 0)  # Ensure terms were extracted

    def test_extract_terms_with_threshold(self):
        """Test TF-IDF extraction with a threshold of 0.1."""
        threshold = 0.1
        extractor = TermExtractor(method="tfidf", threshold=threshold)
        results = extractor.extract(self.corpus)

        # Verify that results are returned for each document
        self.assertEqual(len(results), len(self.corpus))

        # Check that terms with low scores are filtered out
        for doc_tfidf in results:
            # Ensure only terms with TF-IDF >= threshold are present
            for term, score in doc_tfidf.items():
                self.assertGreaterEqual(score, threshold)

    def test_empty_corpus(self):
        """Test TF-IDF extraction on an empty corpus."""
        extractor = TermExtractor(method="tfidf")
        results = extractor.extract([])

        # Expecting an empty list of results
        self.assertEqual(results, [])

    def test_threshold_exceeds_all_terms(self):
        """Test TF-IDF extraction with a high threshold that filters all terms."""
        high_threshold = 1.0  # Set a high threshold to filter all terms
        extractor = TermExtractor(method="tfidf", threshold=high_threshold)
        results = extractor.extract(self.corpus)

        # Expect all dictionaries to be empty since threshold filters all terms out
        for doc_tfidf in results:
            self.assertEqual(doc_tfidf, {})

if __name__ == "__main__":
    unittest.main()
