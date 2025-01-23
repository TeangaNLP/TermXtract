import unittest
from teanga import Corpus
from termxtract.domaincoherence import DomainCoherenceTermExtractor
from termxtract.utils import ATEResults


class TestDomainCoherenceTermExtractor(unittest.TestCase):
    """Test cases for DomainCoherenceTermExtractor."""

    def setUp(self):
        """Set up test data for both Teanga and string corpora."""
        # Teanga corpus setup
        self.teanga_corpus = Corpus()
        self.teanga_corpus.add_layer_meta("text")
        self.teanga_corpus.add_layer_meta("words", layer_type="span", base="text")

        doc1 = self.teanga_corpus.add_doc("The term extraction algorithm uses domain coherence.")
        doc1.words = [(0, 3), (4, 8), (9, 19), (20, 28), (29, 33), (34, 40), (41, 50)]

        doc2 = self.teanga_corpus.add_doc("Domain coherence identifies important terms in documents.")
        doc2.words = [(0, 6), (7, 15), (16, 27), (28, 37), (38, 43), (44, 54)]

        # String corpus setup
        self.string_corpus = [
            "The term extraction algorithm uses domain coherence.",
            "Domain coherence identifies important terms in documents."
        ]

        # Extractor instance
        self.extractor = DomainCoherenceTermExtractor(threshold=0.1, n=2, window_size=5)

    def test_domaincoherence_teanga(self):
        """Test DomainCoherence term extraction on a Teanga corpus."""
        results = self.extractor.extract_terms_teanga(self.teanga_corpus)

        # Validate results structure
        self.assertIsInstance(results, ATEResults, "Results should be an instance of ATEResults.")
        self.assertTrue(results.terms, "Results should contain terms.")
        self.assertTrue(results.corpus, "Results should contain the original corpus.")

        # Validate each document's terms
        for doc in results.terms:
            self.assertIn("doc_id", doc, "Each term entry should contain 'doc_id'.")
            self.assertIn("terms", doc, "Each term entry should contain 'terms'.")
            for term in doc["terms"]:
                self.assertIn("term", term, "Each term should have a 'term' key.")
                self.assertIn("score", term, "Each term should have a 'score' key.")
                self.assertGreaterEqual(term["score"], 0, "Scores should be non-negative.")

    def test_domaincoherence_strings(self):
        """Test DomainCoherence term extraction on a plain list of strings."""
        results = self.extractor.extract_terms_strings(self.string_corpus)

        # Validate results structure
        self.assertIsInstance(results, ATEResults, "Results should be an instance of ATEResults.")
        self.assertTrue(results.terms, "Results should contain terms.")
        self.assertTrue(results.corpus, "Results should contain the original corpus.")

        # Validate each document's terms
        for doc in results.terms:
            self.assertIn("doc_id", doc, "Each term entry should contain 'doc_id'.")
            self.assertIn("terms", doc, "Each term entry should contain 'terms'.")
            for term in doc["terms"]:
                self.assertIn("term", term, "Each term should have a 'term' key.")
                self.assertIn("score", term, "Each term should have a 'score' key.")
                self.assertGreaterEqual(term["score"], 0, "Scores should be non-negative.")

    def test_results_consistency(self):
        """Ensure consistent results between Teanga and string corpora."""
        teanga_results = self.extractor.extract_terms_teanga(self.teanga_corpus)
        string_results = self.extractor.extract_terms_strings(self.string_corpus)

        # Compare the number of terms extracted
        teanga_term_count = sum(len(doc["terms"]) for doc in teanga_results.terms)
        string_term_count = sum(len(doc["terms"]) for doc in string_results.terms)
        self.assertEqual(
            teanga_term_count, string_term_count,
            "The number of terms extracted should be consistent between Teanga and string corpora."
        )

    def test_no_context_terms(self):
        """Test behavior when there are no valid context terms."""
        empty_corpus = [
            "This is a document without meaningful terms.",
            "Nothing important here."
        ]
        extractor = DomainCoherenceTermExtractor(threshold=0.1, n=2, window_size=5)
        results = extractor.extract_terms_strings(empty_corpus)

        # Validate empty or low-scoring terms
        for doc in results.terms:
            for term in doc["terms"]:
                self.assertLess(term["score"], 0.1, "Terms should not pass the threshold.")

    def test_single_document_corpus(self):
        """Test behavior with a single-document corpus."""
        single_doc_corpus = ["Domain coherence works with a single document."]
        results = self.extractor.extract_terms_strings(single_doc_corpus)

        # Validate structure
        self.assertEqual(len(results.terms), 1, "Results should contain one entry for a single document.")
        self.assertIn("doc_0", [doc["doc_id"] for doc in results.terms], "The single document should have an ID 'doc_0'.")

    def test_small_corpus(self):
        """Test behavior with a very small corpus."""
        small_corpus = [
            "Short text.",
            "Another small document."
        ]
        results = self.extractor.extract_terms_strings(small_corpus)

        # Validate output structure
        self.assertEqual(len(results.terms), 2, "Each document in the small corpus should have results.")
        for doc in results.terms:
            self.assertIn("doc_id", doc, "Each result should contain 'doc_id'.")
            self.assertIn("terms", doc, "Each result should contain terms.")

