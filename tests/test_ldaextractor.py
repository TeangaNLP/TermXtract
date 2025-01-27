import unittest
from teanga import Corpus
from termxtract.term_extractor import TermExtractor

class TestTopicModelingTermExtractor(unittest.TestCase):
    """Test cases for TopicModeling-based term extraction."""

    def test_topic_modeling_extraction_teanga(self):
        """Test TopicModeling term extraction on a Teanga Corpus."""
        # Create a Teanga Corpus
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        # Add documents to the corpus
        doc1 = corpus.add_doc("Machine learning models are very powerful.")
        doc1.words = [(0, 7), (8, 16), (17, 23), (24, 27), (28, 37)]

        doc2 = corpus.add_doc("Deep learning and neural networks are revolutionary.")
        doc2.words = [(0, 4), (5, 13), (14, 17), (18, 24), (25, 34), (35, 38), (39, 53)]

        # Initialize the TopicModelingTermExtractor through TermExtractor
        extractor = TermExtractor(method="topicmodeling", num_topics=10, threshold=0.01, n=2)

        # Extract terms
        results = extractor.extract(corpus)

        # Assertions
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

    def test_topic_modeling_extraction_strings(self):
        """Test TopicModeling term extraction on a list of strings."""
        # Create a string-based corpus
        text_corpus = [
            "Machine learning models are very powerful.",
            "Deep learning and neural networks are revolutionary."
        ]

        # Initialize the TopicModelingTermExtractor through TermExtractor
        extractor = TermExtractor(method="topicmodeling", num_topics=10, threshold=0.01, n=2)

        # Extract terms
        results = extractor.extract(text_corpus)

        # Assertions
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

if __name__ == "__main__":
    unittest.main()

