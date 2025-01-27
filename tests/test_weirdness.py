import unittest
from teanga import Corpus
from termxtract.term_extractor import TermExtractor


class TestWeirdnessExtraction(unittest.TestCase):
    """Test cases for Weirdness term extraction method."""

    def setUp(self):
        """Prepare test data."""
        self.target_corpus_strings = [
            "Weirdness focuses on domain-specific terminology.",
            "Specific terms are more common in the target corpus.",
        ]
        self.reference_corpus_strings = [
            "General reference corpus contains common phrases and terms.",
            "Reference texts are not domain-specific.",
        ]

        self.target_corpus_teanga = Corpus()
        self.target_corpus_teanga.add_layer_meta("text")
        self.target_corpus_teanga.add_layer_meta("words", layer_type="span", base="text")
        doc1 = self.target_corpus_teanga.add_doc("Weirdness focuses on domain-specific terminology.")
        doc1.words = [(0, 9), (10, 17), (18, 20), (21, 37), (38, 50)]
        doc2 = self.target_corpus_teanga.add_doc("Specific terms are more common in the target corpus.")
        doc2.words = [(0, 8), (9, 14), (15, 18), (19, 23), (24, 30), (31, 33), (34, 40), (41, 47)]

        self.reference_corpus_teanga = Corpus()
        self.reference_corpus_teanga.add_layer_meta("text")
        self.reference_corpus_teanga.add_layer_meta("words", layer_type="span", base="text")
        ref_doc1 = self.reference_corpus_teanga.add_doc("General reference corpus contains common phrases and terms.")
        ref_doc1.words = [(0, 8), (9, 18), (19, 25), (26, 34), (35, 41), (42, 49), (50, 53)]
        ref_doc2 = self.reference_corpus_teanga.add_doc("Reference texts are not domain-specific.")
        ref_doc2.words = [(0, 9), (10, 15), (16, 19), (20, 23), (24, 37)]

    def test_weirdness_extraction_strings(self):
        """Test Weirdness extraction with a list of strings."""
        extractor = TermExtractor(
            method="weirdness",
            reference_corpus=self.reference_corpus_strings,
            threshold=0.1,
            n=2,
        )
        results = extractor.extract(self.target_corpus_strings)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        for term_entry in results.terms:
            self.assertIn("doc_id", term_entry, "Each result should have 'doc_id'.")
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

    def test_weirdness_extraction_teanga(self):
        """Test Weirdness extraction with a Teanga corpus."""
        extractor = TermExtractor(
            method="weirdness",
            reference_corpus=self.reference_corpus_teanga,
            threshold=0.1,
            n=2,
        )
        results = extractor.extract(self.target_corpus_teanga)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        for term_entry in results.terms:
            self.assertIn("doc_id", term_entry, "Each result should have 'doc_id'.")
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

