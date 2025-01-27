import unittest
from teanga import Corpus
from termxtract.term_extractor import TermExtractor


class TestDomainPertinenceExtraction(unittest.TestCase):
    """Test cases for DomainPertinence term extraction method."""

    def test_domain_pertinence_extraction_teanga(self):
        # Target corpus
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        doc1 = corpus.add_doc("Domain pertinence emphasizes target specificity.")
        doc1.words = [(0, 6), (7, 17), (18, 29), (30, 36), (37, 49), (50, 61)]

        # Reference corpus
        reference_corpus = Corpus()
        reference_corpus.add_layer_meta("text")
        reference_corpus.add_layer_meta("words", layer_type="span", base="text")

        ref_doc1 = reference_corpus.add_doc("General corpus contains common words.")
        ref_doc1.words = [(0, 7), (8, 14), (15, 24), (25, 31), (32, 37)]

        extractor = TermExtractor(method="domainpertinence", threshold=0.1, n=2)
        extractor.extractor.set_reference_corpus(reference_corpus)  # Set the reference corpus during initialization
        results = extractor.extract(corpus)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

    def test_domain_pertinence_extraction_strings(self):
        # Target corpus
        text_corpus = ["Domain pertinence emphasizes target specificity."]
        # Reference corpus
        reference_corpus = ["General corpus contains common words."]

        extractor = TermExtractor(method="domainpertinence", threshold=0.1, n=2)
        extractor.extractor.set_reference_corpus(reference_corpus)  # Set the reference corpus during initialization
        results = extractor.extract(text_corpus)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

