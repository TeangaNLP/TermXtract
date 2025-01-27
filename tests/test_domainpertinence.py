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

        doc1 = corpus.add_doc("Domain pertinence identifies specific terms.")
        doc1.words = [(0, 6), (7, 17), (18, 27), (28, 37), (38, 43), (44, 49)]

        # Reference corpus
        reference_corpus = Corpus()
        reference_corpus.add_layer_meta("text")
        reference_corpus.add_layer_meta("words", layer_type="span", base="text")

        ref_doc1 = reference_corpus.add_doc("Reference corpus contains common terms.")
        ref_doc1.words = [(0, 9), (10, 16), (17, 25), (26, 33), (34, 39), (40, 45)]

        # Initialize extractor with reference corpus
        extractor = TermExtractor(
            method="domainpertinence", reference_corpus=reference_corpus, threshold=0.1, n=2
        )
        results = extractor.extract(corpus)

        # Assertions
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")
                self.assertGreaterEqual(term_data["score"], 0, "Scores should be non-negative.")

    def test_domain_pertinence_extraction_strings(self):
        # Target corpus
        text_corpus = ["Domain pertinence identifies specific terms."]
        # Reference corpus
        reference_corpus = ["Reference corpus contains common terms."]

        # Initialize extractor with reference corpus
        extractor = TermExtractor(
            method="domainpertinence", reference_corpus=reference_corpus, threshold=0.1, n=2
        )
        results = extractor.extract(text_corpus)

        # Assertions
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")
                self.assertGreaterEqual(term_data["score"], 0, "Scores should be non-negative.")


if __name__ == "__main__":
    unittest.main()
