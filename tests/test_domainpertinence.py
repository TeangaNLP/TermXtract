import unittest
from teanga import Corpus
from termxtract.term_extractor import TermExtractor


class TestDomainPertinenceExtraction(unittest.TestCase):
    """Test cases for Domain Pertinence term extraction method."""

    def setUp(self):
        # Set up a sample target corpus and reference corpus for Teanga
        self.target_corpus = Corpus()
        self.target_corpus.add_layer_meta("text")
        self.target_corpus.add_layer_meta("words", layer_type="span", base="text")

        doc1 = self.target_corpus.add_doc("Domain Pertinence extraction for Teanga.")
        doc1.words = [(0, 6), (7, 17), (18, 28), (29, 32), (33, 39)]

        doc2 = self.target_corpus.add_doc("The second document in the target corpus.")
        doc2.words = [(0, 3), (4, 10), (11, 19), (20, 23), (24, 30), (31, 37)]

        self.reference_corpus = Corpus()
        self.reference_corpus.add_layer_meta("text")
        self.reference_corpus.add_layer_meta("words", layer_type="span", base="text")

        ref_doc1 = self.reference_corpus.add_doc("Reference corpus document one.")
        ref_doc1.words = [(0, 9), (10, 16), (17, 25), (26, 29)]

        ref_doc2 = self.reference_corpus.add_doc("Another reference document.")
        ref_doc2.words = [(0, 7), (8, 17), (18, 26)]

        # Set up a plain text corpus for testing
        self.target_text_corpus = [
            "Domain Pertinence extraction for plain text corpus.",
            "Another document in the target corpus for testing."
        ]
        self.reference_text_corpus = [
            "Reference document one.",
            "Another document in the reference corpus."
        ]

    def test_domain_pertinence_teanga(self):
        """Test Domain Pertinence extraction for Teanga corpus."""
        extractor = TermExtractor(method="domainpertinence", threshold=0.1, n=2)
        results = extractor.extract(self.target_corpus, reference_corpus=self.reference_corpus)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

    def test_domain_pertinence_strings(self):
        """Test Domain Pertinence extraction for plain text corpus."""
        extractor = TermExtractor(method="domainpertinence", threshold=0.1, n=2)
        results = extractor.extract(self.target_text_corpus, reference_corpus=self.reference_text_corpus)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

    def test_missing_reference_corpus_teanga(self):
        """Test that an error is raised if reference corpus is missing for Teanga."""
        extractor = TermExtractor(method="domainpertinence", threshold=0.1, n=2)
        with self.assertRaises(ValueError, msg="DomainPertinence requires a reference corpus of type 'Corpus'."):
            extractor.extract(self.target_corpus)

    def test_missing_reference_corpus_strings(self):
        """Test that an error is raised if reference corpus is missing for plain text."""
        extractor = TermExtractor(method="domainpertinence", threshold=0.1, n=2)
        with self.assertRaises(ValueError, msg="DomainPertinence requires a reference corpus of type 'list[str]'."):
            extractor.extract(self.target_text_corpus)


if __name__ == '__main__':
    unittest.main()
