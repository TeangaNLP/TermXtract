import unittest
from teanga import Corpus
from termxtract.term_extractor import TermExtractor


class TestComboBasicExtraction(unittest.TestCase):
    """Test cases for ComboBasic term extraction method."""

    def test_combobasic_extraction_teanga(self):
        # Step 1: Create a Teanga corpus and add layers
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        # Step 2: Add documents with word offsets
        doc1 = corpus.add_doc("This is an example document.")
        doc1.words = [(0, 4), (5, 7), (8, 10), (11, 18), (19, 28)]

        doc2 = corpus.add_doc("ComboBasic is a term extraction algorithm.")
        doc2.words = [(0, 10), (11, 13), (14, 15), (16, 20), (21, 30), (31, 40)]

        # Step 3: Initialize TermExtractor with ComboBasic
        extractor = TermExtractor(method="combobasic", alpha=0.6, beta=0.4, threshold=0.1, n=2)

        # Step 4: Perform term extraction
        results = extractor.extract(corpus)

        # Step 5: Assertions for structure and terms
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

    def test_combobasic_extraction_strings(self):
        # Step 1: Define a list of strings
        text_corpus = [
            "This is the first document.",
            "ComboBasic algorithm identifies specific terms."
        ]

        # Step 2: Initialize TermExtractor with ComboBasic
        extractor = TermExtractor(method="combobasic", alpha=0.7, beta=0.3, threshold=0.1, n=2)

        # Step 3: Perform term extraction
        results = extractor.extract(text_corpus)

        # Step 4: Assertions for structure and terms
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")


if __name__ == "__main__":
    unittest.main()
