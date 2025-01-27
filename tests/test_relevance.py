import unittest
from teanga import Corpus
from termxtract.term_extractor import TermExtractor


class TestRelevanceExtraction(unittest.TestCase):
    """Test cases for Relevance term extraction method."""

    def test_relevance_extraction_teanga(self):
        corpus = Corpus()
        reference_corpus = Corpus()

        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        ref_doc = reference_corpus.add_doc("Generic terms and concepts are crucial.")
        ref_doc.words = [(0, 7), (8, 13), (14, 17), (18, 25), (26, 29), (30, 37)]

        doc1 = corpus.add_doc("Domain-specific terminology often involves specialized terms.")
        doc1.words = [(0, 13), (14, 24), (25, 30), (31, 38), (39, 49), (50, 59), (60, 65)]

        extractor = TermExtractor(method="relevance", reference_corpus=reference_corpus, threshold=0.1, n=2)
        results = extractor.extract(corpus)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

    def test_relevance_extraction_strings(self):
        text_corpus = ["Domain-specific terminology often involves specialized terms."]
        reference_corpus = ["Generic terms and concepts are crucial."]

        extractor = TermExtractor(method="relevance", reference_corpus=reference_corpus, threshold=0.1, n=2)
        results = extractor.extract(text_corpus)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

