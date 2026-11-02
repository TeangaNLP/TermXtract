import re
import unittest

from teanga import Corpus

from termxtract.unsupervised.term_extractor import TermExtractor


class TestNMFExtraction(unittest.TestCase):
    """Test cases for NMF-based term extraction method."""

    def test_nmf_extraction_teanga(self):
        """Test NMF extraction using a Teanga corpus."""
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        doc1 = corpus.add_doc(
            "Natural language processing is an important field of AI."
        )
        doc2 = corpus.add_doc("Machine learning techniques are applied to NLP tasks.")
        doc1.words = [(0, 7), (8, 16), (17, 27), (28, 35), (36, 38), (39, 48)]
        doc2.words = [(0, 7), (8, 16), (17, 27), (28, 35), (36, 38), (39, 48)]

        extractor = TermExtractor(
            method="nmf", threshold=0.1, n_topics=2, max_features=500, n=2
        )
        results = extractor.extract(corpus)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(
            all("doc_id" in item for item in results.terms),
            "Each result should have 'doc_id'.",
        )
        for term_entry in results.terms:
            self.assertIn(
                "terms", term_entry, "Each result should have a 'terms' field."
            )
            for term_data in term_entry["terms"]:
                self.assertIn(
                    "term", term_data, "Each term entry should have a 'term'."
                )
                self.assertIn(
                    "score", term_data, "Each term entry should have a 'score'."
                )

    def test_nmf_extraction_strings(self):
        """Test NMF extraction using a list of strings."""
        text_corpus = [
            "Natural language processing is an important field of AI.",
            "Machine learning techniques are applied to NLP tasks.",
        ]
        extractor = TermExtractor(
            method="nmf", threshold=0.1, n_topics=2, max_features=500, n=2
        )
        results = extractor.extract(text_corpus)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(
            all("doc_id" in item for item in results.terms),
            "Each result should have 'doc_id'.",
        )
        for term_entry in results.terms:
            self.assertIn(
                "terms", term_entry, "Each result should have a 'terms' field."
            )
            for term_data in term_entry["terms"]:
                self.assertIn(
                    "term", term_data, "Each term entry should have a 'term'."
                )
                self.assertIn(
                    "score", term_data, "Each term entry should have a 'score'."
                )


if __name__ == "__main__":
    unittest.main()


def teanga_corpus(texts):
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    for text in texts:
        doc = corpus.add_doc(text)
        doc.words = [match.span() for match in re.finditer(r"\w+", text)]
    return corpus


def test_teanga_matches_strings():
    # Lower-case text, so the Teanga words match the lower-cased plain-text words
    texts = [
        "the chloroplast of a plant cell absorbs light energy",
        "light energy is converted into chemical energy in a plant",
        "plant cells store chemical energy as glucose",
    ]
    extractor = TermExtractor(method="nmf", n=2, n_topics=2, stopwords=None)
    from_strings = extractor.extract(texts).terms
    from_teanga = extractor.extract(teanga_corpus(texts)).terms
    assert [doc["terms"] for doc in from_teanga] == [
        doc["terms"] for doc in from_strings
    ]
    terms = {t["term"] for t in from_teanga[0]["terms"]}
    # One-letter words are kept, and no n-gram is formed across a skipped word
    assert {"a", "a plant", "of a"} <= terms
    assert not {"of plant", "plant plant"} & terms
