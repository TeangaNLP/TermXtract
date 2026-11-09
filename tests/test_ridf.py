import math
import re
import unittest

from teanga import Corpus

from termxtract.unsupervised.term_extractor import TermExtractor


class TestRIDFExtraction(unittest.TestCase):
    def test_ridf_extraction_teanga(self):
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        doc1 = corpus.add_doc("This is an example document.")
        doc1.words = [(0, 4), (5, 7), (8, 10), (11, 18), (19, 28)]

        doc2 = corpus.add_doc("RIDF extraction with unigrams, bigrams, and trigrams.")
        doc2.words = [(0, 4), (5, 15), (16, 19), (20, 28), (29, 37), (38, 41), (42, 50)]

        extractor = TermExtractor(method="ridf", threshold=0.1, n=3)
        results = extractor.extract(corpus)

        self.assertIsNotNone(results.terms, "RIDF terms should not be None.")
        self.assertTrue(
            all("doc_id" in item for item in results.terms),
            "Each result should have 'doc_id'.",
        )

    def test_ridf_extraction_strings(self):
        text_corpus = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]

        extractor = TermExtractor(method="ridf", threshold=0.1, n=2)
        results = extractor.extract(text_corpus)

        self.assertIsNotNone(results.terms, "RIDF terms should not be None.")
        self.assertTrue(
            all("doc_id" in item for item in results.terms),
            "Each result should have 'doc_id'.",
        )


def scores(corpus, n=1):
    results = TermExtractor(method="ridf", n=n, stopwords=None).extract(corpus)
    by_term = {}
    for doc in results.terms:
        for term in doc["terms"]:
            by_term.setdefault(term["term"], set()).add(term["score"])
    assert all(len(values) == 1 for values in by_term.values())
    return {term: next(iter(values)) for term, values in by_term.items()}


# D = 4 documents
SCORED_CORPUS = ("cell cell cell", "cell protein", "protein", "protein")


class TestRIDFScores(unittest.TestCase):
    def test_formula(self):
        by_term = scores(list(SCORED_CORPUS))
        # cell: cf 4, df 2
        self.assertAlmostEqual(
            by_term["cell"], math.log2(4 / 2) + math.log2(1 - math.exp(-4 / 4))
        )
        # protein: cf 3, df 3
        self.assertAlmostEqual(
            by_term["protein"], math.log2(4 / 3) + math.log2(1 - math.exp(-3 / 4))
        )

    def test_concentrated_terms_score_higher(self):
        # cell occurs more often but in fewer documents than protein
        by_term = scores(list(SCORED_CORPUS))
        self.assertGreater(by_term["cell"], by_term["protein"])

    def test_spread_terms_score_about_zero(self):
        # A term seen once in each document is as spread as Poisson predicts, or more
        by_term = scores(["word"] * 100)
        self.assertLess(by_term["word"], 0)

    def test_rare_terms(self):
        # cf = df = 1: IDF is close to the Poisson prediction, so RIDF is close to 0
        by_term = scores([f"word{i}" for i in range(1000)])
        self.assertAlmostEqual(by_term["word0"], 0, delta=0.001)

    def test_same_score_for_both_input_types(self):
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")
        for text in SCORED_CORPUS:
            doc = corpus.add_doc(text)
            doc.words = [(m.start(), m.end()) for m in re.finditer(r"\w+", text)]
        self.assertEqual(scores(corpus), scores(list(SCORED_CORPUS)))


if __name__ == "__main__":
    unittest.main()
