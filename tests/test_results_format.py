import io
import re
import unittest
from teanga import Corpus
from termxtract.unsupervised.base_extractor import BaseTermExtractor
from termxtract.unsupervised.term_extractor import TermExtractor
from termxtract.utils import rank_terms


TEXTS = [
    "Photosynthesis converts light energy into chemical energy in plant cells.",
    "Chlorophyll in the chloroplast absorbs light for photosynthesis.",
    "A neural network learns weights from training data.",
    "Gradient descent updates the weights of a neural network.",
    "A black hole bends light with its strong gravity.",
    "Gravity holds the planets in orbit around the sun.",
]
REFERENCE = [
    "The cat sat on the mat.",
    "A dog barked at the light in the house.",
    "We went to the shops for bread and energy drinks.",
]
STOPLIST = ["the", "a", "in", "of", "for", "with", "into", "from", "its", "around"]

# Keyword arguments for each method. "REF" is replaced by a reference corpus of the same type as the input.
METHODS = {
    "tfidf": dict(n=2),
    "ridf": dict(n=2),
    "cvalue": dict(n=2),
    "basic": dict(n=2),
    "combobasic": dict(n=2),
    "domaincoherence": dict(n=1),
    "topicmodeling": dict(n=2, num_topics=3),
    "nmf": dict(n=2, n_topics=2),
    "rake": dict(stoplist=STOPLIST, phrase_delimiters=[".", ","]),
    "domainpertinence": dict(n=2, reference_corpus="REF"),
    "weirdness": dict(n=2, reference_corpus="REF"),
    "relevance": dict(n=2, reference_corpus="REF"),
}
# Methods that score terms over the whole corpus rather than the n-grams of each document.
NO_OFFSETS = {"nmf", "domaincoherence"}


def make_teanga(texts):
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    for text in texts:
        doc = corpus.add_doc(text)
        doc.words = [match.span() for match in re.finditer(r"\w+", text)]
    return corpus


def serialize(corpus):
    buffer = io.StringIO()
    corpus.to_json(buffer)
    return buffer.getvalue()


def make_extractor(method, teanga, **overrides):
    kwargs = dict(METHODS[method], **overrides)
    if kwargs.get("reference_corpus") == "REF":
        kwargs["reference_corpus"] = make_teanga(REFERENCE) if teanga else REFERENCE
    return TermExtractor(method=method, **kwargs)


class TestResultsFormat(unittest.TestCase):
    """Every extractor returns ATEResults in the same format."""

    def assert_terms_format(self, results, doc_ids):
        self.assertEqual([doc["doc_id"] for doc in results.terms], doc_ids)
        for doc in results.terms:
            for term in doc["terms"]:
                self.assertEqual(set(term), {"term", "score"})
                self.assertIsInstance(term["term"], str)
                self.assertIsInstance(term["score"], float)
            scores = [term["score"] for term in doc["terms"]]
            self.assertEqual(scores, sorted(scores, reverse=True), "Terms should be ranked by score.")

    def test_extractors_use_base_class(self):
        for method in METHODS:
            with self.subTest(method=method):
                self.assertIsInstance(make_extractor(method, teanga=False).extractor, BaseTermExtractor)

    def test_strings(self):
        for method in METHODS:
            with self.subTest(method=method):
                results = make_extractor(method, teanga=False).extract(TEXTS)
                self.assert_terms_format(results, [f"doc_{i}" for i in range(len(TEXTS))])
                self.assertTrue(any(doc["terms"] for doc in results.terms), "Expected some terms.")
                self.assertIsNone(results.offsets)

    def test_teanga(self):
        for method in METHODS:
            with self.subTest(method=method):
                corpus = make_teanga(TEXTS)
                results = make_extractor(method, teanga=True).extract(corpus)
                doc_ids = list(corpus.doc_ids)
                self.assert_terms_format(results, doc_ids)
                self.assertTrue(any(doc["terms"] for doc in results.terms), "Expected some terms.")

                if method in NO_OFFSETS:
                    self.assertIsNone(results.offsets)
                    continue

                self.assertEqual([doc["doc_id"] for doc in results.offsets], doc_ids)
                for doc_terms, doc_offsets in zip(results.terms, results.offsets):
                    text = corpus.doc_by_id(doc_offsets["doc_id"]).text
                    terms = {term["term"] for term in doc_terms["terms"]}
                    found = set()
                    for occurrence in doc_offsets["offsets"]:
                        start, end = occurrence["offset"]
                        span = text[start:end]
                        # RAKE matches lower-cased text; the n-gram extractors keep the words as written
                        self.assertEqual(span.lower() if method == "rake" else span, occurrence["term"])
                        found.add(occurrence["term"])
                    self.assertEqual(found, terms, "Every extracted term should have its offsets.")

    def test_corpus_not_modified(self):
        """Extractors leave the input corpus unchanged, so several can run on the same Corpus object."""
        corpus = make_teanga(TEXTS)
        before = serialize(corpus)
        for _ in range(2):
            for method in METHODS:
                with self.subTest(method=method):
                    make_extractor(method, teanga=True).extract(corpus)
                    self.assertEqual(serialize(corpus), before)

    def test_threshold(self):
        for method in METHODS:
            with self.subTest(method=method):
                unfiltered = make_extractor(method, teanga=False).extract(TEXTS)
                scores = sorted(term["score"] for doc in unfiltered.terms for term in doc["terms"])
                threshold = scores[len(scores) // 2]
                results = make_extractor(method, teanga=False, threshold=threshold).extract(TEXTS)
                kept = [term["score"] for doc in results.terms for term in doc["terms"]]
                self.assertTrue(kept, "Expected some terms above the median score.")
                self.assertTrue(all(score >= threshold for score in kept))

    def test_rank_terms(self):
        self.assertEqual(
            rank_terms({"a": 1, "b": 3.0, "c": 2.0, "d": 3.0}, threshold=2.0),
            [{"term": "b", "score": 3.0}, {"term": "d", "score": 3.0}, {"term": "c", "score": 2.0}],
        )


if __name__ == "__main__":
    unittest.main()
