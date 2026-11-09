import math
import re
import unittest

import pytest
from teanga import Corpus

from termxtract.unsupervised.term_extractor import TermExtractor


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
        extractor = TermExtractor(
            method="topicmodeling", num_topics=10, threshold=0.01, n=2
        )

        # Extract terms
        results = extractor.extract(corpus)

        # Assertions
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

    def test_topic_modeling_extraction_strings(self):
        """Test TopicModeling term extraction on a list of strings."""
        # Create a string-based corpus
        text_corpus = [
            "Machine learning models are very powerful.",
            "Deep learning and neural networks are revolutionary.",
        ]

        # Initialize the TopicModelingTermExtractor through TermExtractor
        extractor = TermExtractor(
            method="topicmodeling", num_topics=10, threshold=0.01, n=2
        )

        # Extract terms
        results = extractor.extract(text_corpus)

        # Assertions
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


CORPUS = [
    "The degree of freedom of the model.",
    "Each degree of freedom adds a parameter to the model.",
    "Model selection penalises every extra parameter, as in model selection.",
    "Gradient descent finds the parameters of the model.",
]


def extract(corpus, **params):
    extractor = TermExtractor(method="topicmodeling", n=3, num_topics=3, **params)
    return extractor.extractor, extractor.extract(corpus)


def scores(results):
    by_term = {}
    for doc in results.terms:
        for term in doc["terms"]:
            by_term.setdefault(term["term"], set()).add(term["score"])
    assert all(len(values) == 1 for values in by_term.values())
    return {term: next(iter(values)) for term, values in by_term.items()}


def test_topic_model_vocabulary_is_single_content_words():
    extractor, _ = extract(CORPUS)
    vocabulary = set(extractor.vectorizer.get_feature_names_out())
    assert {"degree", "freedom", "model", "parameter"} <= vocabulary
    assert not vocabulary & {"the", "of", "a", "."}
    assert all(" " not in word for word in vocabulary)


def test_score_is_log_frequency_times_word_probabilities():
    extractor, results = extract(CORPUS)
    probabilities = extractor.word_probabilities
    by_term = scores(results)
    # "degree of freedom" occurs twice; "of" is a stopword, so it adds nothing
    assert by_term["degree of freedom"] == pytest.approx(
        math.log(2) * (probabilities["degree"] + probabilities["freedom"])
    )
    assert by_term["model"] == pytest.approx(math.log(5) * probabilities["model"])


def test_only_typical_words_count():
    extractor, results = extract(CORPUS, top_words=1)
    assert len(extractor.word_probabilities) <= 3
    by_term = scores(results)
    assert all(
        any(word in extractor.word_probabilities for word in term.split())
        for term in by_term
    )


def test_same_scores_for_both_input_types():
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    for text in CORPUS:
        doc = corpus.add_doc(text.lower())
        doc.words = [(m.start(), m.end()) for m in re.finditer(r"\w+", text)]
    assert scores(extract(corpus)[1]) == pytest.approx(scores(extract(CORPUS)[1]))


if __name__ == "__main__":
    unittest.main()
