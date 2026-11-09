import math
import re
from collections import Counter

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from teanga import Corpus

from ..stopwords import Stopwords
from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


def _words(words: list[str]) -> list[str]:
    """Pass a document that is already split into words through `CountVectorizer`."""
    return words


class TopicModelingTermExtractor(BaseTermExtractor):
    r"""
    Topic-model term extraction, after the Novel Topic Model of Li et al. (2013) as
    implemented in ATR4S (Astrakhantsev, 2018).

    An LDA topic model is trained on the single words of the corpus, leaving out
    stopwords and words without a letter or digit. The typical words are the
    `top_words` most probable words of each topic. A candidate \( t \) with words
    \( w_1 \dots w_L \) and frequency \( f(t) \) in the corpus scores
    \( \log f(t) \cdot \sum_j \max_k \phi_{k w_j} \), where \( \phi_k \) is the word
    distribution of topic \( k \), and words that are not typical count 0. Scores are
    computed over the whole corpus, so a term has the same score in every document it
    occurs in.

    Unlike the Novel Topic Model, the topic model has no background or
    document-specific topics, and is trained with scikit-learn's variational LDA rather
    than Gibbs sampling.
    """

    def __init__(
        self,
        num_topics: int = 20,
        top_words: int = 200,
        threshold: float | None = None,
        n: int = 1,
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        lemma_layer: str | None = None,
    ):
        """
        Initialize the Topic Modeling extractor.

        Args:
            num_topics (int): Number of topics for topic modeling.
            top_words (int): The number of most probable words of each topic that count
                towards the scores of candidates.
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
            stopwords (Union[str, Iterable[str], None]): Drop candidates that start or end
                with a stopword, and leave stopwords out of the topic model: "english", a
                list of words, or None.
            words_layer (Optional[str]): For Teanga input, the span layer holding the
                words, or None for `words` or, failing that, `tokens`.
            lemma_layer (Optional[str]): For Teanga input, a sequence layer holding the
                lemma of each word, to count inflected forms such as "cell" and "cells"
                as one term, or None to use the words as written.
        """
        super().__init__(
            threshold=threshold,
            n=n,
            stopwords=stopwords,
            words_layer=words_layer,
            lemma_layer=lemma_layer,
        )
        self.num_topics = num_topics
        self.top_words = top_words
        self.lda_model: LatentDirichletAllocation | None = None
        self.vectorizer: CountVectorizer | None = None
        self.word_probabilities: dict[str, float] = {}

    def topic_words(self, words: list[str]) -> list[str]:
        """
        Get the words of a document that the topic model is trained on.

        Args:
            words (List[str]): The words of the document.

        Returns:
            List[str]: The words, lower-cased, without stopwords or words that have no
            letter or digit.
        """
        lowered = (word.lower() for word in words)
        return [
            word
            for word in lowered
            if word not in self.stopwords and any(c.isalnum() for c in word)
        ]

    def train_topic_model(self, docs: list[list[str]]) -> None:
        """
        Train an LDA topic model and find the typical words of its topics.

        Args:
            docs (List[List[str]]): The words of each document, as given by
                `topic_words`.
        """
        self.vectorizer = CountVectorizer(analyzer=_words, lowercase=False)
        doc_term_matrix = self.vectorizer.fit_transform(docs)
        self.lda_model = LatentDirichletAllocation(
            n_components=self.num_topics, max_iter=10, random_state=42
        )
        self.lda_model.fit(doc_term_matrix)

        vocabulary = self.vectorizer.get_feature_names_out()
        phi = (
            self.lda_model.components_
            / self.lda_model.components_.sum(axis=1)[:, np.newaxis]
        )
        typical = {
            index
            for topic in phi
            for index in np.argsort(-topic, kind="stable")[: self.top_words]
        }
        self.word_probabilities = {
            vocabulary[index]: float(phi[:, index].max()) for index in sorted(typical)
        }

    def compute_ntm_scores(self, ngrams: list[str]) -> dict[str, float]:
        """
        Score the candidates of a corpus.

        Args:
            ngrams (List[str]): Every n-gram occurrence in the corpus.

        Returns:
            Dict[str, float]: The score of each candidate with at least one typical word.
        """
        if not self.lda_model:
            raise ValueError("LDA model is not trained yet.")

        scores = {}
        for ngram, frequency in Counter(ngrams).items():
            probabilities = [
                self.word_probabilities[word]
                for word in ngram.lower().split()
                if word in self.word_probabilities
            ]
            if probabilities:
                scores[ngram] = math.log(frequency) * sum(probabilities)
        return scores

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using topic modeling.

        Args:
            corpus: Teanga Corpus object.

        Returns:
            ATEResults: Results with ranked terms and scores, and term offsets.
        """
        words_by_doc = {
            doc_id: self.teanga_words(corpus.doc_by_id(doc_id))
            for doc_id in corpus.doc_ids
        }
        if not self.lda_model:
            self.train_topic_model(
                [
                    self.topic_words([text for _, _, text in words])
                    for words in words_by_doc.values()
                ]
            )
        ngrams_by_doc = {
            doc_id: self.generate_ngrams_teanga(words)
            for doc_id, words in words_by_doc.items()
        }
        scores = self.compute_ntm_scores(
            [ngram for ngrams in ngrams_by_doc.values() for ngram, _ in ngrams]
        )

        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            doc_terms = self.doc_terms(
                doc_id,
                {
                    ngram: scores[ngram]
                    for ngram, _ in ngrams_with_offsets
                    if ngram in scores
                },
            )
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(
                self.doc_offsets(doc_id, ngrams_with_offsets, doc_terms)
            )

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """
        Extract terms from a list of strings using topic modeling.

        Args:
            corpus (List[str]): List of documents as strings.

        Returns:
            ATEResults: Results with ranked terms and scores.
        """
        words_by_doc = [re.findall(r"\b\w+\b", doc.lower()) for doc in corpus]
        if not self.lda_model:
            self.train_topic_model([self.topic_words(words) for words in words_by_doc])
        ngrams_by_doc = [self.generate_ngrams_strings(words) for words in words_by_doc]
        scores = self.compute_ntm_scores(
            [ngram for ngrams in ngrams_by_doc for ngram in ngrams]
        )

        terms_by_doc = [
            self.doc_terms(
                f"doc_{idx}",
                {ngram: scores[ngram] for ngram in ngrams if ngram in scores},
            )
            for idx, ngrams in enumerate(ngrams_by_doc)
        ]
        return ATEResults(corpus=corpus, terms=terms_by_doc)
