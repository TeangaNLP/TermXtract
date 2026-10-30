import math
import re
from collections import Counter

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from teanga import Corpus

from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class TopicModelingTermExtractor(BaseTermExtractor):
    """Topic Modeling-based term extraction with n-gram support."""

    def __init__(
        self, num_topics: int = 20, threshold: float | None = None, n: int = 1
    ):
        """
        Initialize the Topic Modeling extractor.

        Args:
            num_topics (int): Number of topics for topic modeling.
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
        """
        super().__init__(threshold=threshold, n=n)
        self.num_topics = num_topics
        self.lda_model: LatentDirichletAllocation | None = None
        self.vectorizer: CountVectorizer | None = None

    def preprocess_teanga(self, corpus: Corpus) -> list[str]:
        """
        Preprocess a Teanga corpus for topic modeling.

        Args:
            corpus: Teanga Corpus object.

        Returns:
            List[str]: Tokenized documents as raw text.
        """
        processed_corpus = []
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            words = [doc.text[start:end].lower() for start, end in doc.words]
            processed_corpus.append(" ".join(words))
        return processed_corpus

    def preprocess_strings(self, corpus: list[str]) -> list[str]:
        """
        Preprocess a list of strings for topic modeling.

        Args:
            corpus (List[str]): List of documents.

        Returns:
            List[str]: Preprocessed documents.
        """
        return [" ".join(re.findall(r"\b\w+\b", doc.lower())) for doc in corpus]

    def train_topic_model(self, corpus: list[str]) -> None:
        """
        Train an LDA topic model on the given corpus.

        Args:
            corpus (List[str]): List of preprocessed documents.
        """
        self.vectorizer = CountVectorizer(ngram_range=(1, self.n))
        doc_term_matrix = self.vectorizer.fit_transform(corpus)
        self.lda_model = LatentDirichletAllocation(
            n_components=self.num_topics, max_iter=10, random_state=42
        )
        self.lda_model.fit(doc_term_matrix)

    def compute_ntm_scores(self, ngrams: list[str]) -> dict[str, float]:
        """
        Compute NTM scores for n-grams.

        Args:
            ngrams (List[str]): List of n-grams.

        Returns:
            Dict[str, float]: NTM scores for each n-gram.
        """
        if not self.lda_model or not self.vectorizer:
            raise ValueError("LDA model is not trained yet.")

        term_frequencies = Counter(ngrams)
        topic_word_distributions = (
            self.lda_model.components_
            / self.lda_model.components_.sum(axis=1)[:, np.newaxis]
        )
        scores = {}

        for ngram in ngrams:
            words = ngram.split()
            tf = term_frequencies[ngram]
            max_probs = []

            for word in words:
                try:
                    idx = self.vectorizer.vocabulary_[word]
                    max_prob = max(topic_word_distributions[:, idx])
                    max_probs.append(max_prob)
                except KeyError:
                    pass  # Ignore words that are not in the vocabulary

            if max_probs:
                scores[ngram] = math.log(tf + 1) * sum(max_probs)

        return scores

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using topic modeling.

        Args:
            corpus: Teanga Corpus object.

        Returns:
            ATEResults: Results with ranked terms and scores, and term offsets.
        """
        tokenized_corpus = self.preprocess_teanga(corpus)
        if not self.lda_model:
            self.train_topic_model(tokenized_corpus)

        terms_by_doc = []
        offsets_by_doc = []
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            ngrams_with_offsets = self.generate_ngrams_teanga(self.teanga_words(doc))
            scores = self.compute_ntm_scores(
                [ngram for ngram, _ in ngrams_with_offsets]
            )
            doc_terms = self.doc_terms(doc_id, scores)
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
        tokenized_corpus = self.preprocess_strings(corpus)
        if not self.lda_model:
            self.train_topic_model(tokenized_corpus)

        terms_by_doc = []
        for idx, doc in enumerate(tokenized_corpus):
            ngrams = self.generate_ngrams_strings(doc.split())
            terms_by_doc.append(
                self.doc_terms(f"doc_{idx}", self.compute_ntm_scores(ngrams))
            )

        return ATEResults(corpus=corpus, terms=terms_by_doc)
