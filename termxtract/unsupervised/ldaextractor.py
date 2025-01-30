import re
import math
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Tuple
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from ..utils import ATEResults


class TopicModelingTermExtractor:
    """Topic Modeling-based term extraction with n-gram support."""

    def __init__(self, num_topics: int = 20, threshold: Optional[float] = None, n: int = 1):
        """
        Initialize the Topic Modeling extractor.

        Args:
            num_topics (int): Number of topics for topic modeling.
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
        """
        self.num_topics = num_topics
        self.threshold = threshold
        self.n = n
        self.lda_model = None
        self.vectorizer = None

    def preprocess_teanga(self, corpus) -> List[str]:
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

    def preprocess_strings(self, corpus: List[str]) -> List[str]:
        """
        Preprocess a list of strings for topic modeling.

        Args:
            corpus (List[str]): List of documents.

        Returns:
            List[str]: Preprocessed documents.
        """
        return [" ".join(re.findall(r'\b\w+\b', doc.lower())) for doc in corpus]

    def train_topic_model(self, corpus: List[str]) -> None:
        """
        Train an LDA topic model on the given corpus.

        Args:
            corpus (List[str]): List of preprocessed documents.
        """
        self.vectorizer = CountVectorizer(ngram_range=(1, self.n))
        doc_term_matrix = self.vectorizer.fit_transform(corpus)
        self.lda_model = LatentDirichletAllocation(n_components=self.num_topics, max_iter=10, random_state=42)
        self.lda_model.fit(doc_term_matrix)

    def generate_ngrams_teanga(self, words_with_offsets: List[Tuple[int, int, str]]) -> List[str]:
        """
        Generate n-grams with offsets for a Teanga corpus.

        Args:
            words_with_offsets (List[Tuple[int, int, str]]): List of (start, end, text) tuples for words.

        Returns:
            List[str]: List of n-grams.
        """
        ngrams = []
        words = [text for _, _, text in words_with_offsets]
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    ngrams.append(ngram)
        return ngrams

    def generate_ngrams_strings(self, words: List[str]) -> List[str]:
        """
        Generate n-grams for a list of words.

        Args:
            words (List[str]): List of words.

        Returns:
            List[str]: List of n-grams.
        """
        ngrams = []
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    ngrams.append(ngram)
        return ngrams

    def compute_ntm_scores(self, ngrams: List[str]) -> Dict[str, float]:
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
        topic_word_distributions = self.lda_model.components_ / self.lda_model.components_.sum(axis=1)[:, np.newaxis]
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

    def extract_terms_teanga(self, corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using topic modeling.

        Args:
            corpus: Teanga Corpus object.

        Returns:
            ATEResults: Results with terms and scores.
        """
        tokenized_corpus = self.preprocess_teanga(corpus)
        if not self.lda_model:
            self.train_topic_model(tokenized_corpus)

        ngrams_by_doc = {}
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            words_with_offsets = [(start, end, doc.text[start:end]) for start, end in doc.words]
            ngrams_by_doc[doc_id] = self.generate_ngrams_teanga(words_with_offsets)

        terms_by_doc = []
        for doc_id, ngrams in ngrams_by_doc.items():
            scores = self.compute_ntm_scores(ngrams)
            terms = [
                {"term": ngram, "score": score}
                for ngram, score in scores.items()
                if self.threshold is None or score >= self.threshold
            ]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        """
        Extract terms from a list of strings using topic modeling.

        Args:
            corpus (List[str]): List of documents as strings.

        Returns:
            ATEResults: Results with terms and scores.
        """
        tokenized_corpus = self.preprocess_strings(corpus)
        if not self.lda_model:
            self.train_topic_model(tokenized_corpus)

        terms_by_doc = []
        for idx, doc in enumerate(tokenized_corpus):
            ngrams = self.generate_ngrams_strings(doc.split())
            scores = self.compute_ntm_scores(ngrams)
            terms = [
                {"term": ngram, "score": score}
                for ngram, score in scores.items()
                if self.threshold is None or score >= self.threshold
            ]
            terms_by_doc.append({"doc_id": f"doc_{idx}", "terms": terms})

        return ATEResults(corpus=corpus, terms=terms_by_doc)
