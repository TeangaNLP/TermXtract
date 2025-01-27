import re
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from .utils import ATEResults


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
        self.topic_model = None
        self.dictionary = None

    def preprocess_teanga(self, corpus) -> List[List[str]]:
        """
        Preprocess a Teanga corpus for topic modeling.

        Args:
            corpus: Teanga Corpus object.

        Returns:
            List[List[str]]: Tokenized documents as lists of words.
        """
        tokenized_corpus = []
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            words = [doc.text[start:end].lower() for start, end in doc.words]
            tokenized_corpus.append(words)
        return tokenized_corpus

    def preprocess_strings(self, corpus: List[str]) -> List[List[str]]:
        """
        Preprocess a list of strings for topic modeling.

        Args:
            corpus (List[str]): List of documents.

        Returns:
            List[List[str]]: Tokenized and preprocessed documents.
        """
        return [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]

    def train_topic_model(self, corpus: List[List[str]]) -> None:
        """
        Train a topic model on the given tokenized corpus.

        Args:
            corpus (List[List[str]]): Tokenized documents as lists of words.
        """
        self.dictionary = Dictionary(corpus)
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in corpus]
        self.topic_model = LdaModel(
            corpus=bow_corpus, id2word=self.dictionary, num_topics=self.num_topics, passes=10
        )

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
        topic_word_distributions = self.topic_model.get_topics()
        scores = {}
        for ngram in ngrams:
            words = ngram.split()
            tf = ngrams.count(ngram)
            max_probs = [
                max(topic_word_distributions[:, self.dictionary.token2id[word]])
                for word in words if word in self.dictionary.token2id
            ]
            if max_probs:
                scores[ngram] = math.log(tf) * sum(max_probs)
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
        if not self.topic_model:
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
        if not self.topic_model:
            self.train_topic_model(tokenized_corpus)

        terms_by_doc = []
        for idx, doc in enumerate(tokenized_corpus):
            ngrams = self.generate_ngrams_strings(doc)
            scores = self.compute_ntm_scores(ngrams)
            terms = [
                {"term": ngram, "score": score}
                for ngram, score in scores.items()
                if self.threshold is None or score >= self.threshold
            ]
            terms_by_doc.append({"doc_id": f"doc_{idx}", "terms": terms})

        return ATEResults(corpus=corpus, terms=terms_by_doc)

