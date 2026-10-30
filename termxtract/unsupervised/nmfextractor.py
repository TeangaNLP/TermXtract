import re

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from ..stopwords import Stopwords
from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class NMFTermExtractor(BaseTermExtractor):
    """NMF-based term extraction with n-gram support."""

    def __init__(
        self,
        threshold: float | None = None,
        n: int = 1,
        n_topics: int = 10,
        max_features: int | None = None,
        stopwords: Stopwords = "english",
    ):
        """
        Initialize the NMF extractor.

        Args:
            threshold (Optional[float]): Minimum score threshold for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
            n_topics (int): Number of topics for NMF-based extraction.
            max_features (Optional[int]): Maximum number of features for TF-IDF vectorization.
            stopwords (Union[str, Iterable[str], None]): Drop candidates that start or end
                with a stopword: "english", a list of words, or None.
        """
        super().__init__(threshold=threshold, n=n, stopwords=stopwords)
        self.n_topics = n_topics
        self.max_features = max_features

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """Extract terms from a plain list of strings using NMF."""
        tokenized_corpus = [
            " ".join(re.findall(r"\b\w+\b", doc.lower())) for doc in corpus
        ]

        # Apply TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=self.max_features, ngram_range=(1, self.n)
        )
        tfidf_matrix = vectorizer.fit_transform(tokenized_corpus)
        feature_names = vectorizer.get_feature_names_out()

        # Apply NMF for topic modeling
        nmf = NMF(n_components=self.n_topics, random_state=42)
        nmf.fit(tfidf_matrix)
        H = nmf.components_

        # Compute term scores from NMF weight matrix H
        term_scores = {
            feature_names[i]: max(H[:, i]) for i in range(len(feature_names))
        }

        # Prepare results
        terms_by_doc = [
            self.doc_terms(f"doc_{idx}", term_scores) for idx in range(len(corpus))
        ]

        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_teanga(self, corpus) -> ATEResults:
        """Extract terms from a Teanga corpus using NMF."""

        ngrams_by_doc = {}
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            ngrams_by_doc[doc_id] = self.generate_ngrams_teanga(self.teanga_words(doc))

        tokenized_corpus = [
            " ".join(ngram for ngram, _ in ngrams) for ngrams in ngrams_by_doc.values()
        ]

        # Apply TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=self.max_features, ngram_range=(1, self.n)
        )
        tfidf_matrix = vectorizer.fit_transform(tokenized_corpus)
        feature_names = vectorizer.get_feature_names_out()

        # Apply NMF for topic modeling
        nmf = NMF(n_components=self.n_topics, random_state=42)
        nmf.fit(tfidf_matrix)
        H = nmf.components_

        # Compute term scores from NMF weight matrix H
        term_scores = {
            feature_names[i]: max(H[:, i]) for i in range(len(feature_names))
        }

        # Prepare results
        terms_by_doc = [
            self.doc_terms(doc_id, term_scores) for doc_id in corpus.doc_ids
        ]

        return ATEResults(corpus=corpus, terms=terms_by_doc)
