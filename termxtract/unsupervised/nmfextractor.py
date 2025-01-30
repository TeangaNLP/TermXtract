import re
import math
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from ..utils import ATEResults


class NMFTermExtractor:
    """NMF-based term extraction with n-gram support."""

    def __init__(self, threshold: Optional[float] = None, n: int = 1, n_topics: int = 10, max_features: Optional[int] = None):
        """
        Initialize the NMF extractor.

        Args:
            threshold (Optional[float]): Minimum score threshold for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
            n_topics (int): Number of topics for NMF-based extraction.
            max_features (Optional[int]): Maximum number of features for TF-IDF vectorization.
        """
        self.threshold = threshold
        self.n = n
        self.n_topics = n_topics
        self.max_features = max_features

    def generate_ngrams_strings(self, words: List[str]) -> List[str]:
        """Generate n-grams for a plain list of strings."""
        ngrams = []
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    ngrams.append(ngram)
        return ngrams

    def generate_ngrams_teanga(self, words_with_offsets: List[Tuple[int, int, str]]) -> List[Tuple[str, Tuple[int, int]]]:
        """Generate n-grams with offsets for a Teanga corpus."""
        ngrams = []
        words = [text for _, _, text in words_with_offsets]
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    start_offset = words_with_offsets[i][0]
                    end_offset = words_with_offsets[i + j - 1][1]
                    ngrams.append((ngram, (start_offset, end_offset)))
        return ngrams

    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        """Extract terms from a plain list of strings using NMF."""
        tokenized_corpus = [" ".join(re.findall(r'\b\w+\b', doc.lower())) for doc in corpus]

        # Apply TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, self.n))
        tfidf_matrix = vectorizer.fit_transform(tokenized_corpus)
        feature_names = vectorizer.get_feature_names_out()

        # Apply NMF for topic modeling
        nmf = NMF(n_components=self.n_topics, random_state=42)
        W = nmf.fit_transform(tfidf_matrix)
        H = nmf.components_

        # Compute term scores from NMF weight matrix H
        term_scores = {feature_names[i]: max(H[:, i]) for i in range(len(feature_names))}

        # Apply threshold filtering
        filtered_terms = {term: score for term, score in term_scores.items() if self.threshold is None or score >= self.threshold}

        # Prepare results
        terms_by_doc = []
        for idx, doc_text in enumerate(corpus):
            doc_id = f"doc_{idx}"
            terms = [{"term": term, "score": score} for term, score in filtered_terms.items()]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_teanga(self, corpus) -> ATEResults:
        """Extract terms from a Teanga corpus using NMF."""
        corpus.add_layer_meta("terms", layer_type="span", base="text")

        ngrams_by_doc = {}
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            words_with_offsets = [(start, end, doc.text[start:end]) for start, end in doc.words]
            ngrams_with_offsets = self.generate_ngrams_teanga(words_with_offsets)
            ngrams_by_doc[doc_id] = ngrams_with_offsets

        tokenized_corpus = [" ".join(ngram for ngram, _ in ngrams) for ngrams in ngrams_by_doc.values()]

        # Apply TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, self.n))
        tfidf_matrix = vectorizer.fit_transform(tokenized_corpus)
        feature_names = vectorizer.get_feature_names_out()

        # Apply NMF for topic modeling
        nmf = NMF(n_components=self.n_topics, random_state=42)
        W = nmf.fit_transform(tfidf_matrix)
        H = nmf.components_

        # Compute term scores from NMF weight matrix H
        term_scores = {feature_names[i]: max(H[:, i]) for i in range(len(feature_names))}

        # Apply threshold filtering
        filtered_terms = {term: score for term, score in term_scores.items() if self.threshold is None or score >= self.threshold}

        # Prepare results
        terms_by_doc = []
        for doc_id in corpus.doc_ids:
            terms = [{"term": term, "score": score} for term, score in filtered_terms.items()]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=corpus, terms=terms_by_doc)
