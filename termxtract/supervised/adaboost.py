import re
import math
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Tuple
from nltk.corpus import wordnet as wn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from ..utils import ATEResults


class AdaBoostTermExtractor:
    """AdaBoost-based term extraction using linguistic and statistical features."""

    def __init__(
        self,
        threshold: Optional[float] = None,
        n: int = 1,
        max_depth: int = 1,
        n_estimators: int = 50,
        estimator: Optional[str] = "decision_tree"
    ):
        """
        Initialize the AdaBoost extractor.

        Args:
            threshold (Optional[float]): Minimum confidence score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
            max_depth (int): Maximum depth of the decision tree base estimator.
            n_estimators (int): Number of boosting iterations.
            estimator (str): Type of base estimator ("decision_tree" or "logistic_regression").
        """
        self.threshold = threshold
        self.n = n
        self.max_depth = max_depth
        self.n_estimators = n_estimators

        if estimator == "decision_tree":
            base_estimator = DecisionTreeClassifier(max_depth=max_depth)
        else:
            raise ValueError("Currently, only 'decision_tree' is supported as a base estimator.")

        self.classifier = AdaBoostClassifier(estimator=base_estimator, n_estimators=n_estimators)
        self.is_trained = False  # Track if the model has been trained

    def generate_ngrams(self, words: List[str]) -> List[str]:
        """Generate n-grams from a list of words."""
        ngrams = []
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    ngrams.append(ngram)
        return ngrams

    def generate_ngrams_teanga(self, words_with_offsets: List[Tuple[int, int, str]]) -> List[str]:
        """Generate n-grams for a Teanga corpus."""
        ngrams = []
        for i in range(len(words_with_offsets)):
            for j in range(1, self.n + 1):
                if i + j <= len(words_with_offsets):
                    ngram = " ".join(word for _, _, word in words_with_offsets[i:i + j])
                    ngrams.append(ngram)
        return ngrams

    def compute_semantic_score(self, term: str) -> float:
        """Compute semantic content score using WordNet."""
        word_senses = wn.synsets(term)
        return len(word_senses) / (len(term.split()) + 1) if word_senses else 0.0

    def compute_greek_latin_score(self, term: str) -> int:
        """Check if a term contains Greek/Latin morphemes."""
        greek_latin_morphemes = {"bio", "neuro", "cardio", "psycho", "pharm", "ology", "itis", "oma", "osis"}
        return int(any(morpheme in term.lower() for morpheme in greek_latin_morphemes))

    def compute_collocation_score(self, term: str, corpus_ngrams: List[str]) -> float:
        """Compute collocation score using MI3."""
        term_count = corpus_ngrams.count(term)
        if term_count == 0:
            return 0.0
        total_terms = len(corpus_ngrams)
        prob_term = term_count / total_terms
        return math.pow(prob_term, 3) / (prob_term + 1e-9)

    def extract_features(self, corpus: List[str], labels: Dict[str, int]) -> Tuple[List[List[float]], List[int], List[str]]:
        """Extract features and labels for training/testing."""
        tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]
        corpus_ngrams = [self.generate_ngrams(doc) for doc in tokenized_corpus]
        corpus_ngrams_flat = [ngram for doc in corpus_ngrams for ngram in doc]

        feature_matrix = []
        y_labels = []
        terms = list(set(corpus_ngrams_flat))

        for term in terms:
            features = [
                self.compute_semantic_score(term),
                self.compute_greek_latin_score(term),
                self.compute_collocation_score(term, corpus_ngrams_flat),
            ]
            feature_matrix.append(features)
            y_labels.append(labels.get(term, 0))  # Default to 0 if not labeled

        return feature_matrix, y_labels, terms

    def train_model(self, corpus: List[str], labels: Dict[str, int]):
        """Train AdaBoost classifier using labeled data."""
        feature_matrix, y_labels, _ = self.extract_features(corpus, labels)
        self.classifier.fit(feature_matrix, y_labels)
        self.is_trained = True  # Mark model as trained

    def predict_terms_strings(self, corpus: List[str]) -> Dict[str, float]:
        """Predict term scores from a **list of strings**."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        feature_matrix, _, terms = self.extract_features(corpus, {})  # No labels needed for prediction
        probabilities = self.classifier.predict_proba(feature_matrix)[:, 1]

        return {term: prob for term, prob in zip(terms, probabilities)}

    def predict_terms_teanga(self, corpus) -> Dict[str, float]:
        """Predict term scores from a **Teanga corpus**."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        list_corpus = [" ".join(self.generate_ngrams_teanga([(start, end, doc.text[start:end]) for start, end in doc.words])) for doc_id in corpus.doc_ids for doc in [corpus.doc_by_id(doc_id)]]

        return self.predict_terms_strings(list_corpus)

    def extract_terms_teanga(self, corpus, labels: Dict[str, int]) -> ATEResults:
        """Extract terms from a **Teanga corpus** using trained AdaBoost."""
        self.train_model([" ".join(self.generate_ngrams_teanga([(start, end, doc.text[start:end]) for start, end in doc.words])) for doc_id in corpus.doc_ids for doc in [corpus.doc_by_id(doc_id)]], labels)

        term_scores = self.predict_terms_teanga(corpus)

        terms_by_doc = [{"doc_id": doc_id, "terms": [{"term": term, "score": score} for term, score in term_scores.items() if self.threshold is None or score >= self.threshold]} for doc_id in corpus.doc_ids]

        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_strings(self, corpus: List[str], labels: Dict[str, int]) -> ATEResults:
        """Extract terms from a **list of strings** using trained AdaBoost."""
        self.train_model(corpus, labels)
        term_scores = self.predict_terms_strings(corpus)

        terms_by_doc = [{"doc_id": f"doc_{i}", "terms": [{"term": term, "score": score} for term, score in term_scores.items() if self.threshold is None or score >= self.threshold]} for i in range(len(corpus))]

        return ATEResults(corpus=corpus, terms=terms_by_doc)
