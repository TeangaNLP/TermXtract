import re
import math
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Tuple
from scipy.stats import rankdata
from sklearn.utils import resample
from ..utils import ATEResults


class RogerTermExtractor:
    """Roger-based term extraction using evolutionary learning and ranking."""

    def __init__(self, population_size: int = 50, generations: int = 100, subsample_ratio: float = 0.8, n: int = 1):
        """
        Initialize the Roger extractor.

        Args:
            population_size (int): Number of ranking functions in the population.
            generations (int): Number of iterations for optimization.
            subsample_ratio (float): Fraction of training data used in each iteration for bagging.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
        """
        self.population_size = population_size
        self.generations = generations
        self.subsample_ratio = subsample_ratio
        self.n = n
        self.models = []

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

    def compute_features(self, term: str, corpus_ngrams: List[str]) -> List[float]:
        """Compute the 13 statistical measures as features."""
        term_count = corpus_ngrams.count(term)
        total_terms = len(corpus_ngrams)

        if term_count == 0:
            return [0.0] * 13

        freq = term_count / total_terms
        mi = math.log(freq + 1e-9)
        mi3 = math.pow(freq, 3) / (freq + 1e-9)
        dice = 2 * freq / (freq + freq + 1e-9)
        log_likelihood = freq * math.log(freq + 1e-9)
        occ_log_likelihood = term_count * log_likelihood
        j_measure = freq * math.log(freq + 1e-9)
        khi2 = freq * freq
        t_test = freq / math.sqrt(freq + 1e-9)

        return [freq, mi, mi3, dice, log_likelihood, occ_log_likelihood, j_measure, khi2, t_test]

    def extract_features(self, corpus: List[str], labels: Dict[str, int]) -> Tuple[List[List[float]], List[int], List[str]]:
        """Extract features for ranking training."""
        tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]
        corpus_ngrams = [self.generate_ngrams(doc) for doc in tokenized_corpus]
        corpus_ngrams_flat = [ngram for doc in corpus_ngrams for ngram in doc]

        feature_matrix = []
        y_labels = []
        terms = list(set(corpus_ngrams_flat))

        for term in terms:
            features = self.compute_features(term, corpus_ngrams_flat)
            feature_matrix.append(features)
            y_labels.append(labels.get(term, 0))  # Default to 0 if not labeled

        return feature_matrix, y_labels, terms

    def optimize_rank_function(self, X: List[List[float]], y: List[int]):
        """Evolutionary optimization of ranking function."""
        num_features = len(X[0])

        population = np.random.rand(self.population_size, num_features)
        for _ in range(self.generations):
            scores = [self.evaluate_rank_function(model, X, y) for model in population]
            best_indices = np.argsort(scores)[-self.population_size//2:]
            parents = population[best_indices]
            mutations = parents + np.random.normal(0, 0.1, parents.shape)
            population = np.vstack((parents, mutations))

        return population[np.argmax(scores)]

    def evaluate_rank_function(self, model: np.ndarray, X: List[List[float]], y: List[int]) -> float:
        """Evaluate ranking function using Wilcoxon rank test (AUC)."""
        scores = np.dot(X, model)
        ranks = rankdata(scores)
        positive_ranks = ranks[np.array(y) == 1]
        return np.mean(positive_ranks) / len(ranks)

    def train_model(self, corpus: List[str], labels: Dict[str, int]):
        """Train multiple ranking functions using bagging."""
        X, y, _ = self.extract_features(corpus, labels)
        self.models = []

        for _ in range(10):  # Bagging
            X_sample, y_sample = resample(X, y, n_samples=int(len(X) * self.subsample_ratio))
            model = self.optimize_rank_function(X_sample, y_sample)
            self.models.append(model)

    def predict_terms_strings(self, corpus: List[str]) -> Dict[str, float]:
        """Predict term rankings for a list of strings."""
        X, _, terms = self.extract_features(corpus, {})
        scores = np.median([np.dot(X, model) for model in self.models], axis=0)
        return {term: score for term, score in zip(terms, scores)}

    def predict_terms_teanga(self, corpus) -> Dict[str, float]:
        """Predict term rankings for a Teanga corpus."""
        list_corpus = [" ".join(self.generate_ngrams_teanga([(start, end, doc.text[start:end]) for start, end in doc.words])) for doc_id in corpus.doc_ids for doc in [corpus.doc_by_id(doc_id)]]
        return self.predict_terms_strings(list_corpus)

    def extract_terms_teanga(self, corpus, labels: Dict[str, int]) -> ATEResults:
        """Extract terms from a Teanga corpus using Roger."""
        self.train_model([" ".join(self.generate_ngrams_teanga([(start, end, doc.text[start:end]) for start, end in doc.words])) for doc_id in corpus.doc_ids for doc in [corpus.doc_by_id(doc_id)]], labels)

        term_scores = self.predict_terms_teanga(corpus)
        terms_by_doc = [{"doc_id": doc_id, "terms": [{"term": term, "score": score} for term, score in term_scores.items()]} for doc_id in corpus.doc_ids]

        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_strings(self, corpus: List[str], labels: Dict[str, int]) -> ATEResults:
        """Extract terms from a list of strings using Roger."""
        self.train_model(corpus, labels)
        term_scores = self.predict_terms_strings(corpus)
        return ATEResults(corpus=corpus, terms=[{"doc_id": f"doc_{i}", "terms": [{"term": term, "score": score} for term, score in term_scores.items()]} for i in range(len(corpus))])
