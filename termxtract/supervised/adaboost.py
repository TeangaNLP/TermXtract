import math
from collections import Counter

import wn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from wn.morphy import Morphy

from ..stopwords import Stopwords, load_stopwords
from ..utils import ATEResults, rank_terms
from .words import tokenize_strings, tokenize_teanga

DEFAULT_WORDNET_LEXICON = "oewn:2025+"


def load_wordnet(lexicon: str = DEFAULT_WORDNET_LEXICON) -> wn.Wordnet:
    """
    Load a WordNet lexicon, downloading it on first use if it is not installed.

    Args:
        lexicon (str): The `wn` lexicon specifier (e.g. "oewn:2025+").

    Returns:
        wn.Wordnet: The lexicon, with Morphy lemmatization so inflected forms (e.g. "dogs") match.

    Raises:
        RuntimeError: If the lexicon is not installed and cannot be downloaded (e.g. when offline).
    """
    if not wn.lexicons(lexicon=lexicon):
        try:
            wn.download(lexicon, progress_handler=None)
        except Exception as e:
            raise RuntimeError(
                f"WordNet lexicon '{lexicon}' is not installed and could not be downloaded. "
                f"Install it with: python -m wn download {lexicon}"
            ) from e
    return wn.Wordnet(lexicon, lemmatizer=Morphy())


class AdaBoostTermExtractor:
    """AdaBoost-based term extraction using linguistic and statistical features."""

    def __init__(
        self,
        threshold: float | None = None,
        n: int = 1,
        max_depth: int = 1,
        n_estimators: int = 50,
        estimator: str | None = "decision_tree",
        wordnet_lexicon: str = DEFAULT_WORDNET_LEXICON,
        stopwords: Stopwords = "english",
    ):
        """
        Initialize the AdaBoost extractor.

        Args:
            threshold (Optional[float]): Minimum confidence score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
            max_depth (int): Maximum depth of the decision tree base estimator.
            n_estimators (int): Number of boosting iterations.
            estimator (str): Type of base estimator ("decision_tree" or "logistic_regression").
            wordnet_lexicon (str): `wn` lexicon used for the semantic feature. It is loaded
                (and downloaded if needed) on first use, not at construction.
            stopwords (Union[str, Iterable[str], None]): Drop extracted terms that start or
                end with a stopword: "english", a list of words, or None.
        """
        self.threshold = threshold
        self.n = n
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.wordnet_lexicon = wordnet_lexicon
        self.stopwords = load_stopwords(stopwords)
        self._wordnet: wn.Wordnet | None = None

        if estimator == "decision_tree":
            base_estimator = DecisionTreeClassifier(max_depth=max_depth)
        else:
            raise ValueError(
                "Currently, only 'decision_tree' is supported as a base estimator."
            )

        self.classifier = AdaBoostClassifier(
            estimator=base_estimator, n_estimators=n_estimators
        )
        self.is_trained = False  # Track if the model has been trained

    def generate_ngrams(self, words: list[str]) -> list[str]:
        """Generate n-grams from a list of words."""
        ngrams = []
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i : i + j])
                    ngrams.append(ngram)
        return ngrams

    @property
    def wordnet(self) -> wn.Wordnet:
        """The WordNet lexicon, loaded on first access."""
        if self._wordnet is None:
            self._wordnet = load_wordnet(self.wordnet_lexicon)
        return self._wordnet

    def compute_semantic_score(self, term: str) -> float:
        """Compute semantic content score using WordNet."""
        word_senses = self.wordnet.synsets(term)
        return len(word_senses) / (len(term.split()) + 1) if word_senses else 0.0

    def compute_greek_latin_score(self, term: str) -> int:
        """Check if a term contains Greek/Latin morphemes."""
        greek_latin_morphemes = {
            "bio",
            "neuro",
            "cardio",
            "psycho",
            "pharm",
            "ology",
            "itis",
            "oma",
            "osis",
        }
        return int(any(morpheme in term.lower() for morpheme in greek_latin_morphemes))

    def compute_collocation_score(self, term: str, ngram_counts: Counter[str]) -> float:
        """Compute collocation score using MI3."""
        term_count = ngram_counts[term]
        if term_count == 0:
            return 0.0
        total_terms = ngram_counts.total()
        prob_term = term_count / total_terms
        return math.pow(prob_term, 3) / (prob_term + 1e-9)

    def extract_features(
        self, docs: list[list[str]], labels: dict[str, int]
    ) -> tuple[list[list[float]], list[int], list[str]]:
        """
        Extract features and labels for training/testing.

        Args:
            docs (list[list[str]]): The words of each document.
            labels (dict[str, int]): Labels of the candidates; unlabelled ones get 0.

        Returns:
            The feature matrix, the labels and the candidates, in order of first
            occurrence.
        """
        corpus_ngrams_flat = [
            ngram for doc in docs for ngram in self.generate_ngrams(doc)
        ]
        ngram_counts = Counter(corpus_ngrams_flat)

        feature_matrix = []
        y_labels = []
        terms = list(ngram_counts)  # In order of first occurrence

        for term in terms:
            features = [
                self.compute_semantic_score(term),
                self.compute_greek_latin_score(term),
                self.compute_collocation_score(term, ngram_counts),
            ]
            feature_matrix.append(features)
            y_labels.append(labels.get(term, 0))  # Default to 0 if not labeled

        return feature_matrix, y_labels, terms

    def train(self, docs: list[list[str]], labels: dict[str, int]):
        """Train the AdaBoost classifier on the labelled candidates in `docs`."""
        feature_matrix, y_labels, _ = self.extract_features(docs, labels)
        self.classifier.fit(feature_matrix, y_labels)
        self.is_trained = True  # Mark model as trained

    def predict(self, docs: list[list[str]]) -> dict[str, float]:
        """Score the candidates in `docs` with the trained classifier."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        feature_matrix, _, terms = self.extract_features(docs, {})
        probabilities = self.classifier.predict_proba(feature_matrix)[:, 1]

        return {term: prob for term, prob in zip(terms, probabilities)}

    def train_model(self, corpus: list[str], labels: dict[str, int]):
        """Train AdaBoost classifier using labeled data."""
        self.train(tokenize_strings(corpus), labels)

    def predict_terms_strings(self, corpus: list[str]) -> dict[str, float]:
        """Predict term scores from a **list of strings**."""
        return self.predict(tokenize_strings(corpus))

    def predict_terms_teanga(self, corpus) -> dict[str, float]:
        """Predict term scores from a **Teanga corpus**."""
        return self.predict(tokenize_teanga(corpus))

    def _results(self, corpus, doc_ids: list[str], docs, labels) -> ATEResults:
        self.train(docs, labels)
        term_scores = self.predict(docs)
        # Each document lists the corpus-level scores of its own candidates
        return ATEResults(
            corpus=corpus,
            terms=[
                {
                    "doc_id": doc_id,
                    "terms": rank_terms(
                        {
                            term: term_scores[term]
                            for term in dict.fromkeys(self.generate_ngrams(words))
                        },
                        self.threshold,
                        self.stopwords,
                    ),
                }
                for doc_id, words in zip(doc_ids, docs)
            ],
        )

    def extract_terms_teanga(self, corpus, labels: dict[str, int]) -> ATEResults:
        """Extract terms from a **Teanga corpus** using trained AdaBoost."""
        return self._results(
            corpus, list(corpus.doc_ids), tokenize_teanga(corpus), labels
        )

    def extract_terms_strings(
        self, corpus: list[str], labels: dict[str, int]
    ) -> ATEResults:
        """Extract terms from a **list of strings** using trained AdaBoost."""
        return self._results(
            corpus,
            [f"doc_{i}" for i in range(len(corpus))],
            tokenize_strings(corpus),
            labels,
        )
