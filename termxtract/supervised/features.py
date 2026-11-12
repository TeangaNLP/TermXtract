import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from teanga import Corpus

from ..association import MEASURES, NgramCounts
from ..candidates import split_words
from ..datasets.base import GOLD_LAYER, gold_terms
from ..evaluation import normalize_term
from ..layers import POS_LAYERS, find_layer, pos_tags, words_with_offsets
from ..stopwords import Stopwords
from ..unsupervised.base_extractor import BaseTermExtractor
from ..unsupervised.reference_extractor import ReferenceCorpus
from ..unsupervised.term_extractor import TermExtractor
from ..utils import ATEResults, rank_terms

TERMHOOD_METHODS = ("tfidf", "ridf", "cvalue", "basic", "combobasic", "domaincoherence")
"""The unsupervised methods whose scores are features."""

REFERENCE_METHODS = ("relevance", "domainpertinence", "weirdness")
"""The unsupervised methods whose scores are features when a reference corpus is given."""

LEARNED = "learned"
"""The `pos_pattern` that learns the tag sequences of the terms from the training data."""


@dataclass
class _Candidates:
    """The occurrences of the candidates of a corpus."""

    docs: list[list[str]]
    """The words of each document, as lemmas if `lemma_layer` is set."""
    occurrences: list[list[tuple[str, int, int]]]
    """Each occurrence of a candidate in each document, with its start and end word."""
    tags: list[list[str]] | None
    """The part-of-speech tag of each word, or None without a part-of-speech layer."""
    surface: list[list[str]] | None
    """For Teanga input, each word as written."""
    spans: list[list[tuple[int, int]]] | None
    """For Teanga input, the character offsets of each word."""


class FeatureTermExtractor:
    """
    Supervised term extraction with a classifier over features of the candidates, in
    the style of HAMLET (Rigouts Terryn, Hoste & Lefever, 2021).

    The candidates are those of the unsupervised methods (see `BaseTermExtractor`), and
    each is described by:

    - **termhood:** its scores by the unsupervised methods in `TERMHOOD_METHODS` and,
      given a reference corpus, `REFERENCE_METHODS`;
    - **unithood:** the association measures of `termxtract.association.MEASURES`, each
      over the weakest split of the candidate into two parts, or 0 for single words;
    - **frequency:** its number of occurrences and the share of documents it occurs in;
    - **shape:** its numbers of words, characters, digits, hyphens and other
      characters that are neither letters, digits nor spaces, and, for Teanga input,
      the share of its occurrences written in lower case, upper case and title case;
    - **part of speech,** for Teanga input with a part-of-speech layer: the tag of its
      first and last word, as one-hot vectors, and the number of its words with each
      tag, as HAMLET encodes them. A candidate's tags are those of its most frequent
      tag sequence, and the tags are those seen in training.

    The termhood, unithood and frequency features depend on the size of the corpus, so
    they are converted to their percentile rank among the candidates of the corpus,
    which lets a model trained on one corpus score the candidates of another.

    The classifier is scikit-learn's random forest by default, as in HAMLET, which
    found it better than a decision tree, logistic regression or a multi-layer
    perceptron. Terms are a minority of the candidates, so its classes are weighted by
    their frequency in each tree's sample (`class_weight="balanced_subsample"`), and
    each leaf has at least 5 candidates (`min_samples_leaf=5`). Of the settings tried,
    these classified best (F1 0.39, against 0.28 for scikit-learn's defaults) when
    training on two of ACTER's English training domains and testing on the third. Each candidate is scored by the probability the classifier gives it of
    being a term, over the whole corpus, so a term has the same score in every document
    it occurs in.

    Reference:
        Rigouts Terryn, A., Hoste, V. and Lefever, E. (2021). HAMLET: Hybrid Adaptable
        Machine Learning approach to Extract Terminology. Terminology 27(2), 254–293.
        https://doi.org/10.1075/term.20017.rig
    """

    def __init__(
        self,
        classifier: Any = "random_forest",
        threshold: float | None = 0.5,
        reference_corpus: ReferenceCorpus | list[ReferenceCorpus] | None = None,
        n: int = 4,
        random_state: int | None = None,
        n_jobs: int | None = None,
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        lemma_layer: str | None = None,
        min_frequency: int = 1,
        word_pattern: str | None = "alphanumeric",
        pos_pattern: str | None = None,
        pos_layer: str | None = None,
        terms_layer: str = GOLD_LAYER,
    ):
        """
        Initialize the extractor.

        Args:
            classifier (Union[str, Any]): "random_forest" (the default) for
                scikit-learn's `RandomForestClassifier`, or an unfitted scikit-learn
                classifier with `predict_proba`, which is cloned before training.
            threshold (Optional[float]): Keep the candidates that the classifier gives
                at least this probability of being a term: 0.5 (the default) keeps
                those it classifies as terms, and None ranks every candidate.
            reference_corpus (Optional[Union[Corpus, List[str], List[...]]]): A
                general-language reference corpus, of the same type as the corpora
                passed to `fit` and `predict`, to add the scores of the methods of
                `REFERENCE_METHODS` as features, or None.
            n (int): Maximum n-gram size of the candidates.
            random_state (Optional[int]): Seed of the random forest, for reproducible
                scores.
            n_jobs (Optional[int]): Number of trees of the random forest trained in
                parallel, as in joblib: -1 for one per CPU, or None for one at a time.
            stopwords (Union[str, Iterable[str], None]): Drop candidates that start or end
                with a stopword: a name in `STOPWORD_LISTS`, a list of words, or None.
            words_layer (Optional[str]): For Teanga input, the span layer holding the
                words, or None for `words` or, failing that, `tokens`.
            lemma_layer (Optional[str]): For Teanga input, a sequence layer holding the
                lemma of each word, to count inflected forms such as "cell" and "cells"
                as one candidate, given by its lemmas, or None to use the words as
                written. Labels and gold terms are matched to the lemmas.
            min_frequency (int): Drop candidates that occur fewer times in the corpus.
            word_pattern (Optional[str]): Drop candidates with a word that does not fully
                match this regular expression or named pattern ("alphanumeric", "atr4s"),
                or None to keep every word.
            pos_pattern (Optional[str]): For Teanga input, count only the occurrences
                whose part-of-speech tags match this pattern, as for `TermExtractor`;
                "learned" for the tag sequences of the occurrences of the terms in the
                training corpus, as HAMLET does; or None (the default) to allow every
                tag sequence.
            pos_layer (Optional[str]): The part-of-speech layer, or None for `upos` or,
                failing that, `pos`.
            terms_layer (str): For Teanga input without labels, the span layer of gold
                terms to train on.
        """
        self.classifier = classifier
        self.threshold = threshold
        self.reference_corpus = reference_corpus
        self.n = n
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.words_layer = words_layer
        self.lemma_layer = lemma_layer
        self.pos_layer = pos_layer
        self.terms_layer = terms_layer
        self.learns_pos_pattern = pos_pattern == LEARNED
        methods = TERMHOOD_METHODS + (
            () if reference_corpus is None else REFERENCE_METHODS
        )
        self.extractors: dict[str, BaseTermExtractor] = {
            method: TermExtractor(
                method=method,
                n=n,
                stopwords=stopwords,
                words_layer=words_layer,
                lemma_layer=lemma_layer,
                min_frequency=min_frequency,
                word_pattern=word_pattern,
                pos_pattern=None if self.learns_pos_pattern else pos_pattern,
                pos_layer=pos_layer,
                reference_corpus=reference_corpus,
            ).extractor
            for method in methods
        }
        """The unsupervised extractors whose scores are features, which also find the
        candidates."""
        self.model: Any = None
        self.feature_names: list[str] = []
        self.tagset: list[str] = []

    @property
    def _candidate_extractor(self) -> BaseTermExtractor:
        return self.extractors[TERMHOOD_METHODS[0]]

    def _has_pos(self, corpus: Corpus | list[str]) -> bool:
        if not isinstance(corpus, Corpus):
            return False
        try:
            find_layer(corpus.meta, self.pos_layer, POS_LAYERS, "pos_layer")
        except ValueError:
            return False
        return True

    def _words(self, corpus: Corpus | list[str]) -> _Candidates:
        """Read the words of a corpus, without finding the candidates yet."""
        if not isinstance(corpus, Corpus):
            return _Candidates(
                [split_words(doc) for doc in corpus], [], None, None, None
            )
        docs = [corpus.doc_by_id(doc_id) for doc_id in corpus.doc_ids]
        words = [
            words_with_offsets(doc, self.words_layer, self.lemma_layer) for doc in docs
        ]
        surface = [
            [text for _, _, text in words_with_offsets(doc, self.words_layer)]
            for doc in docs
        ]
        tags = (
            [pos_tags(doc, self.pos_layer, self.words_layer) for doc in docs]
            if self._has_pos(corpus)
            else None
        )
        return _Candidates(
            [[text for _, _, text in doc] for doc in words],
            [],
            tags,
            surface,
            [[(start, end) for start, end, _ in doc] for doc in words],
        )

    def _candidates(self, corpus: Corpus | list[str]) -> _Candidates:
        """Find the occurrences of the candidates of a corpus."""
        candidates = self._words(corpus)
        candidates.occurrences = self._candidate_extractor.candidate_occurrences(
            candidates.docs, candidates.tags
        )
        return candidates

    def learn_pos_pattern(
        self, corpus: Corpus | list[str], terms: set[str]
    ) -> re.Pattern[str]:
        """
        Learn the tag sequences of the terms, and count only the occurrences of
        candidates with one of them from then on.

        Args:
            corpus (Union[Corpus, List[str]]): The training corpus.
            terms (Set[str]): The terms, normalised by `normalize_term`.

        Returns:
            re.Pattern: A pattern matching the tag sequences of the occurrences of the
            terms of up to `n` words in the corpus.

        Raises:
            ValueError: If the corpus has no part-of-speech layer.
        """
        words = self._words(corpus)
        if words.tags is None:
            raise ValueError(
                'pos_pattern="learned" needs a Teanga corpus with a part-of-speech layer.'
            )
        sequences = {
            "".join(f"<{tag}>" for tag in tags[i:j])
            for doc, tags in zip(words.docs, words.tags)
            for i in range(len(doc))
            for j in range(i + 1, min(i + self.n, len(doc)) + 1)
            if normalize_term(" ".join(doc[i:j])) in terms
        }
        pattern = re.compile("|".join(map(re.escape, sorted(sequences))))
        for extractor in self.extractors.values():
            extractor.pos_pattern = pattern
        return pattern

    def _termhood(self, corpus: Corpus | list[str]) -> dict[str, dict[str, float]]:
        """The score of each candidate by each unsupervised method."""
        scores = {}
        for method, extractor in self.extractors.items():
            results = (
                extractor.extract_terms_teanga(corpus)
                if isinstance(corpus, Corpus)
                else extractor.extract_terms_strings(corpus)
            )
            best: dict[str, float] = {}
            for doc in results.terms:
                for term in doc["terms"]:
                    best[term["term"]] = max(
                        best.get(term["term"], -np.inf), term["score"]
                    )
            scores[method] = best
        return scores

    def features(
        self, corpus: Corpus | list[str], candidates: _Candidates, training: bool
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Compute the features of the candidates of a corpus.

        Args:
            corpus (Union[Corpus, List[str]]): The corpus.
            candidates: The occurrences of its candidates, from `_candidates`.
            training (bool): Whether the corpus is the training corpus, whose tags are
                the part-of-speech features.

        Returns:
            The feature matrix, with one row per candidate and one column per feature,
            the candidates, in order of first occurrence, and the names of the
            features.
        """
        terms = list(
            dict.fromkeys(
                ngram for doc in candidates.occurrences for ngram, _, _ in doc
            )
        )
        frequency = Counter(
            ngram for doc in candidates.occurrences for ngram, _, _ in doc
        )
        document_frequency = Counter(
            ngram for doc in candidates.occurrences for ngram in {o[0] for o in doc}
        )
        num_docs = max(len(candidates.occurrences), 1)

        # Features that depend on the size of the corpus, ranked within it
        termhood = self._termhood(corpus)
        counts = NgramCounts(candidates.docs, self.n)
        ranked_names = [f"termhood:{method}" for method in termhood]
        ranked_names += [f"unithood:{measure}" for measure in MEASURES]
        ranked_names += ["frequency:occurrences", "frequency:documents"]
        ranked = np.array(
            [
                [scores.get(term, -np.inf) for scores in termhood.values()]
                + self._unithood(counts, term)
                + [frequency[term], document_frequency[term] / num_docs]
                for term in terms
            ]
        ).reshape(len(terms), len(ranked_names))
        if len(terms):
            ranked = rankdata(ranked, axis=0) / len(terms)

        shape_names = [
            "shape:words",
            "shape:characters",
            "shape:digits",
            "shape:hyphens",
            "shape:other characters",
        ]
        shape = np.array(
            [
                [
                    len(term.split(" ")),
                    len(term),
                    sum(c.isdigit() for c in term),
                    term.count("-"),
                    sum(not (c.isalnum() or c in "- ") for c in term),
                ]
                for term in terms
            ]
        ).reshape(len(terms), len(shape_names))
        columns = [ranked, shape]
        names = ranked_names + shape_names

        if candidates.surface is not None:
            case_names = ["shape:lower case", "shape:upper case", "shape:title case"]
            columns.append(self._case(terms, candidates))
            names += case_names
        if candidates.tags is not None:
            columns.append(self._pos(terms, candidates, training))
            names += [
                f"pos:{position}={tag}"
                for position in ("first", "last", "count")
                for tag in self.tagset
            ]
        return np.hstack(columns), terms, names

    @staticmethod
    def _unithood(counts: NgramCounts, term: str) -> list[float]:
        """The measures of the weakest split of a candidate, or 0 for single words."""
        words = tuple(term.split(" "))
        if len(words) < 2:
            return [0.0] * len(MEASURES)
        splits = np.array([counts.measures(words, i) for i in range(1, len(words))])
        return list(splits.min(axis=0))

    @staticmethod
    def _case(terms: list[str], candidates: _Candidates) -> np.ndarray:
        """The share of the occurrences of each candidate in each case."""
        assert candidates.surface is not None
        cases: dict[str, np.ndarray] = {term: np.zeros(4) for term in terms}
        for words, occurrences in zip(candidates.surface, candidates.occurrences):
            for ngram, i, j in occurrences:
                text = " ".join(words[i:j])
                cases[ngram] += [text.islower(), text.isupper(), text.istitle(), 1]
        return np.array([cases[term][:3] / cases[term][3] for term in terms]).reshape(
            len(terms), 3
        )

    def _pos(
        self, terms: list[str], candidates: _Candidates, training: bool
    ) -> np.ndarray:
        """The first tag, last tag and tag counts of each candidate."""
        assert candidates.tags is not None
        sequences: dict[str, Counter[tuple[str, ...]]] = {t: Counter() for t in terms}
        for tags, occurrences in zip(candidates.tags, candidates.occurrences):
            for ngram, i, j in occurrences:
                sequences[ngram][tuple(tags[i:j])] += 1
        if training:
            self.tagset = sorted(
                {tag for counts in sequences.values() for seq in counts for tag in seq}
            )
        index = {tag: k for k, tag in enumerate(self.tagset)}
        size = len(self.tagset)
        features = np.zeros((len(terms), 3 * size))
        for row, term in enumerate(terms):
            sequence = sequences[term].most_common(1)[0][0]
            for position, tag in ((0, sequence[0]), (size, sequence[-1])):
                if tag in index:
                    features[row, position + index[tag]] = 1
            for tag in sequence:
                if tag in index:
                    features[row, 2 * size + index[tag]] += 1
        return features

    def train(self, corpus: Corpus | list[str], labels: dict[str, int]) -> None:
        """
        Train the classifier on the labelled candidates of a corpus.

        Labels are matched to the candidates after lower-casing. If none is 0, as with
        a list of gold terms, every candidate is an example, and those that are not
        labelled 1 are the non-terms. Otherwise only the labelled candidates are.

        Args:
            corpus (Union[Corpus, List[str]]): The training corpus.
            labels (Dict[str, int]): The label, 1 for a term or 0 for a non-term, of
                candidates.

        Raises:
            ValueError: If no candidate is labelled 1, or every one is.
        """
        normalized = {normalize_term(term): label for term, label in labels.items()}
        if self.learns_pos_pattern:
            self.learn_pos_pattern(
                corpus, {term for term, label in normalized.items() if label == 1}
            )
        X, terms, self.feature_names = self.features(
            corpus, self._candidates(corpus), training=True
        )
        keys = [normalize_term(term) for term in terms]
        if all(label == 1 for label in normalized.values()):
            rows = list(range(len(terms)))
        else:
            rows = [i for i, key in enumerate(keys) if key in normalized]
        y = np.array([normalized.get(keys[i]) == 1 for i in rows], dtype=int)
        if y.all() or not y.any():
            raise ValueError(
                "The features method needs at least one candidate in the corpus "
                "labelled 1 and one labelled 0."
            )
        model = (
            RandomForestClassifier(
                class_weight="balanced_subsample",
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            if self.classifier == "random_forest"
            else clone(self.classifier)
        )
        self.model = model.fit(X[rows], y)

    def fit_strings(self, corpus: list[str], labels: dict[str, int]) -> None:
        """Train the classifier on a list of strings."""
        self.train(corpus, labels)

    def fit_teanga(self, corpus: Corpus, labels: dict[str, int] | None = None) -> None:
        """Train the classifier on a Teanga corpus, with the gold terms of
        `terms_layer` as the terms if `labels` is None."""
        if labels is None:
            gold = gold_terms(corpus, self.terms_layer, lemma_layer=self.lemma_layer)
            assert isinstance(gold, set)
            labels = dict.fromkeys(gold, 1)
        self.train(corpus, labels)

    def predict_scores(
        self, corpus: Corpus | list[str], candidates: _Candidates
    ) -> dict[str, float]:
        """
        Score the candidates of a corpus by the probability the classifier gives them
        of being terms.

        Raises:
            ValueError: If the classifier has not been trained, or the corpus does not
                have the features it was trained on: a Teanga corpus with a
                part-of-speech layer, if it was trained on one.
        """
        if self.model is None:
            raise ValueError("The features method has not been trained yet.")
        X, terms, names = self.features(corpus, candidates, training=False)
        if names != self.feature_names:
            raise ValueError(
                "The corpus does not have the features the classifier was trained on: "
                "it must be of the same type, with the same layers."
            )
        if not terms:
            return {}
        positive = list(self.model.classes_).index(1)
        probabilities = self.model.predict_proba(X)[:, positive]
        return {term: float(p) for term, p in zip(terms, probabilities)}

    def _results(
        self, corpus: Corpus | list[str], doc_ids: list[str], processed: Any
    ) -> ATEResults:
        candidates = self._candidates(corpus)
        scores = self.predict_scores(corpus, candidates)
        terms_by_doc = []
        offsets_by_doc = []
        for k, (doc_id, occurrences) in enumerate(zip(doc_ids, candidates.occurrences)):
            doc_terms = {
                "doc_id": doc_id,
                "terms": rank_terms(
                    {ngram: scores[ngram] for ngram, _, _ in occurrences},
                    self.threshold,
                ),
            }
            terms_by_doc.append(doc_terms)
            if candidates.spans is not None:
                spans = candidates.spans[k]
                offsets_by_doc.append(
                    BaseTermExtractor.doc_offsets(
                        doc_id,
                        [
                            (ngram, (spans[i][0], spans[j - 1][1]))
                            for ngram, i, j in occurrences
                        ],
                        doc_terms,
                    )
                )
        return ATEResults(
            corpus=processed,
            terms=terms_by_doc,
            offsets=offsets_by_doc if candidates.spans is not None else None,
        )

    def predict_strings(self, corpus: list[str]) -> ATEResults:
        """Score the candidates of a list of strings."""
        doc_ids = [f"doc_{i}" for i in range(len(corpus))]
        processed = [
            {"doc_id": doc_id, "text": doc} for doc_id, doc in zip(doc_ids, corpus)
        ]
        return self._results(corpus, doc_ids, processed)

    def predict_teanga(self, corpus: Corpus) -> ATEResults:
        """Score the candidates of a Teanga corpus, with their offsets."""
        return self._results(corpus, list(corpus.doc_ids), corpus)

    def feature_importances(self) -> dict[str, float]:
        """
        Get the importance of each feature to the trained classifier.

        Returns:
            Dict[str, float]: The `feature_importances_` of the classifier, by feature
            name, from the most to the least important.

        Raises:
            ValueError: If the classifier has not been trained, or has no
                `feature_importances_`, as with logistic regression.
        """
        if self.model is None:
            raise ValueError("The features method has not been trained yet.")
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError(f"{type(self.model).__name__} has no feature importances.")
        importances = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importances.items(), key=lambda item: -item[1]))
