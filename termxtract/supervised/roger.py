import math
from bisect import bisect_left
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata

from ..association import MEASURES, NgramCounts
from ..datasets.base import GOLD_LAYER
from ..layers import find_layer, words_with_offsets
from ..stopwords import Stopwords, load_stopwords
from ..utils import ATEResults, rank_terms
from .words import tokenize_strings, tokenize_teanga


class RogerTermExtractor:
    """
    Supervised term ranking with n'Roger (Azé et al., 2005).

    Each multi-word candidate `w1 ... wk` is described by the association measures in
    `MEASURES` between its first words `w1 ... wk-1` and its last word `wk`, computed
    over all the k-word sequences in the corpus. Single words have no parts to
    associate, so they are not ranked.

    A (μ+λ) evolution strategy learns a ranking function that maximises the area under
    the ROC curve (the Wilcoxon rank statistic) of the labelled candidates. Ranking
    functions are non-linear: a candidate with (normalised) measures `x` is scored
    `sum(w[i] * abs(x[i] - c[i]))`, with weights `w` and centre `c`. The final score of
    a candidate is the median of its scores under the ranking functions learned by
    `runs` independent runs.

    Reference:
        Azé, J., Roche, M., Kodratoff, Y. and Sebag, M. (2005). Preference learning in
        terminology extraction: a ROC-based approach. In Proceedings of Applied
        Stochastic Models and Data Analysis, pp. 209–219. https://arxiv.org/abs/cs/0512050
    """

    crossover_rate = 0.6
    """Probability that an offspring is created by uniform crossover of two parents."""

    def __init__(
        self,
        population_size: int = 20,
        offspring_size: int = 100,
        generations: int = 100,
        runs: int = 21,
        n: int = 2,
        random_state: int | None = None,
        max_negatives: int | None = 10_000,
        patience: int | None = None,
        n_jobs: int | None = None,
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        lemma_layer: str | None = None,
        terms_layer: str = GOLD_LAYER,
    ):
        """
        Initialize the Roger extractor.

        The defaults for `population_size`, `offspring_size` and `runs` are those of
        Azé et al. (2005).

        Args:
            population_size (int): Number of parents (μ) kept in each generation.
            offspring_size (int): Number of offspring (λ) created in each generation.
            generations (int): Number of generations in each run.
            runs (int): Number of independent runs whose ranking functions are combined.
            n (int): Maximum n-gram size, at least 2.
            random_state (Optional[int]): Seed for the evolutionary search, for
                reproducible scores.
            max_negatives (Optional[int]): Train on at most this many of the candidates
                labelled 0, chosen at random for each run, and on all those labelled 1.
                The area under the ROC curve does not depend on the share of each
                label, so this speeds up training on large corpora with little effect
                on the ranking functions: on the English benchmark, 10,000 negatives
                train about 30 times faster than all of them, with similar results.
                None trains on every labelled candidate.
            patience (Optional[int]): Stop a run early when its best ranking function
                has not improved for this many generations, or None to run every
                generation.
            n_jobs (Optional[int]): Number of runs trained in parallel, as in joblib: -1
                for one per CPU, or None for one at a time.
            stopwords (Union[str, Iterable[str], None]): Drop extracted terms that start or
                end with a stopword: a name in `STOPWORD_LISTS`, a list of words, or None.
            words_layer (Optional[str]): For Teanga input, the span layer holding the
                words, or None for `words` or, failing that, `tokens`.
            lemma_layer (Optional[str]): For Teanga input, a sequence layer, based on
                the words layer, holding the lemma of each word. If given, every word
                is read as its lemma, so the measures count inflected forms together,
                candidates are written as lemmas, and labels and gold terms are matched
                to them by their lemmas. None reads the words as written.
            terms_layer (str): For Teanga input without labels, the span layer of gold
                terms to train on.

        Raises:
            ValueError: If `n` is less than 2.
        """
        if n < 2:
            raise ValueError("Roger ranks multi-word terms, so n must be at least 2.")
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.generations = generations
        self.runs = runs
        self.n = n
        self.random_state = random_state
        self.max_negatives = max_negatives
        self.patience = patience
        self.n_jobs = n_jobs
        self.stopwords = load_stopwords(stopwords)
        self.words_layer = words_layer
        self.lemma_layer = lemma_layer
        self.terms_layer = terms_layer
        self.models: list[tuple[np.ndarray, np.ndarray]] = []
        self.feature_min = np.zeros(len(MEASURES))
        self.feature_range = np.ones(len(MEASURES))

    def extract_features(self, docs: list[list[str]]) -> tuple[np.ndarray, list[str]]:
        """
        Compute the association measures of every candidate of 2 to `n` words.

        Args:
            docs (list[list[str]]): The words of each document.

        Returns:
            The feature matrix (one row per candidate, one column per measure) and the
            candidates, in order of first occurrence.
        """
        counts = NgramCounts(docs, self.n)
        candidates = list(counts.ngrams)
        features = np.array(
            [counts.measures(ngram, len(ngram) - 1) for ngram in candidates]
        ).reshape(len(candidates), len(MEASURES))
        return features, [" ".join(ngram) for ngram in candidates]

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_min) / self.feature_range

    @staticmethod
    def rank_scores(w: np.ndarray, c: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Score the candidates `X` with each ranking function `(w[j], c[j])`.

        Returns:
            np.ndarray: One row of candidate scores per ranking function.
        """
        dtype = np.result_type(w, c, X)
        scores = np.zeros((len(w), len(X)), dtype=dtype)
        term = np.empty_like(scores)
        for i in range(X.shape[1]):  # One measure at a time, to save memory
            np.subtract(X[:, i], c[:, i, np.newaxis], out=term)
            np.abs(term, out=term)
            term *= w[:, i, np.newaxis]
            scores += term
        return scores

    @staticmethod
    def auc(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Area under the ROC curve of each row of `scores`.

        This is the Wilcoxon rank statistic: the probability that a positive example is
        scored higher than a negative one, counting ties as one half.
        """
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        ranks = rankdata(scores, axis=1)
        return (ranks[:, y == 1].sum(axis=1) - n_pos * (n_pos + 1) / 2) / (
            n_pos * n_neg
        )

    def optimize_rank_function(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the (μ+λ) evolution strategy once.

        Each individual is a ranking function `(w, c)` with a self-adaptive mutation
        step size for each of its genes. Offspring are created by uniform crossover of
        two random parents (with probability `crossover_rate`) or by copying one, and are
        then mutated. The best `population_size` of the parents and offspring become the
        next parents.

        Returns:
            The weights and centre of the best ranking function, with `sum(abs(w)) == 1`.
        """
        mu, lam = self.population_size, self.offspring_size
        d = X.shape[1]
        genes = np.hstack([rng.uniform(-1, 1, (mu, d)), rng.uniform(0, 1, (mu, d))])
        steps = np.full_like(genes, 0.1)
        # Learning rates of the self-adaptive log-normal mutation (Bäck, 1995)
        tau_global = 1 / math.sqrt(2 * genes.shape[1])
        tau_local = 1 / math.sqrt(2 * math.sqrt(genes.shape[1]))

        def fitness(genes: np.ndarray) -> np.ndarray:
            w, c = genes[:, :d].astype(X.dtype), genes[:, d:].astype(X.dtype)
            return self.auc(self.rank_scores(w, c, X), y)

        scores = fitness(genes)
        stalled = 0
        for _ in range(self.generations):
            first = rng.integers(mu, size=lam)
            second = rng.integers(mu, size=lam)
            crossover = (rng.random(lam) < self.crossover_rate)[:, np.newaxis]
            from_second = crossover & (rng.random(genes[first].shape) < 0.5)
            child_genes = np.where(from_second, genes[second], genes[first])
            child_steps = np.where(from_second, steps[second], steps[first])
            child_steps *= np.exp(
                tau_global * rng.standard_normal((lam, 1))
                + tau_local * rng.standard_normal(child_steps.shape)
            )
            child_genes = child_genes + child_steps * rng.standard_normal(
                child_genes.shape
            )

            genes = np.vstack([genes, child_genes])
            steps = np.vstack([steps, child_steps])
            scores = np.concatenate([scores, fitness(child_genes)])
            # Stable sort, so parents win ties against their offspring
            best = np.argsort(-scores, kind="stable")[:mu]
            stalled = 0 if best[0] >= mu else stalled + 1
            genes, steps, scores = genes[best], steps[best], scores[best]
            if self.patience is not None and stalled >= self.patience:
                break

        w, c = genes[0, :d], genes[0, d:]
        return w / max(np.abs(w).sum(), 1e-12), c

    def train(self, docs: list[list[str]], labels: dict[str, int]):
        """
        Learn `runs` ranking functions from the labelled candidates in `docs`.

        As in Azé et al. (2005), only the labelled candidates are training examples.
        If none of them is labelled 0, as with a list of gold terms, every unlabelled
        candidate is a negative example instead. The measures are normalised to the
        range of all the candidates in `docs`.

        Raises:
            ValueError: If no candidate in `docs` is labelled 1, or every one is.
        """
        X, terms = self.extract_features(docs)
        if terms:
            self.feature_min = X.min(axis=0)
            self.feature_range = np.where(
                X.max(axis=0) > self.feature_min, X.max(axis=0) - self.feature_min, 1.0
            )
        labelled = [i for i, term in enumerate(terms) if term in labels]
        if all(labels[terms[i]] == 1 for i in labelled):
            # Only terms are labelled, so the other candidates are the non-terms
            labelled = list(range(len(terms)))
        y = np.array([labels.get(terms[i]) == 1 for i in labelled], dtype=int)
        if y.all() or not y.any():
            raise ValueError(
                "Roger needs at least one candidate in the corpus labelled 1 "
                "and one labelled 0."
            )
        # Single precision halves the work of the fitness, and is enough for ranking
        X = self._normalize(X[labelled]).astype(np.float32)
        seeds = np.random.SeedSequence(self.random_state).spawn(self.runs)
        # The runs are independent, and numpy releases the GIL, so threads suffice
        self.models = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._run)(X, y, np.random.default_rng(seed)) for seed in seeds
        )

    def _run(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """One run of `optimize_rank_function`, on a sample of the negatives."""
        negatives = np.flatnonzero(y == 0)
        if self.max_negatives is not None and len(negatives) > self.max_negatives:
            keep = np.concatenate(
                [
                    np.flatnonzero(y == 1),
                    rng.choice(negatives, self.max_negatives, replace=False),
                ]
            )
            X, y = X[keep], y[keep]
        return self.optimize_rank_function(X, y, rng)

    def predict(self, docs: list[list[str]]) -> dict[str, float]:
        """Score the candidates in `docs` with the median of the learned functions."""
        if not self.models:
            raise ValueError("Roger has not been trained yet.")
        X, terms = self.extract_features(docs)
        if not terms:
            return {}
        X = self._normalize(X)
        w = np.array([w for w, _ in self.models])
        c = np.array([c for _, c in self.models])
        scores = np.median(self.rank_scores(w, c, X), axis=0)
        return {term: float(score) for term, score in zip(terms, scores)}

    def train_model(self, corpus: list[str], labels: dict[str, int]):
        """Learn the ranking functions from a list of strings."""
        self.train(tokenize_strings(corpus), labels)

    def predict_terms_strings(self, corpus: list[str]) -> dict[str, float]:
        """Score the candidates in a list of strings."""
        return self.predict(tokenize_strings(corpus))

    def predict_terms_teanga(self, corpus) -> dict[str, float]:
        """Score the candidates in a Teanga corpus."""
        return self.predict(tokenize_teanga(corpus, self.words_layer, self.lemma_layer))

    def fit_strings(self, corpus: list[str], labels: dict[str, int]) -> None:
        """Learn the ranking functions from a list of strings."""
        self.train(tokenize_strings(corpus), labels)

    def fit_teanga(self, corpus, labels: dict[str, int] | None = None) -> None:
        """Learn the ranking functions from a Teanga corpus, with the gold terms of
        `terms_layer` as the terms if `labels` is None."""
        if labels is None:
            labels = dict.fromkeys(self.gold_terms(corpus), 1)
        self.train(tokenize_teanga(corpus, self.words_layer, self.lemma_layer), labels)

    def gold_terms(self, corpus) -> set[str]:
        """The gold terms of `terms_layer`, each as the lower-cased words, or their
        lemmas if `lemma_layer` is set, within its span, as candidates are written."""
        layer = find_layer(corpus.meta, self.terms_layer, (), "terms_layer")
        terms = set()
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            words = words_with_offsets(doc, self.words_layer, self.lemma_layer)
            starts = [start for start, _, _ in words]
            for span in doc[layer]:
                last = bisect_left(starts, span[0])
                term = []
                while last < len(words) and words[last][1] <= span[1]:
                    term.append(words[last][2].lower())
                    last += 1
                if term:
                    terms.add(" ".join(term))
        return terms

    def predict_strings(self, corpus: list[str]) -> ATEResults:
        """Rank the candidates of a list of strings with the learned functions."""
        return self._results(
            corpus, [f"doc_{i}" for i in range(len(corpus))], tokenize_strings(corpus)
        )

    def predict_teanga(self, corpus) -> ATEResults:
        """Rank the candidates of a Teanga corpus, with their offsets."""
        words = [
            words_with_offsets(
                corpus.doc_by_id(doc_id), self.words_layer, self.lemma_layer
            )
            for doc_id in corpus.doc_ids
        ]
        return self._results(
            corpus,
            list(corpus.doc_ids),
            [[word.lower() for _, _, word in doc] for doc in words],
            [[(start, end) for start, end, _ in doc] for doc in words],
        )

    def _results(
        self,
        corpus,
        doc_ids: list[str],
        docs: list[list[str]],
        spans: list[list[tuple[int, int]]] | None = None,
    ) -> ATEResults:
        """
        Score the candidates of `docs`, listing each document's own candidates with
        their corpus-level scores, and their offsets if the word `spans` are given.
        """
        term_scores = self.predict(docs)
        terms_by_doc = []
        offsets_by_doc = []
        for k, (doc_id, words) in enumerate(zip(doc_ids, docs)):
            occurrences = [
                (" ".join(words[i : i + length]), i, i + length)
                for length in range(2, self.n + 1)
                for i in range(len(words) - length + 1)
            ]
            doc_terms: dict[str, Any] = {
                "doc_id": doc_id,
                "terms": rank_terms(
                    {
                        term: term_scores[term]
                        for term in dict.fromkeys(term for term, _, _ in occurrences)
                    },
                    stopwords=self.stopwords,
                ),
            }
            terms_by_doc.append(doc_terms)
            if spans is not None:
                kept = {term["term"] for term in doc_terms["terms"]}
                offsets_by_doc.append(
                    {
                        "doc_id": doc_id,
                        "offsets": [
                            {
                                "term": term,
                                "offset": (spans[k][start][0], spans[k][end - 1][1]),
                            }
                            for term, start, end in sorted(
                                occurrences, key=lambda o: (o[1], o[2])
                            )
                            if term in kept
                        ],
                    }
                )
        return ATEResults(
            corpus=corpus,
            terms=terms_by_doc,
            offsets=offsets_by_doc if spans is not None else None,
        )

    def extract_terms_teanga(self, corpus, labels: dict[str, int]) -> ATEResults:
        """Learn from a Teanga corpus with Roger and rank its candidates."""
        self.fit_teanga(corpus, labels)
        return self.predict_teanga(corpus)

    def extract_terms_strings(
        self, corpus: list[str], labels: dict[str, int]
    ) -> ATEResults:
        """Learn from a list of strings with Roger and rank its candidates."""
        self.fit_strings(corpus, labels)
        return self.predict_strings(corpus)
