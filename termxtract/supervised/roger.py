import math
from collections import Counter
from collections.abc import Iterator

import numpy as np
from scipy.stats import rankdata

from ..stopwords import Stopwords, load_stopwords
from ..utils import ATEResults, rank_terms
from .words import tokenize_strings, tokenize_teanga

MEASURES = (
    "MI",
    "MI3",
    "Dice",
    "L",
    "OccL",
    "SeSc",
    "J",
    "Conv",
    "LC",
    "CM",
    "Khi2",
    "T test",
)
"""Names of the association measures used as features, in order."""


def _xlogx(x: float) -> float:
    return x * math.log(x) if x > 0 else 0.0


def association_measures(a: int, n_x: int, n_y: int, total: int) -> list[float]:
    """
    Compute the association measures of a collocation `x y` from its contingency table.

    The table counts the `total` positions of collocations of the same length: `a` of
    them are `x y`, `n_x` start with `x` and `n_y` end with `y`. The measures are those
    of Azé et al. (2005), section 2, without Jacquemin's association measure, whose
    definition is not given there. Rule-based measures (SeSc, J, Conv, LC, CM) treat the
    collocation as the rule `x → y`.

    The rule-based measures divide by the number of counter-examples `b`, the positions
    starting with `x` but not ending with `y`. When `x` is always followed by `y`, `b`
    is 0 and these measures are infinite, so `b` is then taken to be 0.5.

    Args:
        a (int): Occurrences of the collocation `x y`.
        n_x (int): Positions starting with `x`.
        n_y (int): Positions ending with `y`.
        total (int): Positions of collocations of this length.

    Returns:
        list[float]: The measures named in `MEASURES`, in order.
    """
    b = n_x - a  # x, not y
    c = n_y - a  # y, not x
    d = total - a - b - c  # neither
    n_not_x = total - n_x
    n_not_y = total - n_y
    b_smoothed = b if b > 0 else 0.5

    # Mutual information (Church and Hanks, 1990) and its cube (Daille et al., 1998)
    mi = math.log2(a * total / (n_x * n_y))
    mi3 = math.log2(a**3 / (n_x * n_y))
    # Dice coefficient (Smadja et al., 1996)
    dice = 2 * a / (n_x + n_y)
    # Log-likelihood ratio (Dunning, 1993)
    log_likelihood = 2 * (
        _xlogx(a)
        + _xlogx(b)
        + _xlogx(c)
        + _xlogx(d)
        - _xlogx(n_x)
        - _xlogx(n_not_x)
        - _xlogx(n_y)
        - _xlogx(n_not_y)
        + _xlogx(total)
    )
    log_likelihood = max(log_likelihood, 0.0)  # Rounding error for independent words
    # Occurrences, with ties broken by log-likelihood (Roche et al., 2004)
    occ_l = a + log_likelihood / (log_likelihood + 1)
    # Sebag-Schoenauer: examples over counter-examples of x → y
    sebag_schoenauer = a / b_smoothed
    # J-measure (Goodman and Smyth, 1988)
    p_y_given_x = a / n_x
    p_y = n_y / total
    j_measure = (n_x / total) * (
        (p_y_given_x * math.log(p_y_given_x / p_y))
        + (
            (1 - p_y_given_x) * math.log((1 - p_y_given_x) / (1 - p_y))
            if p_y_given_x < 1
            else 0.0
        )
    )
    # Conviction (Brin et al., 1997): P(x) P(not y) / P(x, not y)
    conviction = n_x * n_not_y / (total * b_smoothed)
    # Least contradiction (Azé and Kodratoff, 2004): (examples - counter-examples) / n_y
    least_contradiction = (a - b) / n_y
    # Odds multiplier ("cote multiplier", Lallich and Teytaud, 2004)
    odds_multiplier = a * n_not_y / (b_smoothed * n_y)
    # Chi-squared and t-test (Manning and Schütze, 1999, chapter 5)
    denominator = n_x * n_y * n_not_x * n_not_y
    khi2 = total * (a * d - b * c) ** 2 / denominator if denominator else 0.0
    t_test = (a - n_x * n_y / total) / math.sqrt(a)

    return [
        mi,
        mi3,
        dice,
        log_likelihood,
        occ_l,
        sebag_schoenauer,
        j_measure,
        conviction,
        least_contradiction,
        odds_multiplier,
        khi2,
        t_test,
    ]


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
        stopwords: Stopwords = "english",
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
            stopwords (Union[str, Iterable[str], None]): Drop extracted terms that start or
                end with a stopword: "english", a list of words, or None.

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
        self.stopwords = load_stopwords(stopwords)
        self.models: list[tuple[np.ndarray, np.ndarray]] = []
        self.feature_min = np.zeros(len(MEASURES))
        self.feature_range = np.ones(len(MEASURES))

    def _ngrams(self, docs: list[list[str]]) -> Iterator[tuple[str, ...]]:
        for words in docs:
            for k in range(2, self.n + 1):
                for i in range(len(words) - k + 1):
                    yield tuple(words[i : i + k])

    def extract_features(self, docs: list[list[str]]) -> tuple[np.ndarray, list[str]]:
        """
        Compute the association measures of every candidate of 2 to `n` words.

        Args:
            docs (list[list[str]]): The words of each document.

        Returns:
            The feature matrix (one row per candidate, one column per measure) and the
            candidates, in order of first occurrence.
        """
        ngram_counts = Counter(self._ngrams(docs))
        prefix_counts: Counter[tuple[str, ...]] = Counter()
        last_word_counts: Counter[tuple[int, str]] = Counter()
        totals: Counter[int] = Counter()
        for ngram, count in ngram_counts.items():
            prefix_counts[ngram[:-1]] += count
            last_word_counts[len(ngram), ngram[-1]] += count
            totals[len(ngram)] += count

        candidates = list(ngram_counts)
        features = np.array(
            [
                association_measures(
                    ngram_counts[ngram],
                    prefix_counts[ngram[:-1]],
                    last_word_counts[len(ngram), ngram[-1]],
                    totals[len(ngram)],
                )
                for ngram in candidates
            ]
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
        scores = np.zeros((len(w), len(X)))
        for i in range(X.shape[1]):  # One measure at a time, to save memory
            scores += w[:, i, np.newaxis] * np.abs(X[:, i] - c[:, i, np.newaxis])
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
            return self.auc(self.rank_scores(genes[:, :d], genes[:, d:], X), y)

        scores = fitness(genes)
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
            genes, steps, scores = genes[best], steps[best], scores[best]

        w, c = genes[0, :d], genes[0, d:]
        return w / max(np.abs(w).sum(), 1e-12), c

    def train(self, docs: list[list[str]], labels: dict[str, int]):
        """
        Learn `runs` ranking functions from the labelled candidates in `docs`.

        As in Azé et al. (2005), only the labelled candidates are training examples.
        The measures are normalised to the range of all the candidates in `docs`.

        Raises:
            ValueError: If no candidate in `docs` is labelled 1, or none is labelled 0.
        """
        X, terms = self.extract_features(docs)
        if terms:
            self.feature_min = X.min(axis=0)
            self.feature_range = np.where(
                X.max(axis=0) > self.feature_min, X.max(axis=0) - self.feature_min, 1.0
            )
        labelled = [i for i, term in enumerate(terms) if term in labels]
        y = np.array([labels[terms[i]] == 1 for i in labelled], dtype=int)
        if y.all() or not y.any():
            raise ValueError(
                "Roger needs at least one candidate in the corpus labelled 1 "
                "and one labelled 0."
            )
        X = self._normalize(X[labelled])
        rng = np.random.default_rng(self.random_state)
        self.models = [self.optimize_rank_function(X, y, rng) for _ in range(self.runs)]

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
        return self.predict(tokenize_teanga(corpus))

    def _results(self, corpus, doc_ids: list[str], docs, labels) -> ATEResults:
        self.train(docs, labels)
        term_scores = self.predict(docs)
        return ATEResults(
            corpus=corpus,
            terms=[
                {
                    "doc_id": doc_id,
                    "terms": rank_terms(term_scores, stopwords=self.stopwords),
                }
                for doc_id in doc_ids
            ],
        )

    def extract_terms_teanga(self, corpus, labels: dict[str, int]) -> ATEResults:
        """Extract terms from a Teanga corpus using Roger."""
        return self._results(
            corpus, list(corpus.doc_ids), tokenize_teanga(corpus), labels
        )

    def extract_terms_strings(
        self, corpus: list[str], labels: dict[str, int]
    ) -> ATEResults:
        """Extract terms from a list of strings using Roger."""
        return self._results(
            corpus,
            [f"doc_{i}" for i in range(len(corpus))],
            tokenize_strings(corpus),
            labels,
        )
