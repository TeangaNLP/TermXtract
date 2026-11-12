"""
Association (unithood) measures, which score how strongly the words of an n-gram stick
together, from the contingency table of a split of the n-gram into two parts.
"""

import math
from collections import Counter

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


class NgramCounts:
    """
    The contingency tables of the binary splits of the n-grams of a corpus.

    Every sequence of 2 to `n` words in the documents is counted. A k-word n-gram can be
    split after its i-th word into `x`, its first i words, and `y`, the rest. The
    table of that split counts the k-word sequences of the corpus: how many there are,
    how many are the n-gram, how many start with `x` and how many end with `y`.

    Args:
        docs (list[list[str]]): The words of each document.
        n (int): The maximum length of the n-grams.
    """

    def __init__(self, docs: list[list[str]], n: int):
        self.ngrams: Counter[tuple[str, ...]] = Counter(
            tuple(words[i : i + k])
            for words in docs
            for k in range(2, n + 1)
            for i in range(len(words) - k + 1)
        )
        """The number of occurrences of each n-gram, in order of first occurrence."""
        self.totals: Counter[int] = Counter()
        for ngram, count in self.ngrams.items():
            self.totals[len(ngram)] += count
        self._parts: dict[tuple[int, int], tuple[Counter, Counter]] = {}

    def table(self, ngram: tuple[str, ...], i: int) -> tuple[int, int, int, int]:
        """
        Get the contingency table of a split of an n-gram.

        Args:
            ngram (tuple[str, ...]): The words of an n-gram of the corpus.
            i (int): The number of words before the split, from 1 to `len(ngram) - 1`.

        Returns:
            tuple[int, int, int, int]: The arguments `a`, `n_x`, `n_y` and `total` of
            `association_measures`.
        """
        k = len(ngram)
        if (k, i) not in self._parts:
            starts: Counter[tuple[str, ...]] = Counter()
            ends: Counter[tuple[str, ...]] = Counter()
            for other, count in self.ngrams.items():
                if len(other) == k:
                    starts[other[:i]] += count
                    ends[other[i:]] += count
            self._parts[k, i] = starts, ends
        starts, ends = self._parts[k, i]
        return self.ngrams[ngram], starts[ngram[:i]], ends[ngram[i:]], self.totals[k]

    def measures(self, ngram: tuple[str, ...], i: int) -> list[float]:
        """
        Compute the association measures of a split of an n-gram.

        Args:
            ngram (tuple[str, ...]): The words of an n-gram of the corpus.
            i (int): The number of words before the split, from 1 to `len(ngram) - 1`.

        Returns:
            list[float]: The measures named in `MEASURES`, in order.
        """
        return association_measures(*self.table(ngram, i))
