import math

import numpy as np
import pytest
from scipy.stats import chi2_contingency

from termxtract.association import MEASURES, NgramCounts, association_measures


def test_association_measures():
    # x y occurs 10 times; 15 bigrams start with x, 30 end with y, out of 1000
    measures = dict(zip(MEASURES, association_measures(10, 15, 30, 1000)))
    table = np.array([[10, 5], [20, 965]])
    assert measures["Khi2"] == pytest.approx(
        chi2_contingency(table, correction=False)[0]
    )
    assert measures["L"] == pytest.approx(
        chi2_contingency(table, correction=False, lambda_="log-likelihood")[0]
    )
    assert measures["MI"] == pytest.approx(math.log2(10 * 1000 / (15 * 30)))
    assert measures["MI3"] == pytest.approx(math.log2(10**3 / (15 * 30)))
    assert measures["Dice"] == pytest.approx(2 * 10 / (15 + 30))
    assert measures["T test"] == pytest.approx((10 - 15 * 30 / 1000) / math.sqrt(10))
    assert measures["SeSc"] == 2
    assert measures["LC"] == pytest.approx((10 - 5) / 30)
    assert 10 < measures["OccL"] < 11


def test_measures_finite_without_counter_examples():
    # x is always followed by y, so there are no counter-examples
    assert np.all(np.isfinite(association_measures(3, 3, 5, 100)))


def test_tables_of_the_splits():
    counts = NgramCounts([["a", "b", "c"], ["a", "b", "d"]], 3)
    # Bigrams a b, b c, a b, b d
    assert counts.table(("a", "b"), 1) == (2, 2, 2, 4)
    # Trigrams a b c, a b d, split into a | b c and a b | c
    assert counts.table(("a", "b", "c"), 1) == (1, 2, 1, 2)
    assert counts.table(("a", "b", "c"), 2) == (1, 2, 1, 2)
    assert counts.measures(("a", "b"), 1) == association_measures(2, 2, 2, 4)


def test_sequences_are_counted_within_documents():
    counts = NgramCounts([["a", "b"], ["c", "d"]], 2)
    assert ("b", "c") not in counts.ngrams
    assert counts.totals[2] == 2
