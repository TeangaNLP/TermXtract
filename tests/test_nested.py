import math

import pytest

from termxtract import TermExtractor
from termxtract.unsupervised.basic import BasicTermExtractor
from termxtract.unsupervised.cvalue import CValueTermExtractor

# Frequencies with n=2: heart 3, failure 2, heart failure 2, and 1 for the others
CORPUS = ["heart failure risk", "heart failure", "heart art"]


def scores(method, n=2, **params):
    results = TermExtractor(method=method, n=n, **params).extract(CORPUS)
    by_term = {}
    for doc in results.terms:
        for term in doc["terms"]:
            by_term.setdefault(term["term"], set()).add(term["score"])
    return by_term


def score(method, n=2, **params):
    """The score of each term, which must be the same in every document."""
    by_term = scores(method, n, **params)
    assert all(len(values) == 1 for values in by_term.values())
    return {term: next(iter(values)) for term, values in by_term.items()}


def test_nesting_is_by_whole_words():
    extractor = BasicTermExtractor(n=2)
    containing = extractor.nesting({"heart", "art", "heart failure", "heart art"})
    assert containing["heart"] == {"heart failure", "heart art"}
    assert containing["art"] == {"heart art"}
    assert containing["heart failure"] == set()


def test_nesting_counts_a_repeated_part_once():
    containing = BasicTermExtractor(n=3).nesting({"heart", "heart to heart"})
    assert containing["heart"] == {"heart to heart"}


@pytest.mark.parametrize("method", ["cvalue", "basic", "combobasic"])
def test_scores_are_the_same_in_every_document(method):
    assert set(score(method)) == {
        "heart",
        "failure",
        "risk",
        "art",
        "heart failure",
        "failure risk",
        "heart art",
    }


def test_cvalue():
    by_term = score("cvalue")
    # heart is nested in heart failure (2) and heart art (1)
    assert by_term["heart"] == pytest.approx(math.log2(1.1) * (3 - 3 / 2))
    assert by_term["heart failure"] == pytest.approx(math.log2(2.1) * 2)
    # Single words are ranked, unless only ever seen inside a longer candidate
    assert by_term["risk"] == pytest.approx(0)
    assert by_term["art"] == pytest.approx(0)


def test_basic():
    by_term = score("basic", n=3, alpha=0.5)
    # Single words gain nothing from being nested
    assert by_term["heart"] == pytest.approx(math.log(3))
    # heart failure is nested in heart failure risk
    assert by_term["heart failure"] == pytest.approx(2 * math.log(2) + 0.5)
    assert by_term["heart failure risk"] == pytest.approx(0)


def test_combobasic():
    by_term = score("combobasic", n=3, alpha=0.5, beta=0.25)
    assert by_term["heart"] == pytest.approx(math.log(3))
    # heart failure contains heart and failure
    assert by_term["heart failure"] == pytest.approx(2 * math.log(2) + 0.5 + 0.25 * 2)
    # heart failure risk contains five candidates
    assert by_term["heart failure risk"] == pytest.approx(0.25 * 5)


def test_stopword_bounded_ngrams_do_not_contain_candidates():
    extractor = CValueTermExtractor(n=2)
    by_term = extractor.corpus_scores(
        extractor.candidate_ngrams_strings([["the", "heart"]])
    )
    assert by_term == {"heart": pytest.approx(math.log2(1.1))}
