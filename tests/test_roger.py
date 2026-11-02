import math
import re
import warnings

import numpy as np
import pytest
from scipy.stats import chi2_contingency
from teanga import Corpus

from termxtract.supervised.roger import (
    MEASURES,
    RogerTermExtractor,
    association_measures,
)
from termxtract.supervised.term_extractor import SupervisedTermExtractor
from termxtract.supervised.words import tokenize_strings

CORPUS = [
    "Genetic algorithms optimize ranking functions.",
    "Machine learning and term extraction.",
    "Genetic algorithms and machine learning rank terms.",
    "Term extraction finds terms in a corpus.",
]

LABELS = {
    "genetic algorithms": 1,
    "machine learning": 1,
    "term extraction": 1,
    "ranking functions": 0,
    "and machine": 0,
    "finds terms": 0,
}


def roger(**params):
    params = {"generations": 10, "runs": 3, "n": 2, "random_state": 0, **params}
    return SupervisedTermExtractor(method="roger", **params)


def scores(results):
    return {t["term"]: t["score"] for doc in results.terms for t in doc["terms"]}


def teanga_corpus(texts):
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    for text in texts:
        doc = corpus.add_doc(text)
        doc.words = [match.span() for match in re.finditer(r"\w+", text)]
    return corpus


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


def test_equally_frequent_candidates_differ():
    # "machine learning" and "and machine" both occur twice (#30)
    extractor = RogerTermExtractor(n=2)
    X, terms = extractor.extract_features(tokenize_strings(CORPUS))
    features = dict(zip(terms, X.tolist()))
    assert features["machine learning"] != features["and machine"]


def test_only_multiword_candidates():
    results = roger(n=3, stopwords=None).extract(CORPUS, LABELS)
    terms = scores(results)
    assert "genetic algorithms optimize" in terms
    assert all(len(term.split()) in (2, 3) for term in terms)


def test_requires_n_at_least_2():
    with pytest.raises(ValueError, match="n must be at least 2"):
        roger(n=1)


def test_learns_labelled_ranking():
    results = roger(generations=30, stopwords=None).extract(CORPUS, LABELS)
    term_scores = scores(results)
    positives = [term_scores[t] for t, label in LABELS.items() if label == 1]
    negatives = [term_scores[t] for t, label in LABELS.items() if label == 0]
    assert min(positives) > max(negatives)


def test_reproducible():
    assert scores(roger().extract(CORPUS, LABELS)) == scores(
        roger().extract(CORPUS, LABELS)
    )


def test_teanga_matches_strings():
    from_strings = roger().extract(CORPUS, LABELS)
    from_teanga = roger().extract(teanga_corpus(CORPUS), LABELS)
    assert [d["terms"] for d in from_teanga.terms] == [
        d["terms"] for d in from_strings.terms
    ]
    assert [doc["doc_id"] for doc in from_teanga.terms] == list(
        from_teanga.corpus.doc_ids
    )


def test_rare_positives_give_finite_scores():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        results = roger().extract(
            CORPUS, {"machine learning": 1, "and machine": 0, "and term": 0}
        )
    assert np.all(np.isfinite(list(scores(results).values())))


@pytest.mark.parametrize(
    "labels",
    [{}, {"machine learning": 0}, {"machine learning": 1}, {"machine": 1, "and": 0}],
)
def test_requires_positive_and_negative_labels(labels):
    with pytest.raises(ValueError, match="labelled 1 and one labelled 0"):
        roger().extract(CORPUS, labels)


def test_predict_new_documents():
    extractor = roger()
    extractor.extract(CORPUS, LABELS)
    new_scores = extractor.extractor.predict_terms_strings(
        ["Genetic algorithms learn ranking functions."]
    )
    assert set(new_scores) == {
        "genetic algorithms",
        "algorithms learn",
        "learn ranking",
        "ranking functions",
    }


@pytest.mark.parametrize("method", ["roger", "adaboost"])
def test_each_document_lists_its_own_candidates(method):
    params = (
        {"generations": 5, "runs": 2, "random_state": 0} if method == "roger" else {}
    )
    results = SupervisedTermExtractor(
        method=method, n=2, stopwords=None, **params
    ).extract(CORPUS, LABELS)
    corpus_scores = {}
    for doc, words in zip(results.terms, tokenize_strings(CORPUS)):
        doc_ngrams = {
            " ".join(words[i : i + k])
            for k in (1, 2)
            for i in range(len(words) - k + 1)
        }
        doc_terms = {t["term"]: t["score"] for t in doc["terms"]}
        assert set(doc_terms) <= doc_ngrams
        for term, score in doc_terms.items():
            # Scores are corpus-level, so a term has the same score in every document
            assert corpus_scores.setdefault(term, score) == score
    assert "genetic algorithms" not in {t["term"] for t in results.terms[1]["terms"]}
    assert "genetic algorithms" in {t["term"] for t in results.terms[2]["terms"]}
