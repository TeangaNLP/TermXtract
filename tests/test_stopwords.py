import re

import numpy as np
import pytest
from teanga import Corpus

from termxtract import SupervisedTermExtractor, TermExtractor
from termxtract.stopwords import ENGLISH_STOPWORDS, is_candidate, load_stopwords

CORPUS = [
    "The number of degrees of freedom of a system is small.",
    "Each degree of freedom of the system adds energy.",
    "A degree of freedom is a parameter of the system.",
]


def all_terms(results):
    return {t["term"] for doc in results.terms for t in doc["terms"]}


def test_load_stopwords():
    assert load_stopwords("english") == ENGLISH_STOPWORDS
    assert load_stopwords(None) == frozenset()
    assert load_stopwords(["The", "OF"]) == {"the", "of"}
    with pytest.raises(ValueError, match="Unknown stopword list"):
        load_stopwords("klingon")


def test_english_stopwords_are_function_words():
    assert len(ENGLISH_STOPWORDS) == 198
    assert {"the", "of", "don", "ll"} <= ENGLISH_STOPWORDS
    assert not {"system", "computer", "interest", "part"} & ENGLISH_STOPWORDS


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("system", True),
        ("degree of freedom", True),
        ("of", False),
        ("of a", False),
        ("the system", False),
        ("system is", False),
        ("The system", False),
    ],
)
def test_is_candidate(term, expected):
    assert is_candidate(term, ENGLISH_STOPWORDS) is expected


def test_filtered_by_default():
    terms = all_terms(TermExtractor(method="cvalue", n=3).extract(CORPUS))
    assert "degree of freedom" in terms
    assert "system" in terms
    assert not any(t.split()[0] in ENGLISH_STOPWORDS for t in terms)
    assert not any(t.split()[-1] in ENGLISH_STOPWORDS for t in terms)


def test_no_filtering():
    terms = all_terms(
        TermExtractor(method="cvalue", n=3, stopwords=None).extract(CORPUS)
    )
    assert {"of", "of a", "the system"} <= terms


def test_custom_stopwords():
    terms = all_terms(
        TermExtractor(method="tfidf", n=1, stopwords=["system"]).extract(CORPUS)
    )
    assert "system" not in terms
    assert "the" in terms


def test_filtered_scores_unchanged():
    filtered = TermExtractor(method="tfidf", n=2).extract(CORPUS)
    unfiltered = TermExtractor(method="tfidf", n=2, stopwords=None).extract(CORPUS)
    for doc, all_doc in zip(filtered.terms, unfiltered.terms):
        all_scores = {t["term"]: t["score"] for t in all_doc["terms"]}
        assert doc["terms"] == [
            t for t in all_doc["terms"] if is_candidate(t["term"], ENGLISH_STOPWORDS)
        ]
        assert all(all_scores[t["term"]] == t["score"] for t in doc["terms"])


def test_teanga_case_and_offsets():
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    for text in CORPUS:
        doc = corpus.add_doc(text)
        doc.words = [match.span() for match in re.finditer(r"\w+", text)]
    results = TermExtractor(method="tfidf", n=2).extract(corpus)
    # "The" and "A" are capitalised in the text but still stopwords
    assert not {"The", "A", "The number", "A degree"} & all_terms(results)
    kept = all_terms(results)
    assert all(o["term"] in kept for doc in results.offsets for o in doc["offsets"])


def test_rake_uses_its_own_stoplist():
    terms = all_terms(
        TermExtractor(
            method="rake", stoplist=["of"], phrase_delimiters=["."], stopwords=None
        ).extract(CORPUS)
    )
    assert any(t.startswith("the ") for t in terms)


def test_supervised_filtered_by_default():
    np.random.seed(0)
    labels = {"degree of freedom": 1, "system": 1, "the": 0, "of": 0}
    results = SupervisedTermExtractor(method="roger", n=3, generations=5).extract(
        CORPUS, labels
    )
    terms = all_terms(results)
    assert "degree of freedom" in terms
    assert all(is_candidate(t, ENGLISH_STOPWORDS) for t in terms)
