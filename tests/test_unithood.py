import math
import re

import pytest
from teanga import Corpus

from termxtract import TermExtractor
from termxtract.association import MEASURES, NgramCounts

UNITHOOD_METHODS = {
    "pmi": "MI",
    "llr": "L",
    "chi2": "Khi2",
    "dice": "Dice",
    "tscore": "T test",
}

CORPUS = [
    "Machine learning is a field of artificial intelligence.",
    "Deep learning is a subfield of machine learning.",
    "Natural language processing applies machine learning to text.",
]


def scores(results):
    by_term = {}
    for doc in results.terms:
        for term in doc["terms"]:
            by_term.setdefault(term["term"], set()).add(term["score"])
    assert all(len(values) == 1 for values in by_term.values())
    return {term: next(iter(values)) for term, values in by_term.items()}


def teanga_corpus(texts):
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    for text in texts:
        doc = corpus.add_doc(text)
        doc.words = [match.span() for match in re.finditer(r"\w+", text)]
    return corpus


@pytest.mark.parametrize("method", list(UNITHOOD_METHODS))
def test_scores_are_the_measure_of_the_split(method):
    results = TermExtractor(method=method, stopwords=None).extract(CORPUS)
    counts = NgramCounts([re.findall(r"\w+", doc.lower()) for doc in CORPUS], 2)
    column = MEASURES.index(UNITHOOD_METHODS[method])
    expected = counts.measures(("machine", "learning"), 1)[column]
    assert scores(results)["machine learning"] == pytest.approx(expected)


def test_pmi():
    # 21 bigrams: "machine learning" occurs 3 times, "machine" starts 3 bigrams and
    # "learning" ends 4
    results = TermExtractor(method="pmi", stopwords=None).extract(CORPUS)
    assert scores(results)["machine learning"] == pytest.approx(
        math.log2(3 * 21 / (3 * 4))
    )


def test_single_words_are_not_ranked():
    results = TermExtractor(method="pmi", n=3).extract(CORPUS)
    terms = scores(results)
    assert terms
    assert all(" " in term for term in terms)


def test_n_defaults_to_2():
    assert TermExtractor(method="dice").extractor.n == 2
    assert TermExtractor(method="tfidf").extractor.n == 1


def test_n_must_be_at_least_2():
    with pytest.raises(ValueError, match="n must be at least 2"):
        TermExtractor(method="pmi", n=1)


def test_unknown_aggregation():
    with pytest.raises(ValueError, match="Available aggregations: min, mean"):
        TermExtractor(method="pmi", aggregation="max")


@pytest.mark.parametrize("aggregation", ["min", "mean"])
def test_aggregation_over_splits(aggregation):
    results = TermExtractor(
        method="dice", n=3, aggregation=aggregation, stopwords=None
    ).extract(CORPUS)
    counts = NgramCounts([re.findall(r"\w+", doc.lower()) for doc in CORPUS], 3)
    column = MEASURES.index("Dice")
    trigram = ("of", "machine", "learning")
    splits = [counts.measures(trigram, i)[column] for i in (1, 2)]
    # of | machine learning: 2 / (2 + 2), of machine | learning: 2 / (1 + 2)
    assert splits == pytest.approx([1 / 2, 2 / 3])
    expected = min(splits) if aggregation == "min" else sum(splits) / 2
    assert scores(results)["of machine learning"] == pytest.approx(expected)


def test_filters_change_the_candidates_but_not_the_scores():
    unfiltered = scores(TermExtractor(method="llr", stopwords=None).extract(CORPUS))
    filtered = scores(TermExtractor(method="llr").extract(CORPUS))
    assert "is a" in unfiltered
    assert "is a" not in filtered
    assert filtered["machine learning"] == unfiltered["machine learning"]


def test_teanga_offsets():
    # Teanga words are read as written, so lower-case them to match the strings
    texts = [text.lower() for text in CORPUS]
    results = TermExtractor(method="tscore").extract(teanga_corpus(texts))
    assert scores(results)["machine learning"] == pytest.approx(
        scores(TermExtractor(method="tscore").extract(texts))["machine learning"]
    )
    offsets = [
        occurrence["offset"]
        for doc in results.offsets
        for occurrence in doc["offsets"]
        if occurrence["term"] == "machine learning"
    ]
    assert offsets == [(0, 16), (31, 47), (36, 52)]
