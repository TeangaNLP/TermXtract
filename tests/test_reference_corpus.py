import math
import re

import pytest
from teanga import Corpus

from termxtract import TermExtractor

TARGET = [
    "photosynthesis needs light",
    "the leaf uses photosynthesis",
]
REFERENCE = [
    "the light is on",
    "the dog is in the park",
]
# Target: 7 words, photosynthesis x2 (in both docs). Reference: 10 words, no
# photosynthesis, light x1, the x3.
METHODS = ["weirdness", "domainpertinence", "relevance"]


def teanga_corpus(texts):
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    for text in texts:
        doc = corpus.add_doc(text)
        doc.words = [match.span() for match in re.finditer(r"\w+", text)]
    return corpus


def scores(method, corpus, reference):
    results = TermExtractor(method=method, reference_corpus=reference).extract(corpus)
    return {t["term"]: t["score"] for doc in results.terms for t in doc["terms"]}


def test_weirdness_scores():
    s = scores("weirdness", TARGET, REFERENCE)
    assert s["photosynthesis"] == pytest.approx((2 / 7) / (1 / 10))
    assert s["light"] == pytest.approx((1 / 7) / (2 / 10))
    assert s["the"] == pytest.approx((1 / 7) / (4 / 10))


def test_domain_pertinence_scores():
    s = scores("domainpertinence", TARGET, REFERENCE)
    assert s["photosynthesis"] == 1.0
    assert s["light"] == 1.0
    assert s["the"] == pytest.approx((1 / 7) / (3 / 10))


def test_relevance_scores():
    s = scores("relevance", TARGET, REFERENCE)
    assert s["photosynthesis"] == pytest.approx(
        1 - 1 / math.log2(2 + (2 / 7) * 1.0 / (1 / 10))
    )
    assert s["the"] == pytest.approx(1 - 1 / math.log2(2 + (1 / 7) * 0.5 / (4 / 10)))
    assert all(0 <= score < 1 for score in s.values())


@pytest.mark.parametrize("method", METHODS)
def test_domain_term_ranked_first(method):
    results = TermExtractor(method=method, reference_corpus=REFERENCE).extract(TARGET)
    terms = results.terms[1]["terms"]
    # Domain Pertinence ties every term not more typical of the reference at 1
    top = [t["term"] for t in terms if t["score"] == terms[0]["score"]]
    assert "photosynthesis" in top
    assert "the" not in top


@pytest.mark.parametrize("method", METHODS)
def test_teanga_matches_strings(method):
    # Lower-case text, so the Teanga words match the lower-cased plain-text tokens
    assert scores(method, teanga_corpus(TARGET), teanga_corpus(REFERENCE)) == (
        pytest.approx(scores(method, TARGET, REFERENCE))
    )


@pytest.mark.parametrize("method", METHODS)
def test_reference_type_must_match(method):
    with pytest.raises(TypeError, match="must also be a Teanga Corpus"):
        TermExtractor(method=method, reference_corpus=teanga_corpus(REFERENCE)).extract(
            TARGET
        )
    with pytest.raises(TypeError, match="must also be a list of strings"):
        TermExtractor(method=method, reference_corpus=REFERENCE).extract(
            teanga_corpus(TARGET)
        )


@pytest.mark.parametrize("method", METHODS)
def test_reference_required(method):
    with pytest.raises(ValueError, match="requires a reference corpus"):
        TermExtractor(method=method)
    with pytest.raises(ValueError, match="contains no words"):
        TermExtractor(method=method, reference_corpus=[""]).extract(TARGET)
