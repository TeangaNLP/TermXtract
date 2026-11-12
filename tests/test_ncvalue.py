import math
import re

import pytest
from teanga import Corpus

from termxtract import TermExtractor
from termxtract.unsupervised.ncvalue import NCValueTermExtractor

TEXTS = ["heart failure patients", "heart failure patients", "the cat"]

# The C-value of a single word is log2(1.1) times its frequency
CVALUE = math.log2(1.1)


def corpus(texts, tags=None):
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    if tags is not None:
        corpus.add_layer_meta("upos", layer_type="seq", base="words", data="string")
    for k, text in enumerate(texts):
        doc = corpus.add_doc(text)
        doc.words = [m.span() for m in re.finditer(r"\w+", text)]
        if tags is not None:
            doc.upos = tags[k].split()
    return corpus


def scores(results):
    return {t["term"]: t["score"] for doc in results.terms for t in doc["terms"]}


def test_scores():
    # "heart", "failure" and "patients" tie on C-value, so the top term is "failure",
    # first in alphabetical order. Its context words, "heart" and "patients", occur
    # with the only top term, so they weigh 1. Each occurs twice near "failure" and
    # the other, and "failure" weighs 0, as it is not a context word of a top term.
    extractor = NCValueTermExtractor(top_terms=1)
    assert scores(extractor.extract_terms_strings(TEXTS)) == pytest.approx(
        {
            "failure": 0.8 * 2 * CVALUE + 0.2 * (2 + 2),
            "heart": 0.8 * 2 * CVALUE + 0.2 * 2,
            "patients": 0.8 * 2 * CVALUE + 0.2 * 2,
            "cat": 0.8 * CVALUE,
        }
    )


def test_context_words_are_content_words():
    # Tagged as an adverb, "heart" can't be a context word
    tagged = corpus(TEXTS, ["ADV NOUN NOUN", "ADV NOUN NOUN", "DET NOUN"])
    extractor = NCValueTermExtractor(top_terms=1)
    assert scores(extractor.extract_terms_teanga(tagged)) == pytest.approx(
        {
            "failure": 0.8 * 2 * CVALUE + 0.2 * 2,
            "heart": 0.8 * 2 * CVALUE + 0.2 * 2,
            "patients": 0.8 * 2 * CVALUE,
            "cat": 0.8 * CVALUE,
        }
    )


def test_window_size():
    # "patients" is two words from "heart", so out of a window of one word
    extractor = NCValueTermExtractor(top_terms=1, window_size=1)
    found = scores(extractor.extract_terms_strings(TEXTS))
    assert found["heart"] == pytest.approx(0.8 * 2 * CVALUE)


def test_cvalue_weight_of_one_is_cvalue():
    docs = [
        "Heart failure patients need heart failure treatment.",
        "Chronic heart failure affects patients.",
    ]
    ncvalue = TermExtractor(method="ncvalue", n=2, cvalue_weight=1).extract(docs)
    cvalue = TermExtractor(method="cvalue", n=2).extract(docs)
    assert scores(ncvalue) == pytest.approx(scores(cvalue))


def test_multi_word_terms():
    docs = [
        "Heart failure patients need heart failure treatment.",
        "Chronic heart failure affects patients.",
        "Treatment of heart failure reduces symptoms in patients.",
        "Patients with heart failure take medication daily.",
    ] * 3
    found = scores(TermExtractor(method="ncvalue", n=2, top_terms=5).extract(docs))
    assert found["heart failure"] > found["medication daily"]


def test_documents_list_their_own_terms():
    results = TermExtractor(method="ncvalue").extract(corpus(TEXTS))
    assert [t["term"] for t in results.terms[2]["terms"]] == ["cat"]
    assert results.offsets[2]["offsets"] == [{"term": "cat", "offset": (4, 7)}]


def test_strings_and_teanga_agree():
    extractor = NCValueTermExtractor(n=2)
    assert scores(extractor.extract_terms_teanga(corpus(TEXTS))) == scores(
        extractor.extract_terms_strings(TEXTS)
    )


def test_empty_corpus():
    results = NCValueTermExtractor().extract_terms_strings(["The", "a"])
    assert results.terms == [
        {"doc_id": "doc_0", "terms": []},
        {"doc_id": "doc_1", "terms": []},
    ]
