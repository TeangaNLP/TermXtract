import re

import pytest
from teanga import Corpus

from termxtract import TermExtractor
from termxtract.unsupervised.domaincoherence import DomainCoherenceTermExtractor

TEXTS = ["Heart failure patients", "Heart failure patients", "The cat"]


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
    # The corpus has 8 words, and "heart", "failure" and "patients" occur twice each,
    # always together. The best seed term by Basic is "failure" (tied with the others,
    # and first in alphabetical order), whose best context word is "heart", with an
    # NPMI of log(2 * 8 / (2 * 2)) / -log(2 / 8) = 1. A candidate never seen near
    # "heart" has an NPMI of -1, including "heart" itself.
    extractor = DomainCoherenceTermExtractor(seed_terms=1, context_words=1)
    assert scores(extractor.extract_terms_strings(TEXTS)) == pytest.approx(
        {"failure": 1.0, "patients": 1.0, "heart": -1.0, "cat": -1.0}
    )


def test_context_words_are_content_words():
    # Tagged as an adverb, "heart" can't be a context word, so "patients" is chosen
    texts = [text.lower() for text in TEXTS]
    tagged = corpus(texts, ["ADV NOUN NOUN", "ADV NOUN NOUN", "DET NOUN"])
    extractor = DomainCoherenceTermExtractor(seed_terms=1, context_words=1)
    assert scores(extractor.extract_terms_teanga(tagged)) == pytest.approx(
        {"heart": 1.0, "failure": 1.0, "patients": -1.0, "cat": -1.0}
    )


def test_multi_word_terms():
    # The example of #52, where every candidate scored 0
    docs = [
        "Heart failure patients need heart failure treatment.",
        "Chronic heart failure affects patients.",
        "Treatment of heart failure reduces symptoms in patients.",
        "Patients with heart failure take medication daily.",
    ] * 3
    found = scores(TermExtractor(method="domaincoherence", n=2).extract(docs))
    assert len(set(found.values())) > 1
    assert found["heart failure"] > found["medication daily"]


def test_documents_list_their_own_terms():
    results = TermExtractor(method="domaincoherence").extract(corpus(TEXTS))
    assert [t["term"] for t in results.terms[2]["terms"]] == ["cat"]
    assert results.offsets[2]["offsets"] == [{"term": "cat", "offset": (4, 7)}]


def test_strings_and_teanga_agree():
    texts = [text.lower() for text in TEXTS]
    extractor = DomainCoherenceTermExtractor(n=2)
    assert scores(extractor.extract_terms_teanga(corpus(texts))) == scores(
        extractor.extract_terms_strings(texts)
    )


def test_empty_corpus():
    results = DomainCoherenceTermExtractor().extract_terms_strings(["The", "a"])
    assert results.terms == [
        {"doc_id": "doc_0", "terms": []},
        {"doc_id": "doc_1", "terms": []},
    ]
