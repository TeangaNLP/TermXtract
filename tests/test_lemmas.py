import re

import pytest
from teanga import Corpus

from termxtract import SupervisedTermExtractor, TermExtractor
from termxtract.datasets import gold_terms
from termxtract.supervised.words import tokenize_teanga

TEXTS = [
    "Cells divide quickly. A dividing cell grows.",
    "Each cell divides into two cells.",
    "Stem cells and blood cells divide.",
]
LEMMAS = {
    "cells": "cell",
    "divides": "divide",
    "dividing": "divide",
    "grows": "grow",
}
TAGS = {"quickly": "ADV", "a": "DET", "each": "DET", "into": "ADP", "two": "NUM"}
VERBS = {"divide", "divides", "dividing", "grows"}


def corpus(texts=TEXTS, with_lemmas=True):
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    corpus.add_layer_meta("upos", layer_type="seq", base="words", data="string")
    if with_lemmas:
        corpus.add_layer_meta("lemma", layer_type="seq", base="words", data="string")
    corpus.add_layer_meta("terms", layer_type="span", base="text")
    for text in texts:
        doc = corpus.add_doc(text)
        spans = [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]
        words = [text[s:e].lower() for s, e in spans]
        doc.words = spans
        doc.upos = [
            "PUNCT"
            if not w.isalnum()
            else "VERB"
            if w in VERBS
            else TAGS.get(w, "NOUN")
            for w in words
        ]
        if with_lemmas:
            doc.lemma = [LEMMAS.get(w, w) for w in words]
        doc.terms = [
            (m.start(), m.end())
            for m in re.finditer(r"(?:stem |blood )?cells?", text, re.IGNORECASE)
        ]
    return corpus


def terms(results):
    return {t["term"] for doc in results.terms for t in doc["terms"]}


METHODS = {
    "tfidf": {},
    "ridf": {},
    "cvalue": {},
    "ncvalue": {},
    "basic": {},
    "combobasic": {},
    "domaincoherence": {},
    "yake": {},
    "topicmodeling": {"num_topics": 2},
    "nmf": {"n_topics": 2},
    "weirdness": {"reference_corpus": None},
    "domainpertinence": {"reference_corpus": None},
    "relevance": {"reference_corpus": None},
}


@pytest.mark.parametrize("method", list(METHODS))
def test_terms_are_lemmas(method):
    params = {
        name: corpus(["The cell is small.", "Small cells grow."])
        if name == "reference_corpus"
        else value
        for name, value in METHODS[method].items()
    }
    results = TermExtractor(method=method, n=2, lemma_layer="lemma", **params).extract(
        corpus()
    )
    found = terms(results)
    assert not found & {"cells", "divides", "dividing", "stem cells"}
    assert "cell" in found


def test_inflected_forms_are_counted_together():
    by_lemma = TermExtractor(method="cvalue", lemma_layer="lemma").extract(corpus())
    as_written = TermExtractor(method="cvalue").extract(corpus())
    lemma_scores = {t["term"]: t["score"] for t in by_lemma.terms[1]["terms"]}
    scores = {t["term"]: t["score"] for t in as_written.terms[1]["terms"]}
    # The C-value of a single word is log2(1.1) times its frequency: the forms of "cell"
    # occur 6 times in all, and "cell" itself twice
    assert lemma_scores["cell"] == pytest.approx(3 * scores["cell"])
    assert "cells" in scores


def test_offsets_are_those_of_the_words():
    results = TermExtractor(method="tfidf", lemma_layer="lemma").extract(corpus())
    offsets = [o for o in results.offsets[1]["offsets"] if o["term"] == "cell"]
    text = TEXTS[1]
    assert [text[s:e] for s, e in (o["offset"] for o in offsets)] == ["cell", "cells"]


def test_pos_pattern_matches_lemmas():
    results = TermExtractor(
        method="tfidf", n=2, lemma_layer="lemma", pos_pattern="noun_phrase"
    ).extract(corpus())
    found = terms(results)
    assert {"cell", "stem cell", "blood cell"} <= found
    assert "divide" not in found


def test_missing_lemma_layer():
    with pytest.raises(ValueError, match="no 'lemma' layer"):
        TermExtractor(method="tfidf", lemma_layer="lemma").extract(
            corpus(with_lemmas=False)
        )


def test_lemma_layer_must_be_based_on_the_words():
    c = corpus()
    c.add_layer_meta("tokens", layer_type="span", base="text")
    for doc_id in c.doc_ids:
        doc = c.doc_by_id(doc_id)
        doc.tokens = list(doc.words)
    with pytest.raises(ValueError, match="Pass words_layer='words'"):
        TermExtractor(
            method="tfidf", words_layer="tokens", lemma_layer="lemma"
        ).extract(c)


def test_gold_terms_by_lemma():
    assert gold_terms(corpus()) == {"cell", "cells", "stem cells", "blood cells"}
    assert gold_terms(corpus(), lemma_layer="lemma") == {
        "cell",
        "stem cell",
        "blood cell",
    }


def roger(**params):
    return SupervisedTermExtractor(
        method="roger", generations=5, runs=3, random_state=0, **params
    )


def test_roger_terms_are_lemmas():
    results = roger(lemma_layer="lemma").extract(corpus())
    found = terms(results)
    assert {"stem cell", "blood cell"} <= found
    assert not any("cells" in term.split() for term in found)
    offsets = [o for o in results.offsets[2]["offsets"] if o["term"] == "stem cell"]
    assert [TEXTS[2][s:e] for s, e in (o["offset"] for o in offsets)] == ["Stem cells"]


def test_roger_counts_inflected_forms_together():
    by_lemma = roger(lemma_layer="lemma").extractor
    as_written = roger().extractor
    # "cell divide" is written "cells divide" in two documents and "cell divides" once
    _, lemma_candidates = by_lemma.extract_features(
        tokenize_teanga(corpus(), lemma_layer="lemma")
    )
    _, candidates = as_written.extract_features(tokenize_teanga(corpus()))
    assert "cell divide" in lemma_candidates
    assert {"cells divide", "cell divides"} <= set(candidates)
    assert len(lemma_candidates) < len(candidates)


def test_roger_gold_terms_by_lemma():
    assert roger(lemma_layer="lemma").extractor.gold_terms(corpus()) == gold_terms(
        corpus(), lemma_layer="lemma"
    )


def test_roger_labels_are_matched_to_lemmas():
    labels = {"stem cell": 1, "blood cell": 1}
    by_lemma = roger(lemma_layer="lemma").fit(corpus(), labels).predict(corpus())
    assert {"stem cell", "blood cell"} <= terms(by_lemma)
    # As written, no candidate is labelled 1
    with pytest.raises(ValueError, match="labelled 1"):
        roger().fit(corpus(), labels)


def test_supervised_pos_pattern_matches_lemmas():
    results = roger(lemma_layer="lemma", pos_pattern="noun_phrase").extract(corpus())
    assert {"stem cell", "blood cell"} <= terms(results)
    assert "cell divide" not in terms(results)


def test_supervised_missing_lemma_layer():
    with pytest.raises(ValueError, match="no 'lemma' layer"):
        roger(lemma_layer="lemma").extract(corpus(with_lemmas=False))
