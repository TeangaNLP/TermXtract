import math
import re
from collections import Counter

import pytest
from teanga import Corpus

from termxtract import TermExtractor
from termxtract.candidates import candidate_spans, compile_word_pattern
from termxtract.pos import compile_pos_pattern
from termxtract.unsupervised.cvalue import CValueTermExtractor

TEXTS = [
    "Heart failure raises the risk (95 % CI) of heart failure.",
    "Acute heart failure is treated. The heart beats.",
    "Chronic heart failure risk is high (95 % CI).",
]


def corpus(tagged):
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    corpus.add_layer_meta("upos", layer_type="seq", base="words", data="string")
    for text, tags in tagged:
        doc = corpus.add_doc(text)
        doc.words = [m.span() for m in re.finditer(r"\S+", text)]
        doc.upos = tags.split()
    return corpus


def terms(results):
    return {t["term"].lower() for doc in results.terms for t in doc["terms"]}


def counts(texts, n):
    words = [re.findall(r"\b\w+\b", text.lower()) for text in texts]
    return Counter(
        " ".join(doc[i : i + k])
        for doc in words
        for k in range(1, n + 1)
        for i in range(len(doc) - k + 1)
    )


METHODS = {
    "tfidf": {},
    "ridf": {},
    "cvalue": {},
    "basic": {},
    "combobasic": {},
    "topicmodeling": {"num_topics": 2},
    "nmf": {"n_topics": 2},
    "weirdness": {"reference_corpus": ["The risk is high."]},
    "domainpertinence": {"reference_corpus": ["The risk is high."]},
    "relevance": {"reference_corpus": ["The risk is high."]},
}


@pytest.mark.parametrize("method", list(METHODS))
def test_min_frequency(method):
    frequencies = counts(TEXTS, 2)
    found = terms(
        TermExtractor(method=method, n=2, min_frequency=2, **METHODS[method]).extract(
            TEXTS
        )
    )
    assert found
    assert all(frequencies[term] >= 2 for term in found)
    assert "heart failure" in found
    assert "acute" not in found


def test_candidate_spans():
    words = ["the", "heart", ",", "heart", "failure", "of", "rats"]
    spans = candidate_spans(
        words, 3, frozenset({"the", "of"}), compile_word_pattern("alphanumeric")
    )
    assert [" ".join(words[i:j]) for i, j in spans] == [
        "heart",
        "heart",
        "heart failure",
        "failure",
        "failure of rats",
        "rats",
    ]


def test_word_patterns():
    tagged = [
        (
            "Heart failure ( 95 % CI ) is a CI risk",
            "NOUN NOUN PUNCT NUM SYM NOUN PUNCT AUX DET NOUN NOUN",
        ),
        ("Heart failure ( 95 % CI ) risk", "NOUN NOUN PUNCT NUM SYM NOUN PUNCT NOUN"),
    ]
    found = terms(TermExtractor(method="cvalue", n=3).extract(corpus(tagged)))
    assert {"heart failure", "95", "ci"} <= found
    assert not {"%", "95 %", "( 95", "95 % ci"} & found
    every = terms(
        TermExtractor(method="cvalue", n=3, word_pattern=None).extract(corpus(tagged))
    )
    assert {"%", "95 % ci"} <= every
    atr4s = terms(
        TermExtractor(method="cvalue", n=3, word_pattern="atr4s").extract(
            corpus(tagged)
        )
    )
    assert "heart failure" in atr4s
    assert not {"95", "ci"} & atr4s
    with pytest.raises(ValueError, match="Invalid word pattern"):
        compile_word_pattern("[")


def test_only_matching_occurrences_are_counted():
    tagged = [
        ("light bulbs", "NOUN NOUN"),
        ("light bulbs", "NOUN NOUN"),
        ("light rain", "ADJ NOUN"),
    ]
    results = TermExtractor(method="cvalue", n=2, pos_pattern="<NOUN>+").extract(
        corpus(tagged)
    )
    scores = {t["term"]: t["score"] for doc in results.terms for t in doc["terms"]}
    # "light" occurs three times, but only twice as a noun, both inside "light bulbs"
    assert scores["light"] == pytest.approx(math.log2(1.1) * (2 - 2))
    assert scores["light bulbs"] == pytest.approx(math.log2(2.1) * 2)
    assert "light rain" not in scores
    # The adjective is not an occurrence, so it has no offsets
    assert results.offsets[2]["offsets"] == [{"term": "rain", "offset": (6, 10)}]


def test_only_candidates_contain_others():
    texts = ["heart failure", "heart failure", "heart beat", "heart"]
    extractor = CValueTermExtractor(n=2, min_frequency=2)
    words = [text.split() for text in texts]
    scores = extractor.corpus_scores(extractor.candidate_ngrams_strings(words))
    # "heart beat" occurs once, so only "heart failure" contains "heart"
    assert set(scores) == {"heart", "failure", "heart failure"}
    assert scores["heart"] == pytest.approx(math.log2(1.1) * (4 - 2))


def test_pos_pattern_needs_tags():
    extractor = CValueTermExtractor(pos_pattern="noun_phrase")
    with pytest.raises(ValueError, match="part-of-speech tags"):
        extractor.extract_terms_strings(TEXTS)
    assert extractor.pos_pattern == compile_pos_pattern("noun_phrase")
