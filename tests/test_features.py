import re

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from teanga import Corpus

from termxtract import SupervisedTermExtractor
from termxtract.association import MEASURES

TEXTS = [
    "Genetic algorithms optimize ranking functions.",
    "Machine learning and term extraction.",
    "Genetic algorithms and machine learning rank terms.",
    "Term extraction finds terms in a corpus.",
    "A corpus of machine learning papers.",
]

TERMS = ["genetic algorithms", "machine learning", "term extraction"]

NOUNS = {"algorithms", "functions", "learning", "extraction", "terms", "corpus"}
ADJECTIVES = {"genetic", "machine", "term", "ranking"}


def tag(word):
    if word.lower() in NOUNS:
        return "NOUN"
    if word.lower() in ADJECTIVES:
        return "ADJ"
    return "X"


def teanga_corpus(texts=TEXTS, terms=TERMS, pos=True):
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    if pos:
        corpus.add_layer_meta("upos", layer_type="seq", base="words", data="string")
    corpus.add_layer_meta("terms", layer_type="span", base="text")
    for text in texts:
        doc = corpus.add_doc(text)
        spans = [match.span() for match in re.finditer(r"\w+", text)]
        doc.words = spans
        if pos:
            doc.upos = [tag(text[start:end]) for start, end in spans]
        doc.terms = [
            match.span()
            for term in terms
            for match in re.finditer(term, text, flags=re.IGNORECASE)
        ]
    return corpus


def features(**params):
    params = {"n": 2, "random_state": 0, "threshold": None, **params}
    return SupervisedTermExtractor(method="features", **params)


def scores(results):
    by_term = {}
    for doc in results.terms:
        for term in doc["terms"]:
            by_term.setdefault(term["term"], set()).add(term["score"])
    assert all(len(values) == 1 for values in by_term.values())
    return {term: next(iter(values)) for term, values in by_term.items()}


def test_trains_on_the_gold_terms():
    results = scores(features().fit(teanga_corpus()).predict(teanga_corpus()))
    terms = [score for term, score in results.items() if term.lower() in TERMS]
    others = [score for term, score in results.items() if term.lower() not in TERMS]
    assert np.mean(terms) > np.mean(others)
    assert all(0 <= score <= 1 for score in results.values())


def test_offsets():
    corpus = teanga_corpus()
    results = features().fit(corpus).predict(corpus)
    for doc_id, doc in zip(corpus.doc_ids, results.offsets):
        text = corpus.doc_by_id(doc_id)["text"].text[0]
        for occurrence in doc["offsets"]:
            start, end = occurrence["offset"]
            assert text[start:end] == occurrence["term"]


def test_threshold_keeps_the_classified_terms():
    corpus = teanga_corpus()
    every = scores(features().fit(corpus).predict(corpus))
    kept = scores(features(threshold=0.5).fit(corpus).predict(corpus))
    assert set(kept) == {term for term, score in every.items() if score >= 0.5}
    assert set(kept) < set(every)


def test_strings_with_labels():
    labels = dict.fromkeys(TERMS, 1)
    results = features().fit(TEXTS, labels).predict(TEXTS)
    assert results.offsets is None
    assert "machine learning" in scores(results)


def test_needs_both_labels():
    with pytest.raises(ValueError, match="labelled 1 and one labelled 0"):
        features().fit(TEXTS, {"no such term": 1})


def test_predict_before_fit():
    with pytest.raises(ValueError, match="not been trained"):
        features().predict(TEXTS)


def test_feature_names():
    extractor = features().fit(teanga_corpus())
    names = extractor.extractor.feature_names
    assert "termhood:cvalue" in names
    assert "termhood:weirdness" not in names
    assert [f"unithood:{m}" for m in MEASURES] == [
        name for name in names if name.startswith("unithood:")
    ]
    assert {
        "shape:words",
        "shape:title case",
        "pos:first=NOUN",
        "pos:count=ADJ",
    } <= set(names)
    # Plain text has no case or tags
    names = features().fit(TEXTS, dict.fromkeys(TERMS, 1)).extractor.feature_names
    assert not any(name.startswith(("pos:", "shape:title")) for name in names)


def test_reference_corpus_adds_features():
    reference = teanga_corpus(["The cat sat on the mat.", "Dogs bark at night."], [])
    extractor = features(reference_corpus=reference).fit(teanga_corpus())
    assert "termhood:weirdness" in extractor.extractor.feature_names


def test_corpus_dependent_features_are_ranked():
    extractor = features()
    corpus = teanga_corpus()
    candidates = extractor.extractor._candidates(corpus)
    X, _, names = extractor.extractor.features(corpus, candidates, training=True)
    ranked = [i for i, name in enumerate(names) if name.startswith("unithood:")]
    assert np.all((X[:, ranked] > 0) & (X[:, ranked] <= 1))


def test_feature_importances():
    extractor = features().fit(teanga_corpus())
    importances = extractor.feature_importances()
    assert list(importances) == sorted(importances, key=importances.get, reverse=True)
    assert sum(importances.values()) == pytest.approx(1)


def test_other_classifiers():
    extractor = features(classifier=LogisticRegression(max_iter=1000)).fit(
        teanga_corpus()
    )
    assert scores(extractor.predict(teanga_corpus()))
    with pytest.raises(ValueError, match="LogisticRegression has no feature"):
        extractor.feature_importances()


def test_feature_importances_of_other_methods():
    with pytest.raises(ValueError, match="'roger' method has no feature importances"):
        SupervisedTermExtractor(method="roger").feature_importances()


def test_reproducible():
    corpus = teanga_corpus()
    assert scores(features().fit(corpus).predict(corpus)) == scores(
        features().fit(corpus).predict(corpus)
    )


def test_learned_pos_pattern():
    # The terms are all adjective + noun, so the candidates are too
    corpus = teanga_corpus()
    results = features(pos_pattern="learned").fit(corpus).predict(corpus)
    for term in scores(results):
        assert [tag(word) for word in term.split()] == ["ADJ", "NOUN"]
    assert "machine learning" in scores(results)


def test_learned_pos_pattern_needs_tags():
    with pytest.raises(ValueError, match="part-of-speech layer"):
        features(pos_pattern="learned").fit(teanga_corpus(pos=False))
    with pytest.raises(ValueError, match="only for the 'features' method"):
        SupervisedTermExtractor(method="roger", pos_pattern="learned")


def test_predict_needs_the_training_features():
    extractor = features().fit(teanga_corpus())
    with pytest.raises(ValueError, match="does not have the features"):
        extractor.predict(teanga_corpus(pos=False))
