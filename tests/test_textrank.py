import re

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from teanga import Corpus

from termxtract import TermExtractor
from termxtract.unsupervised.textrank import (
    DAMPING,
    TextRankTermExtractor,
    pagerank,
)


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


def scores(results, doc=None):
    docs = results.terms if doc is None else [results.terms[doc]]
    return {t["term"]: t["score"] for d in docs for t in d["terms"]}


def words(texts, **params):
    return scores(TextRankTermExtractor(n=1, **params).extract_terms_strings(texts))


def test_pagerank():
    # A path a - b - c with weights 1 and 2, and a node d with no edges
    weights = csr_matrix(
        np.array([[0, 1, 0, 0], [1, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]], dtype=float)
    )
    jump = np.array([0.4, 0.3, 0.2, 0.1])
    found = pagerank(weights, jump)
    assert found.sum() == pytest.approx(1)
    # Each node gets its share of the walk from its neighbours, and of the jumps, which
    # are taken with probability 1 - DAMPING, or 1 from d
    a, b, c, d = found
    jumps = 1 - DAMPING + DAMPING * d
    assert a == pytest.approx(DAMPING * b / 3 + jumps * jump[0])
    assert b == pytest.approx(DAMPING * (a + c) + jumps * jump[1])
    assert c == pytest.approx(DAMPING * 2 * b / 3 + jumps * jump[2])
    assert d == pytest.approx(jumps * jump[3])


def test_window():
    # In a window of one word, "dog" is linked to "cat" and "bird", which are not
    # linked to each other
    found = words(["cat dog bird"])
    assert found["dog"] > found["cat"] == pytest.approx(found["bird"])
    assert found["dog"] + 2 * found["cat"] == pytest.approx(1)
    # In a window of two, all three are linked
    found = words(["cat dog bird"], window_size=2)
    assert found["dog"] == pytest.approx(found["cat"]) == pytest.approx(1 / 3)


def test_stopwords_take_places_in_the_window():
    # "of" is not a node, but keeps "degree" and "freedom" out of a window of one
    found = words(["degree of freedom"])
    assert set(found) == {"degree", "freedom"}
    assert found["degree"] == pytest.approx(0.5)


def test_single_weighs_edges_by_cooccurrences():
    # "dog" co-occurs with "cat" three times and with "bird" once
    texts = ["cat dog cat dog cat dog bird"]
    plain = words(texts)
    single = words(texts, variant="single", window_size=1)
    assert plain["cat"] == pytest.approx(plain["bird"])
    assert single["cat"] > single["bird"]


def test_position_favours_early_words():
    # On a path, the first and last words score alike unless the position counts
    texts = ["alpha beta gamma delta"]
    single = words(texts, variant="single", window_size=1)
    position = words(texts, variant="position", window_size=1)
    assert single["alpha"] == pytest.approx(single["delta"])
    assert position["alpha"] > position["delta"]
    assert position["beta"] > position["gamma"]


def test_default_windows():
    assert TextRankTermExtractor().window_size == 1
    assert TextRankTermExtractor(variant="single").window_size == 9
    assert TextRankTermExtractor(variant="position").window_size == 9


def test_candidates_sum_their_words():
    texts = ["heart failure patients. Patients with heart failure."]
    found = scores(TextRankTermExtractor(n=2).extract_terms_strings(texts))
    assert found["heart failure"] == pytest.approx(found["heart"] + found["failure"])


def test_nodes_are_nouns_and_adjectives():
    # Tagged as a verb, "failure" is not a node, so its candidates score as "heart"
    texts = ["heart failure"]
    tagged = corpus(texts, ["NOUN VERB"])
    found = scores(TextRankTermExtractor(n=2).extract_terms_teanga(tagged))
    assert found["failure"] == 0
    assert found["heart failure"] == pytest.approx(found["heart"]) == 1


def test_document_graphs():
    texts = ["cat dog bird", "cat bird"]
    corpus_graph = TextRankTermExtractor().extract_terms_strings(texts)
    assert scores(corpus_graph, 0)["cat"] == scores(corpus_graph, 1)["cat"]
    by_doc = TextRankTermExtractor(graph="document").extract_terms_strings(texts)
    for k, text in enumerate(texts):
        alone = TextRankTermExtractor().extract_terms_strings([text])
        assert scores(by_doc, k) == pytest.approx(scores(alone))


def test_windows_do_not_cross_documents():
    # "dog" and "bird" would be linked if the documents ran on
    found = words(["cat dog", "bird"])
    assert found["cat"] == pytest.approx(found["dog"])
    assert found["bird"] < found["cat"]


def test_offsets():
    texts = ["Heart failure patients and patients with Heart failure"]
    results = TermExtractor(method="textrank").extract(corpus(texts))
    for occurrence in results.offsets[0]["offsets"]:
        start, end = occurrence["offset"]
        assert texts[0][start:end] == occurrence["term"]
    assert "Heart failure" in scores(results)


def test_strings_and_teanga_agree():
    texts = ["heart failure patients", "patients with heart failure"]
    for variant in ("plain", "single", "position"):
        extractor = TextRankTermExtractor(variant=variant)
        assert scores(extractor.extract_terms_teanga(corpus(texts))) == pytest.approx(
            scores(extractor.extract_terms_strings(texts))
        )


def test_invalid_parameters():
    with pytest.raises(ValueError, match="variant"):
        TextRankTermExtractor(variant="topic")
    with pytest.raises(ValueError, match="graph"):
        TextRankTermExtractor(graph="sentence")
    with pytest.raises(ValueError, match="window_size"):
        TextRankTermExtractor(window_size=0)


def test_empty_corpus():
    results = TextRankTermExtractor().extract_terms_strings(["The", ""])
    assert results.terms == [
        {"doc_id": "doc_0", "terms": []},
        {"doc_id": "doc_1", "terms": []},
    ]
