import re

import numpy as np
import pytest
from teanga import Corpus

from termxtract import ATEResults, TermExtractor
from termxtract.embeddings import Encoder
from termxtract.rerank import AdaText, Reranker, SemReRank
from termxtract.unsupervised.textrank import window_graph

VECTORS = {
    "heart": [1.0, 0.0, 0.0],
    "failure": [0.9, 0.1, 0.0],
    "pump": [0.8, 0.2, 0.0],
    "kidney": [0.0, 1.0, 0.0],
    "disease": [0.0, 0.9, 0.1],
    "tree": [0.0, 0.0, 1.0],
}


class FakeEncoder(Encoder):
    """Embeds a text as the mean of the hand-set vectors of its words."""

    def __init__(self):
        self.texts = []

    def encode(self, texts, masks=None):
        self.texts.extend(texts)
        rows = [np.mean([VECTORS.get(w, [0.0, 0.0, 0.1]) for w in t], 0) for t in texts]
        return np.array([row / np.linalg.norm(row) for row in rows])


class Fixed(Reranker):
    """A re-ranker with given word scores."""

    def __init__(self, scores, **params):
        super().__init__(**params)
        self.scores = scores

    def word_scores(self, docs, terms, seeds):
        self.seen = (docs, terms, seeds)
        return self.scores


def results(*docs, offsets=None):
    return ATEResults(
        ["text"] * len(docs),
        [
            {
                "doc_id": f"doc_{i}",
                "terms": [{"term": t, "score": s} for t, s in terms.items()],
            }
            for i, terms in enumerate(docs)
        ],
        offsets,
    )


def scores(results):
    return {t["term"]: t["score"] for d in results.terms for t in d["terms"]}


def test_rerank_formula():
    reranker = Fixed({"heart": 0.5, "failure": 1.0}, seeds=["heart"])
    revised = reranker.rerank(
        results(
            {"Heart failure": 2.0, "heart of": 1.0, "the": 3.0, "pump": -1.0},
            {"failure": -1.0},
        ),
        [["heart", "failure"]],
    )
    assert scores(revised) == pytest.approx(
        {
            # Words are compared in lower case, and stopwords left out
            "Heart failure": 2.0 * 1.75,
            "heart of": 1.0 * 1.5,
            # No words that can be nodes, so unchanged
            "the": 3.0,
            "pump": -1.0,
            # Negative scores are divided, so they move up too
            "failure": -1.0 / 2.0,
        }
    )
    assert [t["term"] for t in revised.terms[0]["terms"]] == [
        "Heart failure",
        "the",
        "heart of",
        "pump",
    ]


def test_rerank_keeps_offsets():
    offsets = [{"doc_id": "doc_0", "offsets": [{"term": "heart", "offset": (0, 5)}]}]
    revised = Fixed({}, seeds=["heart"]).rerank(
        results({"heart": 1.0}, offsets=offsets), []
    )
    assert revised.offsets is offsets


def test_normalize():
    reranker = Fixed({"heart": 1.0}, seeds=["heart"])
    reranker.normalize = True
    revised = reranker.rerank(results({"heart": 4.0, "pump": -8.0}), [])
    assert scores(revised) == pytest.approx({"heart": 1.0, "pump": -1.0})


def test_seeds_most_frequent():
    reranker = Fixed({}, seeds=2)
    reranker.rerank(
        results({"heart": 1.0, "heart failure": 1.0, "pump": 1.0, "kidney": 1.0}),
        [["Heart", "failure", "pump"], ["heart", "failure", "kidney", "kidney"]],
    )
    docs, terms, seeds = reranker.seen
    assert docs[0] == ["heart", "failure", "pump"]
    assert seeds == ["heart", "heart failure"]
    assert Reranker.most_frequent(docs, terms, 3) == [
        "heart",
        "heart failure",
        "kidney",
    ]


def test_seeds_given():
    reranker = Fixed({}, seeds=["Heart Failure"])
    reranker.rerank(results({"heart": 1.0}), [])
    assert reranker.seen[2] == ["heart failure"]


def test_window_graph():
    vocabulary, ids, weights = window_graph(
        [["heart", "of", "failure"], ["pump", "heart"]],
        [[True, False, True], [True, True]],
        2,
    )
    assert vocabulary == {"heart": 0, "failure": 1, "pump": 2}
    assert ids.tolist() == [0, -1, 1, -1, -1, 2, 0, -1, -1]
    # heart and failure are linked in the first document, pump and heart in the second,
    # but failure and pump are not linked across documents
    assert weights.toarray().tolist() == [[0, 1, 1], [1, 0, 0], [1, 0, 0]]


def test_adatext():
    encoder = FakeEncoder()
    adatext = AdaText(seeds=["heart"], threshold=0.9, encoder=encoder)
    words = adatext.word_scores(
        [["heart", "failure", "of", "pump", "kidney"], ["kidney", "disease"]],
        [],
        ["heart"],
    )
    # pump is 0.97 similar to heart, kidney and disease 0 or less
    assert set(words) == {"heart", "failure", "pump"}
    # A triangle, so every node has the same score, and the scores have unit length
    assert list(words.values()) == pytest.approx([1 / np.sqrt(3)] * 3)
    assert ["heart"] in encoder.texts


def test_adatext_window():
    adatext = AdaText(seeds=["heart"], threshold=0.9, window=2, encoder=FakeEncoder())
    words = adatext.word_scores([["heart", "failure", "of", "pump"]], [], ["heart"])
    # pump is not within a window of 2 words of another node, so it has no links
    heart, failure, pump = (words[w] for w in ("heart", "failure", "pump"))
    assert heart == pytest.approx(failure)
    assert pump < heart
    assert np.linalg.norm([heart, failure, pump]) == pytest.approx(1)


def test_adatext_no_similar_words():
    adatext = AdaText(seeds=["heart"], threshold=0.99, encoder=FakeEncoder())
    assert adatext.word_scores([["kidney", "tree"]], [], ["heart"]) == {}


def test_semrerank_cutoffs():
    semrerank = SemReRank(rel_top=0.5, encoder=FakeEncoder())
    vectors = FakeEncoder().encode([[w] for w in ("heart", "failure", "pump", "tree")])
    cutoffs = semrerank.cutoffs(vectors)
    similarities = vectors @ vectors.T
    # Each word's cutoff is the second highest similarity to the other three
    for i in range(4):
        assert cutoffs[i] == pytest.approx(sorted(np.delete(similarities[i], i))[-2])


def test_semrerank():
    semrerank = SemReRank(seeds=["heart"], rel_top=1.0, encoder=FakeEncoder())
    words = semrerank.word_scores(
        [["heart", "failure", "kidney"], ["failure", "pump"]],
        ["heart", "failure", "pump", "kidney"],
        ["heart"],
    )
    # In the first document, heart and failure are linked and kidney is not related
    # to either; PageRank jumps to heart. The second document has no seed, so counts 0.
    heart = 0.15 / (1 - 0.85**2)
    failure = 0.85 * heart
    assert words == pytest.approx(
        {"heart": 1.0, "failure": failure / heart, "pump": 0.0, "kidney": 0.0}
    )


def test_semrerank_rel_min():
    semrerank = SemReRank(
        seeds=["heart"], rel_min=0.999, rel_top=1.0, encoder=FakeEncoder()
    )
    words = semrerank.word_scores(
        [["heart", "failure"]], ["heart", "failure"], ["heart"]
    )
    # Not related, so failure is not linked and gets nothing from heart
    assert words == pytest.approx({"heart": 1.0, "failure": 0.0})


def corpus(texts):
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    for text in texts:
        doc = corpus.add_doc(text)
        doc.words = [m.span() for m in re.finditer(r"\w+", text)]
    return corpus


TEXTS = [
    "Heart failure is a disease of the heart pump.",
    "Kidney disease and heart failure.",
]


@pytest.mark.parametrize("reranker", [AdaText, SemReRank])
def test_term_extractor(reranker):
    base = TermExtractor(method="tfidf", n=2)
    extractor = TermExtractor(
        method="tfidf", n=2, rerank=reranker(seeds=["heart"], encoder=FakeEncoder())
    )
    for texts in (TEXTS, corpus(TEXTS)):
        before = base.extract(texts)
        after = extractor.extract(texts)
        assert [set(scores(r)) for r in (before, after)] == [set(scores(before))] * 2
        assert after.offsets == before.offsets
        assert scores(after) != scores(before)


def test_invalid():
    with pytest.raises(ValueError, match="seeds"):
        AdaText(seeds=0)
    with pytest.raises(ValueError, match="window"):
        AdaText(window=1)
    with pytest.raises(ValueError, match="rel_top"):
        SemReRank(rel_top=0)
