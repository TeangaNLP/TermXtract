import importlib.util
import re
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
from teanga import Corpus

from termxtract import TermExtractor
from termxtract.embeddings import Encoder
from termxtract.unsupervised.embedrank import EmbedRankTermExtractor

HAS_TRANSFORMERS = all(
    importlib.util.find_spec(module) is not None for module in ("torch", "transformers")
)


class FakeEncoder(Encoder):
    """
    Embeds a text as the count of each word, plus a constant, so that the scores can
    be worked out by hand. Chunks are `size` words long.
    """

    def __init__(self, size=100, mask_token_id=0):
        self.size = size
        self.tokenizer = SimpleNamespace(mask_token_id=mask_token_id)
        self.vocabulary = {}
        self.texts = []

    def vector(self, words, masked=()):
        vector = np.zeros(20)
        vector[0] = 1
        for i, word in enumerate(words):
            if i not in masked:
                vector[self.vocabulary.setdefault(word, len(self.vocabulary) + 1)] += 1
        return vector / np.linalg.norm(vector)

    def chunks(self, words):
        return [
            (i, min(i + self.size, len(words))) for i in range(0, len(words), self.size)
        ]

    def encode(self, texts, masks=None):
        self.texts.extend(texts)
        masks = masks or [set()] * len(texts)
        return np.array([self.vector(t, m) for t, m in zip(texts, masks)]).reshape(
            len(texts), 20
        )


def corpus(texts, lemmas=None):
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    if lemmas is not None:
        corpus.add_layer_meta("lemma", layer_type="seq", base="words", data="string")
    for k, text in enumerate(texts):
        doc = corpus.add_doc(text)
        doc.words = [m.span() for m in re.finditer(r"\w+", text)]
        if lemmas is not None:
            doc.lemma = lemmas[k].split()
    return corpus


def scores(results, doc=None):
    docs = results.terms if doc is None else [results.terms[doc]]
    return {t["term"]: t["score"] for d in docs for t in d["terms"]}


def cosine(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def extract(texts, encoder=None, **params):
    encoder = encoder or FakeEncoder()
    extractor = EmbedRankTermExtractor(encoder=encoder, **params)
    return extractor.extract_terms_strings(texts)


def test_embedrank():
    encoder = FakeEncoder()
    found = scores(extract(["heart failure heart"], encoder, n=2))
    document = encoder.vector(["heart", "failure", "heart"])
    assert found["heart"] == pytest.approx(cosine(encoder.vector(["heart"]), document))
    assert found["heart failure"] == pytest.approx(
        cosine(encoder.vector(["heart", "failure"]), document)
    )
    assert found["heart"] > found["failure"]


def test_long_documents_are_chunked():
    # The document is the mean of its chunks, "heart failure" and "heart"
    encoder = FakeEncoder(size=2)
    found = scores(extract(["heart failure heart"], encoder))
    document = encoder.vector(["heart", "failure"]) + encoder.vector(["heart"])
    assert found["failure"] == pytest.approx(
        cosine(encoder.vector(["failure"]), document)
    )


def test_mderank():
    # Masking both occurrences of "heart" changes both chunks
    encoder = FakeEncoder(size=2)
    found = scores(extract(["heart failure heart"], encoder, variant="mderank"))
    chunks = [["heart", "failure"], ["heart"]]
    document = sum(encoder.vector(chunk) for chunk in chunks)
    masked = encoder.vector(chunks[0], {0}) + encoder.vector(chunks[1], {0})
    assert found["heart"] == pytest.approx(1 - cosine(document, masked))
    masked = encoder.vector(chunks[0], {1}) + encoder.vector(chunks[1])
    assert found["failure"] == pytest.approx(1 - cosine(document, masked))


def test_corpus_aggregation():
    texts = ["heart failure", "heart attack heart rate"]
    by_doc = extract(texts, corpus_aggregation=None)
    first, second = scores(by_doc, 0)["heart"], scores(by_doc, 1)["heart"]
    assert first != second
    assert scores(extract(texts), 0)["heart"] == max(first, second)
    mean = scores(extract(texts, corpus_aggregation="mean"), 1)["heart"]
    assert mean == pytest.approx((first + second) / 2)


def test_centroid():
    encoder = FakeEncoder()
    texts = ["heart failure", "heart attack"]
    found = scores(extract(texts, encoder, corpus_aggregation="centroid"), 1)
    centroid = encoder.vector(["heart", "failure"]) + encoder.vector(
        ["heart", "attack"]
    )
    assert found["attack"] == pytest.approx(
        cosine(encoder.vector(["attack"]), centroid)
    )


def test_documents_as_written_and_candidates_as_lemmas():
    encoder = FakeEncoder()
    tagged = corpus(["Hearts failed"], ["heart fail"])
    extractor = EmbedRankTermExtractor(encoder=encoder, n=1, lemma_layer="lemma")
    results = extractor.extract_terms_teanga(tagged)
    assert encoder.texts == [["Hearts", "failed"], ["fail"], ["heart"]]
    assert results.offsets[0]["offsets"][0] == {"term": "heart", "offset": (0, 6)}


def test_strings_and_teanga_agree():
    texts = ["heart failure patients", "patients with heart failure"]
    for variant in ("embedrank", "mderank"):
        extractor = EmbedRankTermExtractor(encoder=FakeEncoder(), variant=variant)
        assert scores(extractor.extract_terms_teanga(corpus(texts))) == pytest.approx(
            scores(extractor.extract_terms_strings(texts))
        )


def test_invalid_parameters():
    encoder = FakeEncoder()
    with pytest.raises(ValueError, match="variant"):
        EmbedRankTermExtractor(encoder=encoder, variant="keybert")
    with pytest.raises(ValueError, match="corpus_aggregation"):
        EmbedRankTermExtractor(encoder=encoder, corpus_aggregation="min")
    with pytest.raises(ValueError, match="centroid"):
        EmbedRankTermExtractor(
            encoder=encoder, variant="mderank", corpus_aggregation="centroid"
        )
    with pytest.raises(ValueError, match="mask token"):
        EmbedRankTermExtractor(
            encoder=FakeEncoder(mask_token_id=None), variant="mderank"
        )


def test_empty_corpus():
    results = extract(["The", ""])
    assert results.terms == [
        {"doc_id": "doc_0", "terms": []},
        {"doc_id": "doc_1", "terms": []},
    ]


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="requires the 'transformers' extra")
def test_encoder():
    encoder = Encoder(max_length=8)
    words = ["chronic", "heart", "failure", "reduces", "the", "ejection", "fraction"]
    spans = encoder.chunks(words)
    assert spans[0][0] == 0 and spans[-1][1] == len(words)
    assert all(a[1] == b[0] for a, b in pairwise(spans))
    for i, j in spans:
        ids = encoder.tokenizer(words[i:j], is_split_into_words=True)["input_ids"]
        assert len(ids) <= 8

    vectors = encoder.encode([["heart", "failure"], ["cardiac", "failure"], ["sunny"]])
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(1, abs=1e-5)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]
    assert ("heart", "failure") in encoder.cache
    masked = encoder.encode([["heart", "failure"]], [{0}])
    assert masked[0] @ vectors[0] < 1 - 1e-3
    assert encoder.encode([]).shape == (0, vectors.shape[1])


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="requires the 'transformers' extra")
def test_term_extractor():
    texts = ["Chronic heart failure reduces the ejection fraction.", "It was sunny."]
    encoder = Encoder()
    for variant in ("embedrank", "mderank"):
        results = TermExtractor(method="embedrank", variant=variant, encoder=encoder)
        found = scores(results.extract(corpus(texts)))
        assert "heart failure" in found
