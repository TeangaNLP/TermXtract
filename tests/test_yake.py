import math
import re
import statistics

import pytest
from teanga import Corpus

from termxtract import TermExtractor
from termxtract.unsupervised.yake import YAKETermExtractor, is_near_duplicate


def corpus(texts, pattern=r"\w+"):
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    for text in texts:
        doc = corpus.add_doc(text)
        doc.words = [m.span() for m in re.finditer(pattern, text)]
    return corpus


def scores(results, doc=None):
    docs = results.terms if doc is None else [results.terms[doc]]
    return {t["term"]: t["score"] for d in docs for t in d["terms"]}


def yake(**params):
    return YAKETermExtractor(**{"dedup_threshold": None, **params})


def test_word_score():
    # "cat" occurs twice, in sentences 0 and 1, never capitalised, with nothing on its
    # left and two different words on its right. The frequencies are 2, 1 and 1.
    found = scores(yake(n=1).extract_terms_strings(["cat dog. cat bird."]))
    spread = statistics.fmean([2, 1, 1]) + statistics.pstdev([2, 1, 1])
    case = 0
    position = math.log(math.log(3 + 0.5))
    relatedness = 1 + (0 + 2 / 2) * 2 / 2
    word = relatedness * position / (case + (2 / spread + 2 / 2) / relatedness)
    assert found["cat"] == pytest.approx(-word / ((1 + word) * 2))


def test_capitals_score_better():
    # "Kaggle" starts with a capital letter but not a sentence
    lower = scores(yake(n=1).extract_terms_strings(["the kaggle site. a kaggle site."]))
    upper = scores(yake(n=1).extract_terms_strings(["the Kaggle site. a Kaggle site."]))
    assert upper["kaggle"] > lower["kaggle"]
    assert upper["site"] == pytest.approx(lower["site"])


def test_punctuation_splits_candidates():
    found = scores(yake(n=2).extract_terms_strings(["heart, failure. heart failure"]))
    assert set(found) == {"heart", "failure", "heart failure"}
    found = scores(yake(n=2).extract_terms_strings(["heart, failure. heart, failure"]))
    assert set(found) == {"heart", "failure"}


def test_danda_ends_sentences():
    # The Devanagari danda ends a sentence, as a full stop does
    danda = scores(yake(n=1).extract_terms_strings(["cat dog। cat bird।"]))
    stop = scores(yake(n=1).extract_terms_strings(["cat dog. cat bird."]))
    assert danda == stop


def test_stopwords_inside_candidates():
    texts = ["The degree of freedom is low. A degree of freedom matters."]
    found = scores(yake().extract_terms_strings(texts))
    assert "degree of freedom" in found
    assert not any(term.startswith(("the ", "of ")) for term in found)


def test_near_duplicates_are_dropped():
    texts = ["Heart failure patients. Heart failures in patients."]
    every = scores(yake(n=2).extract_terms_strings(texts))
    kept = scores(YAKETermExtractor(n=2).extract_terms_strings(texts))
    assert {"heart failure", "heart failures"} <= set(every)
    assert len({"heart failure", "heart failures"} & set(kept)) == 1
    assert set(kept) < set(every)


def test_is_near_duplicate():
    # One edit in 14 characters is a similarity of 13/14, and in 10 of 0.9
    assert is_near_duplicate("heart failure", "heart failures", 0.9)
    assert not is_near_duplicate("cardiology", "cardiologx", 0.9)
    assert not is_near_duplicate("heart failure", "heart", 0.9)
    assert is_near_duplicate("abc", "abd", 0.5)
    assert not is_near_duplicate("abc", "xyz", 0.5)


def test_corpus_aggregation():
    texts = ["heart failure patients. heart failure.", "heart and lungs. lungs."]
    by_doc = yake(corpus_aggregation=None).extract_terms_strings(texts)
    assert scores(by_doc, 0)["heart"] != scores(by_doc, 1)["heart"]
    best = scores(yake().extract_terms_strings(texts), 1)["heart"]
    assert best == max(scores(by_doc, 0)["heart"], scores(by_doc, 1)["heart"])
    mean = scores(yake(corpus_aggregation="mean").extract_terms_strings(texts), 0)
    assert mean["heart"] == pytest.approx(
        (scores(by_doc, 0)["heart"] + scores(by_doc, 1)["heart"]) / 2
    )
    with pytest.raises(ValueError, match="corpus_aggregation"):
        YAKETermExtractor(corpus_aggregation="min")


def test_offsets():
    texts = ["Heart failure patients. Patients with Heart failure."]
    results = TermExtractor(method="yake").extract(corpus(texts))
    for occurrence in results.offsets[0]["offsets"]:
        start, end = occurrence["offset"]
        assert texts[0][start:end].lower() == occurrence["term"]
    assert "heart failure" in scores(results)


def test_strings_and_teanga_agree():
    texts = [
        "Heart failure patients. Patients with Heart failure!",
        "The heart, again.",
    ]
    extractor = YAKETermExtractor()
    expected = scores(extractor.extract_terms_strings(texts))
    assert scores(extractor.extract_terms_teanga(corpus(texts))) == expected
    # Punctuation as words splits sentences and candidates as it does between words
    tokens = corpus(texts, r"\w+|[^\w\s]")
    assert scores(extractor.extract_terms_teanga(tokens)) == expected


def test_empty_corpus():
    results = YAKETermExtractor().extract_terms_strings(["The", ""])
    assert results.terms == [
        {"doc_id": "doc_0", "terms": []},
        {"doc_id": "doc_1", "terms": []},
    ]
