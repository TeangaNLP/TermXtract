import re

import pytest
from teanga import Corpus

from termxtract import TermExtractor, evaluate
from termxtract.evaluation import average_precision, normalize_term, precision_at
from termxtract.utils import ATEResults


def results(*docs):
    """ATEResults with one document per argument, each a list of (term, score) pairs."""
    return ATEResults(
        corpus=[],
        terms=[
            {
                "doc_id": f"doc_{i}",
                "terms": [{"term": term, "score": score} for term, score in doc],
            }
            for i, doc in enumerate(docs)
        ],
    )


# Corpus ranking: heart failure (0.9), Patients (0.8), care (0.5), blood  pressure (0.4)
RESULTS = results(
    [("heart failure", 0.9), ("care", 0.5)],
    [("Patients", 0.8), ("heart failure", 0.7), ("blood  pressure", 0.4)],
)
GOLD = {
    "heart failure",
    "blood pressure",
    "heart failure patients",
    "ejection fraction",
}


def test_normalize_term():
    assert normalize_term("  Heart\tFailure  ") == "heart failure"


def test_corpus_scores():
    scores = evaluate(RESULTS, GOLD)
    # 2 of the 4 distinct extracted terms are gold; 2 of the 4 gold terms were found
    assert (scores.true_positives, scores.false_positives, scores.false_negatives) == (
        2,
        2,
        2,
    )
    assert scores.precision == scores.recall == scores.f1 == 0.5


def test_no_partial_credit():
    # "heart failure" does not match the nested gold term "heart failure patients"
    scores = evaluate(results([("heart failure", 1.0)]), {"heart failure patients"})
    assert scores.true_positives == 0
    assert scores.f1 == 0.0


def test_ranking_scores():
    scores = evaluate(RESULTS, GOLD, ks=(1, 2, 10))
    # Gold terms at ranks 1 and 4
    assert scores.average_precision == pytest.approx((1 / 1 + 2 / 4) / 4)
    assert scores.r_precision == 2 / 4
    assert scores.precision_at_k == {1: 1.0, 2: 0.5, 10: 0.2}


def test_ranking_helpers():
    assert average_precision(["a", "b", "c"], {"a", "c"}) == pytest.approx(
        (1 + 2 / 3) / 2
    )
    assert average_precision(["a"], set()) == 0.0
    assert precision_at(["a", "b"], {"b"}, 2) == 0.5
    assert precision_at(["a"], {"a"}, 0) == 0.0


def test_best_score_ranks_terms():
    # "b" scores 0.1 in one document but 0.9 in another, so it ranks first
    scores = evaluate(results([("a", 0.5), ("b", 0.1)], [("b", 0.9)]), {"b"}, ks=(1,))
    assert scores.precision_at_k == {1: 1.0}


def test_top_k():
    scores = evaluate(RESULTS, GOLD, top_k=2)
    # Only "heart failure" and "patients" count as extracted
    assert (scores.true_positives, scores.false_positives) == (1, 1)
    # The ranking scores still use the whole ranking
    assert scores.average_precision == pytest.approx((1 / 1 + 2 / 4) / 4)


def test_term_lengths():
    single = evaluate(RESULTS, GOLD, term_lengths={1})
    # Extracted: patients, care; no single-word gold terms
    assert (single.true_positives, single.false_positives) == (0, 2)
    assert single.false_negatives == 0
    multi = evaluate(RESULTS, GOLD, term_lengths={2})
    # Extracted: heart failure, blood pressure; gold: also ejection fraction
    assert (multi.true_positives, multi.false_positives, multi.false_negatives) == (
        2,
        0,
        1,
    )


PER_DOC = [{"heart failure", "care"}, {"blood pressure", "ejection fraction"}]


def test_micro_and_macro():
    micro = evaluate(RESULTS, PER_DOC, average="micro")
    # Document 1: 2 tp; document 2: 1 tp ("blood pressure"), 2 fp, 1 fn
    assert (micro.true_positives, micro.false_positives, micro.false_negatives) == (
        3,
        2,
        1,
    )
    assert micro.precision == 3 / 5
    assert micro.recall == 3 / 4
    macro = evaluate(RESULTS, PER_DOC, average="macro")
    assert macro.precision == pytest.approx((1 + 1 / 3) / 2)
    assert macro.recall == pytest.approx((1 + 1 / 2) / 2)
    assert macro.f1 == pytest.approx((1 + 0.4) / 2)


def test_micro_top_k_per_document():
    micro = evaluate(RESULTS, PER_DOC, average="micro", top_k=1)
    # Document 1 keeps "heart failure"; document 2 keeps "patients"
    assert (micro.true_positives, micro.false_positives) == (1, 1)


def test_gold_by_document_id():
    by_id = {"doc_1": PER_DOC[1], "doc_0": PER_DOC[0], "unused": {"other"}}
    assert evaluate(RESULTS, by_id, average="micro") == evaluate(
        RESULTS, PER_DOC, average="micro"
    )


def gold_corpus(texts, gold_terms, layer="terms"):
    """A Teanga corpus with a token layer and gold terms annotated as token spans."""
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    corpus.add_layer_meta(layer, layer_type="span", base="tokens")
    for text, terms in zip(texts, gold_terms):
        doc = corpus.add_doc(text)
        tokens = [m.span() for m in re.finditer(r"\w+|[^\w\s]", text)]
        doc.tokens = tokens
        words = [text[start:end].lower() for start, end in tokens]
        spans = []
        for term in terms:
            n = len(term.split())
            spans += [
                (i, i + n)
                for i in range(len(words) - n + 1)
                if " ".join(words[i : i + n]) == term
            ]
        doc[layer] = sorted(set(spans))
    return corpus


TEXTS = [
    "Heart failure patients need care. Heart failure is chronic.",
    "Blood pressure and ejection fraction predict heart failure.",
]


def test_teanga_gold():
    corpus = gold_corpus(TEXTS, [{"heart failure", "care"}, {"ejection fraction"}])
    teanga_results = ATEResults(
        corpus=corpus,
        terms=[
            {"doc_id": doc_id, "terms": doc["terms"]}
            for doc_id, doc in zip(corpus.doc_ids, RESULTS.terms)
        ],
    )
    expected = evaluate(RESULTS, [{"heart failure", "care"}, {"ejection fraction"}])
    assert evaluate(teanga_results, corpus, average="corpus") == expected
    assert evaluate(teanga_results, corpus, average="micro") == evaluate(
        RESULTS, [{"heart failure", "care"}, {"ejection fraction"}], average="micro"
    )


def test_extractor_on_gold_corpus():
    corpus = gold_corpus(
        TEXTS, [{"heart failure", "care"}, {"blood pressure", "ejection fraction"}]
    )
    extracted = TermExtractor(method="tfidf", n=2, words_layer="tokens").extract(corpus)
    scores = evaluate(extracted, corpus, average="micro")
    assert scores.true_positives == 4
    assert scores.false_negatives == 0
    assert scores.recall == 1.0


def test_custom_normalize():
    scores = evaluate(
        results([("covid 19", 1.0)]),
        {"COVID-19"},
        normalize=lambda term: normalize_term(term.replace("-", " ")),
    )
    assert scores.true_positives == 1


def test_errors():
    with pytest.raises(ValueError, match="Unknown average"):
        evaluate(RESULTS, GOLD, average="weighted")
    with pytest.raises(ValueError, match="needs gold terms for each document"):
        evaluate(RESULTS, GOLD, average="micro")
    with pytest.raises(ValueError, match="terms for 1 documents"):
        evaluate(RESULTS, [{"heart failure"}])
    with pytest.raises(ValueError, match="no terms for the documents"):
        evaluate(RESULTS, {"doc_0": {"care"}})
    with pytest.raises(TypeError, match="not a string"):
        evaluate(RESULTS, "heart failure")
    with pytest.raises(ValueError, match="no 'gold' layer"):
        evaluate(RESULTS, gold_corpus(TEXTS, [set(), set()]), gold_layer="gold")


def test_empty():
    scores = evaluate(results([]), set())
    assert (scores.precision, scores.recall, scores.f1) == (0.0, 0.0, 0.0)
    assert scores.average_precision == scores.r_precision == 0.0
