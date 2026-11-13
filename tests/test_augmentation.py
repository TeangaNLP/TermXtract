import re

import numpy as np
import pytest
from teanga import Corpus

from termxtract.augmentation import (
    PROMPTS,
    augment,
    make_prompt,
    outermost,
    parse_sentences,
    replace_terms,
    sample,
    tag,
    untag,
)

TEXTS = [
    "Heart failure is treated with diuretics.",
    "Beta blockers reduce mortality in heart failure.",
    "The weather was fine.",
]
TERMS = [["Heart failure", "diuretics"], ["Beta blockers", "heart failure"], []]


class FakeClient:
    """Answers every prompt with the same lines, and keeps the prompts."""

    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.answer


def corpus(texts=TEXTS, terms=TERMS):
    """A Teanga corpus with tokens and gold terms by their text."""
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    corpus.add_layer_meta("terms", layer_type="span", base="text")
    for text, doc_terms in zip(texts, terms):
        doc = corpus.add_doc(text)
        doc.tokens = [m.span() for m in re.finditer(r"\w+|[^\w\s]", text)]
        doc.terms = sorted(
            m.span() for term in doc_terms for m in re.finditer(re.escape(term), text)
        )
    return corpus


def terms_of(corpus):
    """The terms of each document of a Teanga corpus, as written."""
    return {
        corpus.doc_by_id(d).text: [
            corpus.doc_by_id(d).text[a:b] for a, b in corpus.doc_by_id(d).terms
        ]
        for d in corpus.doc_ids
    }


def test_tag_untag():
    text = "Heart failure is treated with diuretics."
    spans = [(0, 13), (30, 39)]
    tagged = tag(text, spans)
    assert tagged == "<tag>Heart failure</tag> is treated with <tag>diuretics</tag>."
    assert untag(tagged) == (text, spans)
    assert untag("the <tag> heart </tag> beats") == ("the heart beats", [(4, 9)])
    # Unbalanced, empty or inside a word
    assert untag("the <tag>heart beats") is None
    assert untag("the <tag></tag> beats") is None
    assert untag("the <tag>hea</tag>rt beats") is None


def test_outermost():
    assert outermost([(0, 13), (6, 13), (20, 25), (22, 30)]) == [(0, 13), (22, 30)]


def test_sample():
    gold = [
        ("a b", [(0, 1), (2, 3)]),
        ("a", [(0, 1)]),
        ("c", [(0, 1)]),
        ("a b c", [(0, 1), (2, 3), (4, 5)]),
    ]
    rng = np.random.default_rng(0)
    # The sentence with most new terms, then those adding none, by most terms
    assert sample(gold, 2, "diversity", rng) == [gold[0], gold[3]]
    assert sample(gold, 10, "diversity", rng) == gold
    assert len(sample(gold, 2, "random", rng)) == 2


def test_make_prompt():
    seeds = [(TEXTS[0], [(0, 13), (30, 39)])]
    prompt = make_prompt("context", seeds, "heart failure", 5)
    assert prompt.startswith(
        "Given a list of terms that occur in the sentences below:\n"
        "Heart failure, diuretics\nSentences are:\n"
        "<tag>Heart failure</tag> is treated with <tag>diuretics</tag>.\n"
    )
    assert "generate 5 sentences" in prompt
    for method in ("term", "combined"):
        prompt = make_prompt(method, seeds, "heart failure", 5)
        assert "related to heart failure" in prompt
        assert prompt.endswith(
            ":\n<tag>Heart failure</tag> is treated with <tag>diuretics</tag>."
        )


def test_parse_sentences():
    answer = (
        "Here are two sentences:\n"
        "1. <tag>Cardiomyopathy</tag> is treated with <tag>ACE inhibitors</tag>.\n"
        '- "the <tag>heart</tag> failed"\n'
        "an untagged sentence about heart failure.\n"
        "a <tag>broken tag sentence.\n"
    )
    assert parse_sentences(answer, "term") == [
        ("Cardiomyopathy is treated with ACE inhibitors.", [(0, 14), (31, 45)]),
        ("the heart failed", [(4, 9)]),
    ]
    # "context" finds the terms of the seeds, the longest where they overlap
    assert parse_sentences(answer, "context", ["heart failure", "heart"]) == [
        ("the heart failed", [(4, 9)]),
        ("an untagged sentence about heart failure.", [(27, 40)]),
    ]


def test_augment_teanga():
    client = FakeClient(
        "<tag>Cardiomyopathy</tag> is treated with <tag>ACE inhibitors</tag>.\n"
        "<tag>Heart failure</tag> is treated with <tag>diuretics</tag>.\n"
        "<tag>Cardiomyopathy</tag> is treated with <tag>ACE inhibitors</tag>."
    )
    out = augment(corpus(), method="term", n=5, client=client, domain="medicine")
    found = terms_of(out)
    # The gold sentences, then the new one: a gold sentence and a repeat are dropped
    assert list(found) == [*TEXTS, "Cardiomyopathy is treated with ACE inhibitors."]
    assert found[TEXTS[1]] == ["Beta blockers", "heart failure"]
    assert found["Cardiomyopathy is treated with ACE inhibitors."] == [
        "Cardiomyopathy",
        "ACE inhibitors",
    ]
    new = out.doc_by_id(list(out.doc_ids)[-1])
    assert [new.text[a:b] for a, b in new.tokens][:3] == [
        "Cardiomyopathy",
        "is",
        "treated",
    ]
    assert [tuple(s) for s in new.sentences] == [(0, len(new.text))]
    # Asked three times as often as needed, then given up
    assert len(client.prompts) == 3
    assert all(p.startswith(PROMPTS["term"][:40]) for p in client.prompts)
    # The sentence without terms is not a seed
    assert "weather" not in client.prompts[0]


def test_augment_strings():
    client = FakeClient("the <tag>left ventricle</tag> is dilated.")
    labels = {"heart failure": 1, "diuretics": 1, "beta blockers": 1, "weather": 0}
    texts, new_labels = augment(
        TEXTS, labels, method="combined", n=1, client=client, domain="medicine"
    )
    assert texts == [*TEXTS, "the left ventricle is dilated."]
    assert new_labels == {**labels, "left ventricle": 1}
    assert len(client.prompts) == 1


def test_augment_per_prompt():
    client = FakeClient("\n".join(f"the <tag>term{i}</tag> works." for i in range(20)))
    out = augment(corpus(), method="term", n=7, client=client, domain="x", per_prompt=3)
    assert len(out.doc_ids) == len(TEXTS) + 3
    assert "generate 3 diverse sentences" in client.prompts[0]
    # The same answer has no new sentences, so it is asked 3 * ceil(7 / 3) times
    assert len(client.prompts) == 9


def test_replace_terms():
    seeds = [(TEXTS[0], [(0, 13), (30, 39)]), ("Wind is fine.", [(0, 4)])]
    lexicon = {"heart failure": ["cardiomyopathy", "myocarditis"]}
    found = replace_terms(seeds, lexicon, 5, np.random.default_rng(0), set())
    # Each replacement once; diuretics, without replacements, stays
    assert sorted(found) == [
        ("cardiomyopathy is treated with diuretics.", [(0, 14), (31, 40)]),
        ("myocarditis is treated with diuretics.", [(0, 11), (28, 37)]),
    ]


def test_augment_wiki():
    lexicon = {"diuretics": ["loop diuretics"]}
    out = augment(corpus(), method="wiki", n=5, lexicon=lexicon, random_state=0)
    assert terms_of(out)["Heart failure is treated with loop diuretics."] == [
        "Heart failure",
        "loop diuretics",
    ]
    assert len(out.doc_ids) == len(TEXTS) + 1


def test_augment_invalid():
    client = FakeClient("")
    with pytest.raises(ValueError, match="Unknown method"):
        augment(corpus(), method="other", client=client, domain="x")
    with pytest.raises(ValueError, match="Unknown sampling"):
        augment(corpus(), client=client, domain="x", sampling="other")
    with pytest.raises(ValueError, match="client and a domain"):
        augment(corpus(), client=client)
    with pytest.raises(ValueError, match="lexicon"):
        augment(corpus(), method="wiki")
    with pytest.raises(ValueError, match="labels"):
        augment(TEXTS, client=client, domain="x")
