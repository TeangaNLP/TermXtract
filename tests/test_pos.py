import re

import pytest
from teanga import Corpus

from termxtract import SupervisedTermExtractor, TermExtractor
from termxtract.pos import compile_pos_pattern

TAGGED = [
    (
        "The number of degrees of freedom of a system is small.",
        "DET NOUN ADP NOUN ADP NOUN ADP DET NOUN AUX ADJ PUNCT",
    ),
    (
        "Each degree of freedom of the system adds energy.",
        "DET NOUN ADP NOUN ADP DET NOUN VERB NOUN PUNCT",
    ),
    (
        "A degree of freedom is a parameter of the system.",
        "DET NOUN ADP NOUN AUX DET NOUN ADP DET NOUN PUNCT",
    ),
]


def tagged_corpus(words_layer="tokens", pos_layer="upos", tagged=TAGGED):
    """A corpus laid out like Teanga's CoNLL-U reader (or spaCy, with pos_layer="pos")."""
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta(words_layer, layer_type="span", base="text")
    corpus.add_layer_meta(pos_layer, layer_type="seq", base=words_layer, data="string")
    for text, tags in tagged:
        doc = corpus.add_doc(text)
        doc[words_layer] = [m.span() for m in re.finditer(r"\w+|[^\w\s]", text)]
        doc[pos_layer] = tags.split()
    return corpus


def all_terms(results):
    return {t["term"] for doc in results.terms for t in doc["terms"]}


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ("NOUN", True),
        ("PROPN", True),
        ("ADJ NOUN", True),
        ("NOUN NOUN", True),
        ("NOUN ADP NOUN", True),
        ("NOUN ADP DET ADJ NOUN", True),
        ("ADJ", False),
        ("DET NOUN", False),
        ("NOUN VERB", False),
        ("NOUN ADP", False),
    ],
)
def test_noun_phrase_pattern(tags, expected):
    sequence = "".join(f"<{tag}>" for tag in tags.split())
    assert bool(compile_pos_pattern("noun_phrase").fullmatch(sequence)) is expected


def test_custom_pattern_syntax():
    pattern = compile_pos_pattern("<ADJ>? <N.*>+")
    assert pattern.fullmatch("<ADJ><NOUN><NUM>")
    assert not pattern.fullmatch("<ADJ><VERB>")
    # "." matches within one tag only
    assert not compile_pos_pattern("<N.*>").fullmatch("<NOUN><VERB>")
    with pytest.raises(ValueError, match="Invalid part-of-speech pattern"):
        compile_pos_pattern("<NOUN>(")


def test_filters_terms_and_offsets():
    corpus = tagged_corpus()
    filtered = TermExtractor(method="tfidf", n=3, pos_pattern="noun_phrase").extract(
        corpus
    )
    unfiltered = TermExtractor(method="tfidf", n=3).extract(corpus)
    terms = all_terms(filtered)
    assert {"degree of freedom", "number of degrees", "system", "energy"} <= terms
    assert not {"system adds", "adds energy", "small", "adds"} & terms
    # Scores are unchanged, and only the remaining terms have offsets
    for doc, all_doc in zip(filtered.terms, unfiltered.terms):
        assert doc["terms"] == [t for t in all_doc["terms"] if t["term"] in terms]
    assert all(o["term"] in terms for doc in filtered.offsets for o in doc["offsets"])
    assert any(o["term"] == "degree of freedom" for o in filtered.offsets[1]["offsets"])


def test_term_kept_if_any_occurrence_matches():
    tagged = [("light shines", "NOUN VERB"), ("light shines", "ADJ NOUN")]
    terms = all_terms(
        TermExtractor(method="tfidf", n=2, pos_pattern="noun_phrase").extract(
            tagged_corpus(tagged=tagged)
        )
    )
    assert terms == {"light", "light shines", "shines"}


def test_spacy_layout_and_explicit_layers():
    corpus = tagged_corpus(words_layer="tokens", pos_layer="pos")
    results = TermExtractor(method="cvalue", n=3, pos_pattern="noun_phrase").extract(
        corpus
    )
    assert "degree of freedom" in all_terms(results)
    explicit = TermExtractor(
        method="cvalue",
        n=3,
        pos_pattern="noun_phrase",
        words_layer="tokens",
        pos_layer="pos",
    ).extract(corpus)
    assert explicit.terms == results.terms


def test_words_layer_preferred_over_tokens():
    corpus = tagged_corpus(words_layer="words")
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    for doc_id in corpus.doc_ids:
        corpus.doc_by_id(doc_id).tokens = [(0, 1)]
    terms = all_terms(TermExtractor(method="tfidf", stopwords=None).extract(corpus))
    assert "freedom" in terms


def test_pos_layer_must_be_based_on_words_layer():
    # Tags on "tokens", but words read from "words"
    corpus = tagged_corpus(words_layer="tokens")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    for doc_id in corpus.doc_ids:
        doc = corpus.doc_by_id(doc_id)
        doc.words = [m.span() for m in re.finditer(r"\w+", doc.text)]
    with pytest.raises(ValueError, match="Pass words_layer='tokens'"):
        TermExtractor(method="tfidf", pos_pattern="noun_phrase").extract(corpus)


def test_missing_layers():
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_doc("No words here.")
    with pytest.raises(ValueError, match="Pass words_layer=..."):
        TermExtractor(method="tfidf").extract(corpus)
    with pytest.raises(ValueError, match="Pass pos_layer=..."):
        TermExtractor(method="tfidf", pos_pattern="noun_phrase").extract(
            tagged_corpus(pos_layer="xpos")
        )


@pytest.mark.parametrize("wrapper", [TermExtractor, SupervisedTermExtractor])
def test_plain_text_rejected(wrapper):
    extractor = wrapper(method="tfidf" if wrapper is TermExtractor else "roger")
    extractor.pos_pattern = compile_pos_pattern("noun_phrase")
    args = ([text for text, _ in TAGGED],)
    if wrapper is SupervisedTermExtractor:
        args += ({"degree of freedom": 1, "of the system": 0},)
    with pytest.raises(ValueError, match="only works with a Teanga corpus"):
        extractor.extract(*args)


def test_supervised():
    labels = {"degree of freedom": 1, "number of degrees": 1, "of the system": 0}
    results = SupervisedTermExtractor(
        method="roger",
        n=3,
        generations=5,
        runs=2,
        random_state=0,
        pos_pattern="noun_phrase",
    ).extract(tagged_corpus(), labels)
    terms = all_terms(results)
    assert "degree of freedom" in terms
    assert "system adds" not in terms
