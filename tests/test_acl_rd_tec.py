import hashlib
import zipfile

import pytest

from termxtract import TermExtractor, datasets, evaluate
from termxtract.datasets import acl_rd_tec, base

FILES = "distribution/annoitation_files"

# A miniature ACL RD-TEC 2.0: P99-1 annotated by both annotators, P99-2 by annotator 2
# only, and M92-1 by annotator 1 only but with no sentence markup
ANNOTATED = {
    (1, "P99-1"): (
        "<S>We train a <term class='model'>language model</term>.</S>\n"
        "    <S><term class='lr'> corpora </term> help.</S>"
    ),
    (2, "P99-1"): (
        "<S>We train a <term class='model'>language</term> model.</S>\n"
        "    <S><term class='lr'>corpora</term> help.</S>"
    ),
    (2, "P99-2"): "<S>A <term class='tool'>parser</term> (fast) works.</S>",
    (1, "M92-1"): "The <term class='tool'>toolset</term> works.",
}
TAGGED = {
    "P99-1": [
        ("We", "we", "PRP"),
        ("train", "train", "VBP"),
        ("a", "a", "DT"),
        ("language", "language", "NN"),
        ("model", "model", "NN"),
        ("<g/>", "", ""),
        (".", ".", "."),
        ("corpora", "corpus", "NNS"),
        ("help", "help", "VBP"),
        ("<g/>", "", ""),
        (".", ".", "."),
    ],
    "P99-2": [
        ("A", "a", "DT"),
        ("parser", "parser", "NN"),
        ("(", "(", "-LRB-"),
        ("fast", "fast", "JJ"),
        (")", ")", "-RRB-"),
        ("works", "work", "VBZ"),
        (".", ".", "."),
    ],
    "M92-1": [],
}


def write_distribution(path):
    """Write a zip laid out like the LINDAT/CLARIN distribution."""
    with zipfile.ZipFile(path, "w") as archive:
        vertical = {1: [], 2: []}
        for (number, acl_id), section in ANNOTATED.items():
            archive.writestr(
                f"{FILES}/annotator{number}/99/{acl_id}_abstr.xml",
                f'<?xml version="1.0" encoding="UTF-8"?><Paper acl-id="{acl_id}">'
                f"<Title>A <term class='other'>title</term></Title><Section>"
                f"<SectionTitle>Abstract</SectionTitle>{section}</Section></Paper>",
            )
            vertical[number].append(f'<doc id="{acl_id}" title="A title">')
            vertical[number] += [
                word if word == "<g/>" else f"{word}\t{lemma}\t{tag}"
                for word, lemma, tag in TAGGED[acl_id]
            ]
            vertical[number].append("</doc>")
        for number, lines in vertical.items():
            archive.writestr(
                f"distribution/pos_tagged_vertical_files/"
                f"annotator{number}-acl_rdtec2.vert",
                "\r\n".join(lines),
            )


@pytest.fixture
def mini(tmp_path, monkeypatch):
    """Serve the miniature distribution in place of the real one."""
    path = tmp_path / "source" / "distribution.zip"
    path.parent.mkdir()
    write_distribution(path)
    monkeypatch.setenv("TERMXTRACT_DATA", str(tmp_path / "cache"))
    resource = datasets.Resource(
        path.as_uri(),
        hashlib.sha256(path.read_bytes()).hexdigest(),
        "distribution.zip",
    )
    monkeypatch.setitem(
        base._REGISTRY,
        "acl-rd-tec",
        datasets.DatasetInfo(
            **{**vars(acl_rd_tec.ACL_RD_TEC), "resources": (resource,)}
        ),
    )


def by_source(corpus):
    return {corpus.doc_by_id(d).source: corpus.doc_by_id(d) for d in corpus.doc_ids}


def test_registered():
    info = datasets.info("acl-rd-tec")
    assert info.splits == ()
    assert info.options == ("annotator", "labels")
    assert info.license == "CC BY-NC-SA 4.0"


def test_load(mini):
    docs = by_source(datasets.load("acl-rd-tec"))
    # M92-1 has no sentences, so it is left out
    assert list(docs) == ["P99-1", "P99-2"]
    doc = docs["P99-1"]
    # Annotator 1 is preferred; the title is not included
    assert doc.text == "We train a language model. corpora  help."
    assert doc.sentences.text == ["We train a language model.", "corpora  help."]
    # Whitespace inside a term annotation is trimmed
    assert list(zip(doc.terms.text, doc.categories)) == [
        ("language model", "model"),
        ("corpora", "lr"),
    ]
    assert doc.tokens.text == [w for w, _, _ in TAGGED["P99-1"] if w != "<g/>"]
    assert list(doc.lemma)[6] == "corpus"
    assert list(doc.pos)[:4] == ["PRP", "VBP", "DT", "NN"]
    assert list(docs["P99-2"].upos) == [
        "DET",
        "NOUN",
        "PUNCT",
        "ADJ",
        "PUNCT",
        "VERB",
        "PUNCT",
    ]


def test_annotators(mini):
    second = by_source(datasets.load("acl-rd-tec", annotator=2))
    assert list(second) == ["P99-1", "P99-2"]
    assert second["P99-1"].terms.text == ["language", "corpora"]
    assert list(by_source(datasets.load("acl-rd-tec", annotator=1))) == ["P99-1"]
    with pytest.raises(ValueError, match="Unknown ACL RD-TEC annotator: 3"):
        datasets.load("acl-rd-tec", annotator=3)


def test_labels(mini):
    corpus = datasets.load("acl-rd-tec", labels=["tool", "lr"])
    assert datasets.gold_terms(corpus) == {"corpora", "parser"}
    with pytest.raises(ValueError, match="Unknown ACL RD-TEC labels: method"):
        datasets.load("acl-rd-tec", labels=["method"])


def test_pos_pattern_on_acl_rd_tec(mini):
    corpus = datasets.load("acl-rd-tec")
    results = TermExtractor(
        method="cvalue", n=2, words_layer="tokens", pos_pattern="noun_phrase"
    ).extract(corpus)
    extracted = {t["term"] for doc in results.terms for t in doc["terms"]}
    assert "language model" in extracted
    assert "train" not in extracted
    assert evaluate(results, corpus).true_positives >= 1


def test_real_acl_rd_tec():
    """Check the real dataset, if it is already in the cache."""
    try:
        corpus = datasets.load("acl-rd-tec", annotator=1, download_if_missing=False)
    except FileNotFoundError:
        pytest.skip("The ACL RD-TEC 2.0 is not cached")
    # The numbers of the LREC paper: 189 abstracts less M92-1025, and 2,857 terms
    assert len(corpus.doc_ids) == 188
    assert sum(len(corpus.doc_by_id(d).terms) for d in corpus.doc_ids) == 2857
