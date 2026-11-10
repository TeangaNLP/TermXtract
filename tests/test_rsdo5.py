import hashlib
import zipfile

import pytest

from termxtract import TermExtractor, datasets
from termxtract.datasets import base, rsdo5


def token(form, lemma, msd, upos):
    """A token line of the vertical format, with dependency columns after the tags."""
    return f"{form}\t{lemma}\t{msd}\tUPosTag={upos}\tUPosTag={upos}\ttok1\troot\t-\t-\t-\t-"


def term(domain):
    return f'<term field="Kemija" type="znanstveni članek" domain="{domain}">'


# Two paragraphs, a token glued to the one before it, and terms of both labels
KEMCLA = "\n".join(
    [
        '<text id="rsdo5kemcla" field="Kemija" type="znanstveni članek">',
        '<p id="rsdo5kemcla.1">',
        '<s id="rsdo5kemcla.1.1">',
        term("in-domain"),
        token("Citratno", "citraten", "Agpmsn", "ADJ"),
        token("nitratni", "nitraten", "Agpmsn", "ADJ"),
        token("postopek", "postopek", "Ncmsn", "NOUN"),
        "</term>",
        "<g/>",
        token(":", ":", "Z", "PUNCT"),
        token("sinteza", "sinteza", "Ncfsn", "NOUN"),
        "</s>",
        "</p>",
        '<p id="rsdo5kemcla.2">',
        '<s id="rsdo5kemcla.2.1">',
        term("out-domain"),
        token("Statistična", "statističen", "Agpfsn", "ADJ"),
        token("analiza", "analiza", "Ncfsn", "NOUN"),
        "</term>",
        token("je", "biti", "Va-r3s-n", "AUX"),
        term("in-domain"),
        token("sinteza", "sinteza", "Ncfsn", "NOUN"),
        "</term>",
        "<g/>",
        token(".", ".", "Z", "PUNCT"),
        "</s>",
        "</p>",
        "</text>",
    ]
)

VETCLA = "\n".join(
    [
        '<text id="rsdo5vetcla" field="Veterina" type="znanstveni članek">',
        '<p id="rsdo5vetcla.1">',
        '<s id="rsdo5vetcla.1.1">',
        term("in-domain"),
        token("Breja", "brej", "Agpfsn", "ADJ"),
        token("maternica", "maternica", "Ncfsn", "NOUN"),
        "</term>",
        "</s>",
        "</p>",
        "</text>",
    ]
)


def write_rsdo5(path):
    """Write a zip laid out like RSDO5's archive of vertical files."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rsdo5.vert/00README.txt", "RSDO5")
        archive.writestr("rsdo5.vert/rsdo5vetcla.vert", VETCLA)
        archive.writestr("rsdo5.vert/rsdo5kemcla.vert", KEMCLA)


@pytest.fixture
def mini(tmp_path, monkeypatch):
    """Serve the miniature corpus in place of the real one."""
    path = tmp_path / "source" / "rsdo5.vert.zip"
    path.parent.mkdir()
    write_rsdo5(path)
    monkeypatch.setenv("TERMXTRACT_DATA", str(tmp_path / "cache"))
    resource = datasets.Resource(
        path.as_uri(), hashlib.sha256(path.read_bytes()).hexdigest(), "rsdo5.vert.zip"
    )
    monkeypatch.setitem(
        base._REGISTRY,
        "rsdo5",
        datasets.DatasetInfo(**{**vars(rsdo5.RSDO5), "resources": (resource,)}),
    )


def by_source(corpus):
    return {
        corpus.doc_by_id(d).source.text[0]: corpus.doc_by_id(d) for d in corpus.doc_ids
    }


def test_registered():
    info = datasets.info("rsdo5")
    assert info.languages == ("sl",)
    assert info.splits == ()
    assert info.options == ("domains", "labels")
    assert info.license == "CC BY-SA 4.0"


def test_load(mini):
    docs = by_source(datasets.load("rsdo5"))
    assert list(docs) == ["rsdo5kemcla", "rsdo5vetcla"]
    doc = docs["rsdo5kemcla"]
    # The colon and full stop are glued to the words before them
    assert doc.text == (
        "Citratno nitratni postopek: sinteza\nStatistična analiza je sinteza."
    )
    assert doc.sentences.text == [
        "Citratno nitratni postopek: sinteza",
        "Statistična analiza je sinteza.",
    ]
    assert list(zip(doc.terms.text, doc.categories)) == [
        ("Citratno nitratni postopek", "in-domain"),
        ("Statistična analiza", "out-domain"),
        ("sinteza", "in-domain"),
    ]
    assert doc.tokens.text[:4] == ["Citratno", "nitratni", "postopek", ":"]
    assert doc.lemma[:3] == ["citraten", "nitraten", "postopek"]
    assert doc.upos[:4] == ["ADJ", "ADJ", "NOUN", "PUNCT"]
    assert doc.msd[:3] == ["Agpmsn", "Agpmsn", "Ncmsn"]


def test_domains(mini):
    assert list(by_source(datasets.load("rsdo5", domains="vet"))) == ["rsdo5vetcla"]
    assert list(by_source(datasets.load("rsdo5", domains=["kem", "vet"]))) == [
        "rsdo5kemcla",
        "rsdo5vetcla",
    ]
    with pytest.raises(ValueError, match="Unknown RSDO5 domains: chem"):
        datasets.load("rsdo5", domains="chem")


def test_labels(mini):
    doc = by_source(datasets.load("rsdo5", labels="in-domain"))["rsdo5kemcla"]
    assert doc.terms.text == ["Citratno nitratni postopek", "sinteza"]
    with pytest.raises(ValueError, match="Unknown RSDO5 labels: domain"):
        datasets.load("rsdo5", labels=["domain"])


def test_lemmas_and_tags(mini):
    corpus = datasets.load("rsdo5")
    assert datasets.gold_terms(corpus, lemma_layer="lemma") == {
        "citraten nitraten postopek",
        "statističen analiza",
        "sinteza",
        "brej maternica",
    }
    results = TermExtractor(
        method="tfidf",
        n=3,
        words_layer="tokens",
        lemma_layer="lemma",
        pos_pattern="noun_phrase",
        stopwords=None,
    ).extract(corpus)
    extracted = {t["term"] for doc in results.terms for t in doc["terms"]}
    assert {"citraten nitraten postopek", "brej maternica"} <= extracted
    assert "biti" not in extracted


def test_real_rsdo5():
    """Check the real corpus, if it is already in the cache."""
    try:
        corpus = datasets.load("rsdo5", download_if_missing=False)
    except FileNotFoundError:
        pytest.skip("RSDO5 is not cached")
    assert len(corpus.doc_ids) == 12
    assert sum(len(corpus.doc_by_id(d).terms) for d in corpus.doc_ids) == 37985
    in_domain = datasets.load("rsdo5", labels="in-domain", download_if_missing=False)
    assert sum(len(in_domain.doc_by_id(d).terms) for d in in_domain.doc_ids) == 24282
