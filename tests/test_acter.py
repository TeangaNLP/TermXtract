import hashlib
import zipfile

import pytest

from termxtract import TermExtractor, datasets, evaluate
from termxtract.datasets import acter, base

ROOT = f"ACTER-{acter._COMMIT}"

# A miniature ACTER: (domain, file, text, tokenised sentences, IOB labels with named
# entities, IOB labels without them)
DOCS = [
    (
        "htfl",
        "htfl_en_001",
        'Heart failure patients at "Mayo"-clinics.\nHeart failure is chronic.',
        ["Heart failure patients at Mayo-clinics .", "Heart failure is chronic ."],
        list("BIIOBOBIOBO"),
        list("BIIOOOBIOBO"),
    ),
    (
        "corp",
        "corp_en_01",
        "Bribery is corruption.",
        ["Bribery is corruption ."],
        list("BOBO"),
        list("BOBO"),
    ),
]
TERMS = {
    "htfl": [
        ("heart failure", "Specific_Term"),
        ("heart failure patients", "Specific_Term"),
        ("patients", "Common_Term"),
        ("chronic", "OOD_Term"),
    ],
    "corp": [("bribery", "Specific_Term"), ("corruption", "Common_Term")],
    # Domains with no texts in the miniature
    "equi": [],
    "wind": [],
}
NES = {"htfl": [("mayo-clinics", "Named_Entity")], "corp": [], "equi": [], "wind": []}


def write_acter(path):
    """Write a zip laid out like the ACTER repository archive."""
    with zipfile.ZipFile(path, "w") as archive:
        for domain, stem, text, sentences, iob_nes, iob in DOCS:
            root = f"{ROOT}/en/{domain}/annotated/"
            archive.writestr(f"{root}texts/{stem}.txt", text)
            archive.writestr(
                f"{root}texts_tokenised/{stem}.txt",
                "".join(sentence + " \n" for sentence in sentences),
            )
            tokens = [token for sentence in sentences for token in sentence.split()]
            for suffix, directory, labels in [
                ("_nes", "with", iob_nes),
                ("", "without", iob),
            ]:
                archive.writestr(
                    f"{root}annotations/sequential_annotations/iob_annotations/"
                    f"{directory}_named_entities/{stem}_seq_terms{suffix}.tsv",
                    "".join(f"{t}\t{label}\n" for t, label in zip(tokens, labels)),
                )
        for domain, terms in TERMS.items():
            lists = f"{ROOT}/en/{domain}/annotated/annotations/unique_annotation_lists/"
            for suffix, listed in [("", terms), ("_nes", terms + NES[domain])]:
                archive.writestr(
                    f"{lists}{domain}_en_tokenised_terms{suffix}.tsv",
                    "".join(f"{term}\t{label}\n" for term, label in listed),
                )


@pytest.fixture
def mini(tmp_path, monkeypatch):
    """Serve the miniature ACTER in place of the real one."""
    path = tmp_path / "source" / "acter.zip"
    path.parent.mkdir()
    write_acter(path)
    monkeypatch.setenv("TERMXTRACT_DATA", str(tmp_path / "cache"))
    resource = datasets.Resource(
        path.as_uri(), hashlib.sha256(path.read_bytes()).hexdigest(), "ACTER-1.5.zip"
    )
    monkeypatch.setitem(
        base._REGISTRY,
        "acter",
        datasets.DatasetInfo(**{**vars(acter.ACTER), "resources": (resource,)}),
    )


def by_source(corpus):
    return {corpus.doc_by_id(d).source: corpus.doc_by_id(d) for d in corpus.doc_ids}


def test_registered():
    info = datasets.info("acter")
    assert info.splits == ("train", "test")
    assert info.options == (
        "language",
        "domains",
        "include_named_entities",
        "labels",
    )
    assert info.license == "CC BY-NC-SA 4.0"


def test_load(mini):
    corpus = datasets.load("acter")
    docs = by_source(corpus)
    assert list(docs) == ["corp_en_01", "htfl_en_001"]
    doc = docs["htfl_en_001"]
    # The tokeniser dropped the quotes, but the tokens still align with the text
    assert doc.tokens.text[4] == 'Mayo"-clinics'
    assert doc.sentences.text == [
        'Heart failure patients at "Mayo"-clinics.',
        "Heart failure is chronic.",
    ]
    # Every occurrence of a listed term, nested ones included, within sentences
    assert list(zip(doc.terms.text, doc.categories)) == [
        ("Heart failure", "Specific_Term"),
        ("Heart failure patients", "Specific_Term"),
        ("patients", "Common_Term"),
        ("Heart failure", "Specific_Term"),
        ("chronic", "OOD_Term"),
    ]
    assert list(doc.iob) == DOCS[0][5]


def test_named_entities(mini):
    doc = by_source(datasets.load("acter", include_named_entities=True))["htfl_en_001"]
    assert ('Mayo"-clinics', "Named_Entity") in zip(doc.terms.text, doc.categories)
    assert list(doc.iob) == DOCS[0][4]


def test_splits_and_domains(mini):
    assert list(by_source(datasets.load("acter", split="test"))) == ["htfl_en_001"]
    assert list(by_source(datasets.load("acter", split="train"))) == ["corp_en_01"]
    assert list(by_source(datasets.load("acter", domains="corp"))) == ["corp_en_01"]
    with pytest.raises(ValueError, match="the 'test' split has only htfl"):
        datasets.load("acter", split="test", domains=["corp"])
    with pytest.raises(ValueError, match="Unknown ACTER domains: heart"):
        datasets.load("acter", domains=["heart"])
    with pytest.raises(ValueError, match="Unknown ACTER language"):
        datasets.load("acter", language="de")


def test_labels(mini):
    corpus = datasets.load("acter", labels=["Specific_Term"])
    assert datasets.gold_terms(corpus) == {
        "heart failure",
        "heart failure patients",
        "bribery",
    }
    with pytest.raises(ValueError, match="include_named_entities=True"):
        datasets.load("acter", labels=["Named_Entity"])


def test_evaluate_on_acter(mini):
    corpus = datasets.load("acter", split="test")
    results = TermExtractor(method="cvalue", n=3, words_layer="tokens").extract(corpus)
    scores = evaluate(results, corpus)
    assert scores.true_positives + scores.false_negatives == 4


def test_real_acter():
    """Check the real dataset, if it is already in the cache."""
    try:
        corpus = datasets.load("acter", split="test", download_if_missing=False)
    except FileNotFoundError:
        pytest.skip("ACTER is not cached")
    assert len(corpus.doc_ids) == 190
    # ACTER's README gives 2,339 tokenised terms for English heart failure; a few
    # more differ only in the quotes inside them
    assert 2330 <= len(datasets.gold_terms(corpus)) <= 2345
