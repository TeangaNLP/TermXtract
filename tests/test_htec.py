import hashlib

import pytest
from teanga import Corpus

from termxtract import TermExtractor, datasets
from termxtract.datasets import base, htec

INSTITUTIONS = htec.CATEGORIES[1]

# Two segments of one article, with nested terms, a term ending inside a word (कक्षा
# in कक्षाओं) and one starting inside a word (शोध in संशोधित)
TEXT_8 = "विश्वविद्यालय अनुदान आयोग शिक्षा देता है। छात्र कक्षाओं में पढ़ते हैं।\nसंशोधित नियम?"
TEXT_8A = "प्राथमिक शिक्षा अनिवार्य है।"
DOCUMENTS = [
    (
        "8",
        TEXT_8,
        [
            ("विश्वविद्यालय अनुदान आयोग", INSTITUTIONS),
            ("विश्वविद्यालय", INSTITUTIONS),
            ("शिक्षा", "Ambiguous"),
            ("छात्र", htec.CATEGORIES[3]),
            ("कक्षा", htec.CATEGORIES[2]),
            ("शोध", "Ambiguous"),
        ],
    ),
    (
        "8_a",
        TEXT_8A,
        [("प्राथमिक शिक्षा", htec.CATEGORIES[2]), ("शिक्षा", htec.CATEGORIES[2])],
    ),
]


def write_htec(path):
    """Write a Teanga file laid out like the one on Zenodo: the first occurrence of
    each term is annotated."""
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("source")
    corpus.add_layer_meta("terms", layer_type="span", base="text")
    corpus.add_layer_meta(
        "categories", layer_type="seq", base="terms", data=list(htec.CATEGORIES)
    )
    for source, text, terms in DOCUMENTS:
        doc = corpus.add_doc(text=text, source=source)
        doc.terms = [
            (text.index(term), text.index(term) + len(term)) for term, _ in terms
        ]
        doc.categories = [category for _, category in terms]
    corpus.to_yaml(str(path))


@pytest.fixture
def mini(tmp_path, monkeypatch):
    """Serve the miniature corpus in place of the real one."""
    path = tmp_path / "source" / "htec-2.0.yaml"
    path.parent.mkdir()
    write_htec(path)
    monkeypatch.setenv("TERMXTRACT_DATA", str(tmp_path / "cache"))
    resource = datasets.Resource(
        path.as_uri(), hashlib.sha256(path.read_bytes()).hexdigest(), "htec-2.0.yaml"
    )
    monkeypatch.setitem(
        base._REGISTRY,
        "htec",
        datasets.DatasetInfo(**{**vars(htec.HTEC), "resources": (resource,)}),
    )


def by_source(corpus):
    return {
        corpus.doc_by_id(d).source.text[0]: corpus.doc_by_id(d) for d in corpus.doc_ids
    }


def test_registered():
    info = datasets.info("htec")
    assert info.languages == ("hi",)
    assert info.splits == ()
    assert info.options == ("partial_words", "labels")
    assert "CC BY-SA 4.0" in info.license


def test_load(mini):
    docs = by_source(datasets.load("htec"))
    assert list(docs) == ["8", "8_a"]
    doc = docs["8"]
    # The term starting inside संशोधित is left out; कक्षा, a stem, is kept
    assert doc.terms.text == [
        "विश्वविद्यालय अनुदान आयोग",
        "विश्वविद्यालय",
        "शिक्षा",
        "छात्र",
        "कक्षा",
    ]
    assert doc.sentences.text == [
        "विश्वविद्यालय अनुदान आयोग शिक्षा देता है।",
        "छात्र कक्षाओं में पढ़ते हैं।",
        "संशोधित नियम?",
    ]
    # Vowel signs and viramas stay inside words; the danda is a token
    assert doc.tokens.text[:7] == [
        "विश्वविद्यालय",
        "अनुदान",
        "आयोग",
        "शिक्षा",
        "देता",
        "है",
        "।",
    ]


def test_partial_words(mini):
    doc = by_source(datasets.load("htec", partial_words=True))["8"]
    assert "शोध" in doc.terms.text


def test_labels(mini):
    corpus = datasets.load("htec", labels=[INSTITUTIONS])
    assert datasets.gold_terms(corpus) == {"विश्वविद्यालय अनुदान आयोग", "विश्वविद्यालय"}
    with pytest.raises(ValueError, match="Unknown HTEC labels: Institutions"):
        datasets.load("htec", labels=["Institutions"])


def test_extract_from_htec(mini):
    corpus = datasets.load("htec")
    results = TermExtractor(method="tfidf", n=3, words_layer="tokens").extract(corpus)
    extracted = {t["term"] for doc in results.terms for t in doc["terms"]}
    assert {"विश्वविद्यालय अनुदान आयोग", "प्राथमिक शिक्षा"} <= extracted


def test_real_htec():
    """Check the real corpus, if it is already in the cache."""
    try:
        corpus = datasets.load("htec", partial_words=True, download_if_missing=False)
    except FileNotFoundError:
        pytest.skip("HTEC 2.0 is not cached")
    assert len(corpus.doc_ids) == 97
    assert sum(len(corpus.doc_by_id(d).terms) for d in corpus.doc_ids) == 5452
