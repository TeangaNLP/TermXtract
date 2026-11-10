import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from termxtract import TermExtractor, datasets
from termxtract.datasets import base, htec

SCRIPT = Path(__file__).parent.parent / "scripts" / "htec_to_teanga.py"
spec = importlib.util.spec_from_file_location("htec_to_teanga", SCRIPT)
assert spec is not None and spec.loader is not None
convert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert)

INSTITUTIONS = htec.CATEGORIES[1]


def annotation(text, term, category, occurrence=0):
    """An annotation of the given occurrence of a term in a text."""
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(term, start + 1)
    return {
        "term": term,
        "category": category,
        "start_offset": start,
        "end_offset": start + len(term),
    }


# Two segments of one article, out of order, with a stray space in an ID, the three
# spellings of the institutions category, nested terms, a term ending inside a word
# (कक्षा in कक्षाओं) and one starting inside a word (शोध in संशोधित)
TEXT_8 = "विश्वविद्यालय अनुदान आयोग शिक्षा देता है। छात्र कक्षाओं में पढ़ते हैं।\nसंशोधित नियम?"
TEXT_8A = "प्राथमिक शिक्षा अनिवार्य है।"
RECORDS = {
    "8_a.json": {
        "document_id": "8_a",
        "text": TEXT_8A,
        "annotations": [
            annotation(TEXT_8A, "प्राथमिक शिक्षा", htec.CATEGORIES[2]),
            annotation(TEXT_8A, "शिक्षा", htec.CATEGORIES[2]),
        ],
    },
    "8 .json": {
        "document_id": "8 ",
        "text": TEXT_8,
        "annotations": [
            annotation(TEXT_8, "शिक्षा", "Ambiguous"),
            annotation(
                TEXT_8,
                "विश्वविद्यालय अनुदान आयोग",
                "Educational Institutions and frameworks, Governing bodies, Think "
                "Tanks and Research Institutes",
            ),
            annotation(
                TEXT_8,
                "विश्वविद्यालय",
                "Educational Institutions , Governing bodies, Think Tanks and "
                "Research Institutes",
            ),
            annotation(TEXT_8, "छात्र", htec.CATEGORIES[3]),
            annotation(TEXT_8, "कक्षा", htec.CATEGORIES[2]),
            annotation(TEXT_8, "शोध", "Ambiguous"),
        ],
    },
    "output_file.json": {
        "document_id": "output_file",
        "text": "शिक्षा\nछात्र",
        "annotations": [annotation("शिक्षा\nछात्र", "शिक्षा", "Ambiguous")],
    },
}


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "output_json.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, record in RECORDS.items():
            zf.writestr(f"output_json/{name}", json.dumps(record, ensure_ascii=False))
    return path


@pytest.fixture
def mini(tmp_path, archive, monkeypatch):
    """Serve the converted miniature corpus in place of the real one."""
    path = tmp_path / "source" / "htec-2.0.yaml"
    path.parent.mkdir()
    convert.convert(str(archive)).to_yaml(str(path))
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
    assert convert.CATEGORIES == htec.CATEGORIES


def test_convert(archive):
    corpus = convert.convert(str(archive))
    docs = [corpus.doc_by_id(d) for d in corpus.doc_ids]
    # In order of ID, without the stray space and the list of terms
    assert [doc["source"].text[0] for doc in docs] == ["8", "8_a"]
    doc = docs[0]
    # Every annotation, sorted, with the categories normalised
    assert list(zip(doc["terms"].text, doc["categories"])) == [
        ("विश्वविद्यालय अनुदान आयोग", INSTITUTIONS),
        ("विश्वविद्यालय", INSTITUTIONS),
        ("शिक्षा", "Ambiguous"),
        ("छात्र", htec.CATEGORIES[3]),
        ("कक्षा", htec.CATEGORIES[2]),
        ("शोध", "Ambiguous"),
    ]


def test_convert_checks_offsets(tmp_path):
    record = {**RECORDS["8_a.json"]}
    record["annotations"] = [{**record["annotations"][0], "start_offset": 1}]
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("output_json/8_a.json", json.dumps(record, ensure_ascii=False))
    with pytest.raises(ValueError, match="is not at"):
        convert.convert(str(path))


def test_document_order():
    names = ["99_aa", "100", "99_b", "99", "99_z", "8"]
    assert sorted(names, key=convert.document_order) == [
        "8",
        "99",
        "99_b",
        "99_z",
        "99_aa",
        "100",
    ]


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
