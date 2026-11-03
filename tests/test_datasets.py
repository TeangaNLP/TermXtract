import hashlib
import json
import re
import zipfile
from pathlib import Path

import pytest
from teanga import Corpus

from termxtract import TermExtractor, datasets
from termxtract.datasets import base

DOCS = {
    "train": [
        ("Heart failure patients need care.", ["heart failure", "care"]),
        ("Blood pressure predicts heart failure.", ["blood pressure", "heart failure"]),
    ],
    "test": [("Ejection fraction is low in heart failure.", ["ejection fraction"])],
}


def build_corpus(docs):
    """A Teanga corpus laid out as the dataset loaders lay it out."""
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    corpus.add_layer_meta(datasets.GOLD_LAYER, layer_type="span", base="text")
    for text, terms in docs:
        doc = corpus.add_doc(text)
        doc.tokens = [m.span() for m in re.finditer(r"\w+|[^\w\s]", text)]
        doc[datasets.GOLD_LAYER] = [
            m.span() for term in terms for m in re.finditer(term, text, re.IGNORECASE)
        ]
    return corpus


def toy_loader(path, *, split=None, lowercase=False):
    """Load the toy dataset: a zip with one JSON file per split."""
    with zipfile.ZipFile(path) as archive:
        splits = [split] if split else ["train", "test"]
        docs = [
            (text.lower() if lowercase else text, terms)
            for name in splits
            for text, terms in json.loads(archive.read(f"{name}.json"))
        ]
    return build_corpus(docs)


@pytest.fixture
def source(tmp_path):
    """A zip file served from a file:// URL, and its checksum."""
    path = tmp_path / "source" / "toy.zip"
    path.parent.mkdir()
    with zipfile.ZipFile(path, "w") as archive:
        for name, docs in DOCS.items():
            archive.writestr(f"{name}.json", json.dumps(docs))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def toy(source, tmp_path, monkeypatch):
    """Register a toy dataset, cached under a temporary TERMXTRACT_DATA."""
    path, sha256 = source
    monkeypatch.setenv("TERMXTRACT_DATA", str(tmp_path / "cache"))
    monkeypatch.setattr(base, "_REGISTRY", {})
    return datasets.register(
        datasets.DatasetInfo(
            name="toy",
            description="A toy dataset.",
            languages=("en",),
            domains=("heart failure",),
            license="CC0-1.0",
            citation="Nobody (2025).",
            homepage="https://example.org/toy",
            resources=(datasets.Resource(path.as_uri(), sha256, "toy.zip"),),
            loader=toy_loader,
            splits=("train", "test"),
        )
    )


def test_load(toy, tmp_path):
    corpus = datasets.load("toy")
    assert len(corpus.doc_ids) == 3
    assert (tmp_path / "cache" / "toy" / "toy.zip").exists()
    assert datasets.gold_terms(corpus) == {
        "heart failure",
        "care",
        "blood pressure",
        "ejection fraction",
    }


def test_split_and_options(toy):
    test = datasets.load("toy", split="test", lowercase=True)
    assert [test.doc_by_id(d).text for d in test.doc_ids] == [
        "ejection fraction is low in heart failure."
    ]
    assert datasets.gold_terms(test, by_document=True) == {
        next(iter(test.doc_ids)): {"ejection fraction"}
    }


def test_cached_after_first_download(toy, source):
    datasets.load("toy")
    source[0].unlink()  # The source is gone, so a second download would fail
    assert len(datasets.load("toy").doc_ids) == 3
    with pytest.raises(OSError):
        datasets.download("toy", force=True)


def test_checksum_mismatch(toy, tmp_path, monkeypatch):
    bad = datasets.Resource(toy.resources[0].url, "0" * 64, "toy.zip")
    monkeypatch.setitem(
        base._REGISTRY,
        "toy",
        datasets.DatasetInfo(**{**vars(toy), "resources": (bad,)}),
    )
    with pytest.raises(ValueError, match="does not match its checksum"):
        datasets.load("toy")
    # Neither the file nor a partial download is left behind
    assert list((tmp_path / "cache" / "toy").iterdir()) == []


def test_download_if_missing(toy):
    with pytest.raises(FileNotFoundError, match="not cached"):
        datasets.load("toy", download_if_missing=False)
    datasets.download("toy")
    assert len(datasets.load("toy", download_if_missing=False).doc_ids) == 3


def test_home_argument(toy, tmp_path):
    datasets.load("toy", home=tmp_path / "elsewhere")
    assert (tmp_path / "elsewhere" / "toy" / "toy.zip").exists()
    assert not (tmp_path / "cache").exists()


def test_data_home(monkeypatch):
    monkeypatch.delenv("TERMXTRACT_DATA", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
    assert datasets.data_home() == Path("/xdg/termxtract")
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert datasets.data_home() == Path.home() / ".cache" / "termxtract"
    monkeypatch.setenv("TERMXTRACT_DATA", "/data")
    assert datasets.data_home() == Path("/data")
    assert datasets.data_home("/given") == Path("/given")


def test_registry(toy):
    assert datasets.list_datasets() == [toy]
    assert datasets.info("toy") is toy
    assert toy.options == ("lowercase",)
    with pytest.raises(ValueError, match="already registered"):
        datasets.register(toy)
    with pytest.raises(
        ValueError, match="Unknown dataset: 'nope'. Available datasets: 'toy'"
    ):
        datasets.load("nope")


def test_unknown_split_and_option(toy):
    with pytest.raises(ValueError, match="Unknown split for 'toy': 'dev'"):
        datasets.load("toy", split="dev")
    with pytest.raises(ValueError, match="Unknown options for 'toy': language"):
        datasets.load("toy", language="en")


def test_extract_from_loaded_corpus(toy):
    corpus = datasets.load("toy")
    results = TermExtractor(method="cvalue", n=2, words_layer="tokens").extract(corpus)
    extracted = {t["term"] for doc in results.terms for t in doc["terms"]}
    assert "heart failure" in extracted
    # Extraction does not touch the gold layer
    assert datasets.gold_terms(corpus) == datasets.gold_terms(datasets.load("toy"))
