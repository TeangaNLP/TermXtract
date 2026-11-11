import hashlib

import pytest

from termxtract import TermExtractor, datasets
from termxtract.datasets import base, irish_wikipedia

# Two articles in the layout of the file: a title running into the first sentence, a
# token containing "_", a double space, a lone quotation mark tagged B, and an article
# that does not end with a full stop
LINES = [
    (
        "Spéirling_B Ophelia_I Stoirm_B ghaoithe_I a_O bhuail_O hÉireann_B ._O "
        "Ba_O é_O Ophelia_B an_O deichiú_O spéirling_B den_O séasúr_B ._O"
    ),
    (
        "Jules_O Mazarin_O  Taidhleoir_B a_O bhí_O ann_O ,_O ``_B Príomh_B Aire_I ``_B "
        "a_b_c_O"
    ),
]


@pytest.fixture
def mini(tmp_path, monkeypatch):
    """Serve the miniature corpus in place of the real one."""
    path = tmp_path / "source" / "irish-wikipedia-terms.txt"
    path.parent.mkdir()
    path.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    monkeypatch.setenv("TERMXTRACT_DATA", str(tmp_path / "cache"))
    resource = datasets.Resource(
        path.as_uri(),
        hashlib.sha256(path.read_bytes()).hexdigest(),
        "irish-wikipedia-terms.txt",
    )
    monkeypatch.setitem(
        base._REGISTRY,
        "irish-wikipedia",
        datasets.DatasetInfo(
            **{**vars(irish_wikipedia.IRISH_WIKIPEDIA), "resources": (resource,)}
        ),
    )


def test_registered():
    info = datasets.info("irish-wikipedia")
    assert info.languages == ("ga",)
    assert info.splits == ()
    assert info.options == ()
    assert "CC BY-SA 4.0" in info.license


def test_load(mini):
    corpus = datasets.load("irish-wikipedia")
    first, second = (corpus.doc_by_id(d) for d in corpus.doc_ids)
    assert first.text == (
        "Spéirling Ophelia Stoirm ghaoithe a bhuail hÉireann . "
        "Ba é Ophelia an deichiú spéirling den séasúr ."
    )
    assert first.terms.text == [
        "Spéirling Ophelia",
        "Stoirm ghaoithe",
        "hÉireann",
        "Ophelia",
        "spéirling",
        "séasúr",
    ]
    assert first.sentences.text == [
        "Spéirling Ophelia Stoirm ghaoithe a bhuail hÉireann .",
        "Ba é Ophelia an deichiú spéirling den séasúr .",
    ]
    # The quotation marks are not terms, and the tag follows the last "_"
    assert second.terms.text == ["Taidhleoir", "Príomh Aire"]
    assert second.tokens.text[-1] == "a_b_c"
    assert second.sentences.text == [second.text]


def test_extract_from_irish_wikipedia(mini):
    corpus = datasets.load("irish-wikipedia")
    results = TermExtractor(method="tfidf", n=2, words_layer="tokens").extract(corpus)
    extracted = {t["term"] for doc in results.terms for t in doc["terms"]}
    assert "Spéirling Ophelia" in extracted


def test_real_irish_wikipedia():
    """Check the real corpus, if it is already in the cache."""
    try:
        corpus = datasets.load("irish-wikipedia", download_if_missing=False)
    except FileNotFoundError:
        pytest.skip("The Irish Wikipedia gold standard is not cached")
    assert len(corpus.doc_ids) == 11
    assert sum(len(corpus.doc_by_id(d).tokens) for d in corpus.doc_ids) == 5178
    assert sum(len(corpus.doc_by_id(d).terms) for d in corpus.doc_ids) == 844
    assert len(datasets.gold_terms(corpus)) == 577
