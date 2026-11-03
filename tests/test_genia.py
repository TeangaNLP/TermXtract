import hashlib
import io
import tarfile

import pytest

from termxtract import TermExtractor, datasets, evaluate
from termxtract.datasets import base, genia

# A miniature GENIA term corpus, with nested terms, a span annotated twice, a
# coordinated term and a word with brackets
CORPUS = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE set SYSTEM "gpml.dtd">
<set>
<import resource="GENIAontology.daml" prefix="G"></import>
<article>
<articleinfo>
<bibliomisc>MEDLINE:1</bibliomisc>
</articleinfo>
<title>
<sentence><cons lex="IL-2_gene_expression" sem="G#other_name"><cons lex="IL-2_gene" \
sem="G#DNA_domain_or_region"><cons lex="IL-2" sem="G#protein_molecule">IL-2</cons> \
gene</cons> expression</cons> in <cons lex="T_cell" sem="G#cell_type"><cons \
lex="T_cell" sem="G#cell_type">T cells</cons></cons>.</sentence>
</title>
<abstract>
<sentence>We studied <cons lex="(AND alpha-globin beta-globin)" \
sem="(AND G#DNA_domain_or_region G#DNA_domain_or_region)"><cons lex="alpha-*">alpha-\
</cons> and <cons lex="beta-*">beta-</cons><cons lex="*globin">globin</cons></cons> \
and <cons lex="1,25(OH)2D3" sem="G#lipid">1,25(OH)2D3</cons> (<cons lex="Ca2+" \
sem="G#atom">Ca2+</cons>).</sentence>
</abstract>
</article>
<article>
<articleinfo>
<bibliomisc>MEDLINE:2</bibliomisc>
</articleinfo>
<title>
<sentence><cons lex="IL-2" sem="G#protein_molecule">IL-2</cons> again.</sentence>
</title>
<abstract>
</abstract>
</article>
</set>
"""


def write_genia(path):
    """Write a tgz laid out like the GENIA term corpus archive."""
    data = CORPUS.encode("utf-8")
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("GENIA_term_3.02/GENIAcorpus3.02.xml")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))


@pytest.fixture
def mini(tmp_path, monkeypatch):
    """Serve the miniature corpus in place of the real one."""
    path = tmp_path / "source" / "genia.tgz"
    path.parent.mkdir()
    write_genia(path)
    monkeypatch.setenv("TERMXTRACT_DATA", str(tmp_path / "cache"))
    resource = datasets.Resource(
        path.as_uri(),
        hashlib.sha256(path.read_bytes()).hexdigest(),
        "GENIAcorpus3.02.tgz",
    )
    monkeypatch.setitem(
        base._REGISTRY,
        "genia",
        datasets.DatasetInfo(**{**vars(genia.GENIA), "resources": (resource,)}),
    )


def by_source(corpus):
    return {corpus.doc_by_id(d).source: corpus.doc_by_id(d) for d in corpus.doc_ids}


def test_registered():
    info = datasets.info("genia")
    assert info.splits == ()
    assert info.options == ("nested", "labels")
    assert len(genia.CLASSES) == 36


def test_load(mini):
    docs = by_source(datasets.load("genia"))
    assert list(docs) == ["MEDLINE:1", "MEDLINE:2"]
    doc = docs["MEDLINE:1"]
    assert doc.sentences.text == [
        "IL-2 gene expression in T cells.",
        "We studied alpha- and beta-globin and 1,25(OH)2D3 (Ca2+).",
    ]
    # Outermost terms only; the doubly annotated "T cells" is one term; the
    # coordination and its unclassified parts are left out
    assert list(zip(doc.terms.text, doc.categories)) == [
        ("IL-2 gene expression", "other_name"),
        ("T cells", "cell_type"),
        ("1,25(OH)2D3", "lipid"),
        ("Ca2+", "atom"),
    ]
    assert doc.tokens.text == [
        "IL-2",
        "gene",
        "expression",
        "in",
        "T",
        "cells",
        ".",
        "We",
        "studied",
        "alpha-",
        "and",
        "beta-globin",
        "and",
        "1,25(OH)2D3",
        "(",
        "Ca2+",
        ")",
        ".",
    ]


def test_nested(mini):
    doc = by_source(datasets.load("genia", nested=True))["MEDLINE:1"]
    assert list(zip(doc.terms.text, doc.categories))[:4] == [
        ("IL-2 gene expression", "other_name"),
        ("IL-2 gene", "DNA_domain_or_region"),
        ("IL-2", "protein_molecule"),
        ("T cells", "cell_type"),
    ]


def test_labels(mini):
    # The outermost protein terms, though they are nested in other terms
    corpus = datasets.load("genia", labels=["protein_molecule"])
    assert datasets.gold_terms(corpus, by_document=True) == {
        doc_id: {"il-2"} for doc_id in corpus.doc_ids
    }
    with pytest.raises(ValueError, match="Unknown GENIA labels: protein"):
        datasets.load("genia", labels=["protein"])


def test_tokenize():
    text = '"NF-kappa B" (p50/p65), [Ca2+]i and f(x)).'
    tokens = [text[start:end] for start, end in genia._tokenize(text)]
    assert tokens == [
        '"',
        "NF-kappa",
        "B",
        '"',
        "(",
        "p50/p65",
        ")",
        ",",
        "[",
        "Ca2+]i",
        "and",
        "f(x)",
        ")",
        ".",
    ]


def test_evaluate_on_genia(mini):
    corpus = datasets.load("genia")
    results = TermExtractor(method="cvalue", n=3, words_layer="tokens").extract(corpus)
    extracted = {t["term"].lower() for doc in results.terms for t in doc["terms"]}
    # Words with brackets and plus signs come out whole, matching the gold terms
    assert {"il-2", "1,25(oh)2d3", "ca2+"} <= extracted
    assert evaluate(results, corpus).true_positives >= 3


def test_real_genia():
    """Check the real corpus, if it is already in the cache."""
    try:
        corpus = datasets.load("genia", nested=True, download_if_missing=False)
    except FileNotFoundError:
        pytest.skip("The GENIA term corpus is not cached")
    assert len(corpus.doc_ids) == 2000
    # 91,125 classified terms, less 107 spans annotated twice
    assert sum(len(corpus.doc_by_id(d).terms) for d in corpus.doc_ids) == 91018
