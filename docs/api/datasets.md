# Datasets

`termxtract.datasets` loads standard term-extraction datasets as Teanga corpora, with the gold terms in a `terms` span layer. Datasets are not distributed with TermXtract: each one is downloaded from its original source (or, for HTEC, its Teanga version on Zenodo) on first use, checked against a SHA-256 checksum, and cached under `~/.cache/termxtract`. Set the `TERMXTRACT_DATA` environment variable, or pass `home=`, to cache them elsewhere.

```python
from termxtract import TermExtractor, datasets, evaluate

corpus = datasets.load("acter", split="test", language="en")
results = TermExtractor(method="cvalue", words_layer="tokens").extract(corpus)
scores = evaluate(results, corpus, top_k=1000)
```

Each dataset keeps its own license, given by `datasets.info(name).license`; please cite the publication in `datasets.info(name).citation` when you use it.

| Name | Dataset | Languages | Domains | License | Splits | Options |
|------|---------|-----------|---------|---------|--------|---------|
| `acl-rd-tec` | [ACL RD-TEC 2.0](http://hdl.handle.net/11372/LRT-1661) | en | computational linguistics | CC BY-NC-SA 4.0 | | `annotator`, `labels` |
| `genia` | [GENIA term corpus 3.02](http://www.geniaproject.org/genia-corpus/term-corpus) | en | biomedicine | GENIA Project License (annotations CC BY 3.0) | | `nested`, `labels` |
| `htec` | [HTEC 2.0](https://doi.org/10.5281/zenodo.24111051) | hi | education | CC BY-SA 4.0 | | `partial_words`, `labels` |
| `acter` | [ACTER 1.5](https://github.com/AylaRT/ACTER) | en, fr, nl | corruption, dressage, heart failure, wind energy | CC BY-NC-SA 4.0 | `train`, `test` | `language`, `domains`, `include_named_entities`, `labels` |

## ACTER

ACTER (Annotated Corpora for Term Extraction Research) is the benchmark of the TermEval 2020 shared task. Each annotated text is a document. The `train` split has the corruption, dressage and wind energy domains, and the `test` split heart failure, as in TermEval. By default only English is loaded, and named entities do not count as terms.

ACTER lists the terms of each language and domain, not their occurrences, so the `terms` layer marks every occurrence of a listed term in a document, nested terms included, and `categories` gives its label. ACTER's own sequential IOB labels, for token classification, are in the `iob` layer, alongside `tokens`, `sentences` and `source` (the file name).

::: termxtract.datasets.acter
    options:
      show_root_heading: false
      members:
        - load_acter

## ACL RD-TEC 2.0

The ACL RD-TEC 2.0 annotates terms in context in 300 abstracts of ACL Anthology papers (1978–2006), and classes each term as a method or technique (`tech`), tool, language resource (`lr`), named language resource (`lr-prod`), model, measure (`measure(ment)`) or other. It is downloaded from LINDAT/CLARIN.

Two annotators annotated the abstracts, and 171 were annotated by both. By default each abstract is loaded once, with the annotations of annotator 1 where there are two, since annotator 1 was the more consistent in the authors' self-agreement study; pass `annotator=1` or `annotator=2` for one annotator's abstracts and annotations. The documents also have the `lemma`, `pos` (Penn Treebank) and `upos` (Universal Dependencies) layers of the distribution's tagged files, so `pos_pattern="noun_phrase"` works without a tagger.

::: termxtract.datasets.acl_rd_tec
    options:
      show_root_heading: false
      members:
        - load_acl_rd_tec
        - CLASSES
        - PTB_TO_UPOS

## GENIA

The GENIA term corpus 3.02 annotates biomedical terms in 2,000 MEDLINE abstracts on transcription factors in human blood cells, in 36 classes of the GENIA ontology such as `protein_molecule` and `cell_type`. Each title and abstract is a document. It is downloaded from NaCTeM. The annotations are licensed under CC BY 3.0 and must be attributed as "Corpus annotations (c) GENIA Project"; the abstracts are under PubMed's terms.

Terms nest deeply in GENIA (`IL-2 gene expression` contains `IL-2 gene`, which contains `IL-2`). By default only the outermost terms are gold terms; pass `nested=True` for all of them. Coordinated terms such as `alpha- and beta-globin` are left out. The corpus is not tokenised, so the loader splits the text at whitespace and splits punctuation off the ends of words, keeping words such as `IL-2`, `Ca2+` and `1,25(OH)2D3` whole so that extracted terms can match the gold terms.

::: termxtract.datasets.genia
    options:
      show_root_heading: false
      members:
        - load_genia
        - CLASSES

## HTEC 2.0

HTEC 2.0, the Hindi Term Extraction Corpus, annotates education terms in 97 documents, segments of 65 Hindi Wikipedia articles, in seven categories such as *Educationists, Learners and Researchers* and *Ambiguous* (Banerjee, Chakravarthi & McCrae, 2025). It is downloaded from Zenodo, as a Teanga file converted from the annotations by `scripts/htec_to_teanga.py`, under CC BY-SA 4.0 like the Wikipedia texts. `source` gives the document ID, such as `99_a`; documents with the same number come from the same article.

The corpus is not tokenised, so the loader splits the text into words, keeping Devanagari vowel signs and viramas inside them, and into sentences at line breaks and after `।`, `?`, `!` and `.`. The annotations include nested and overlapping terms, all kept. Some start inside a word, such as `शोध` in `संशोधित`, and are left out unless `partial_words=True`; those that end inside a word, mostly stems of inflected words such as `कक्षा` in `कक्षाओं`, are kept.

::: termxtract.datasets.htec
    options:
      show_root_heading: false
      members:
        - load_htec
        - CATEGORIES

## Adding a dataset

A dataset is a `DatasetInfo` passed to `register`. Its `loader` receives the paths of the downloaded `resources`, a `split` keyword argument and the dataset's options as keyword-only arguments, and returns a Teanga corpus with a `terms` span layer, plus a `categories` layer if the dataset classifies its terms.

::: termxtract.datasets.base
    options:
      show_root_heading: false
      members:
        - load
        - download
        - list_datasets
        - info
        - gold_terms
        - data_home
        - register
        - DatasetInfo
        - Resource
        - GOLD_LAYER
        - CATEGORY_LAYER
        - SOURCE_LAYER
        - SENTENCES_LAYER
