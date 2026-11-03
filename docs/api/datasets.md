# Datasets

`termxtract.datasets` loads standard term-extraction datasets as Teanga corpora, with the gold terms in a `terms` span layer. Datasets are not distributed with TermXtract: each one is downloaded from its original source on first use, checked against a SHA-256 checksum, and cached under `~/.cache/termxtract`. Set the `TERMXTRACT_DATA` environment variable, or pass `home=`, to cache them elsewhere.

```python
from termxtract import TermExtractor, datasets, evaluate

corpus = datasets.load("acter", split="test", language="en")
results = TermExtractor(method="cvalue", words_layer="tokens").extract(corpus)
scores = evaluate(results, corpus, top_k=1000)
```

Each dataset keeps its own license, given by `datasets.info(name).license`; please cite the publication in `datasets.info(name).citation` when you use it.

| Name | Dataset | Languages | Domains | License | Splits | Options |
|------|---------|-----------|---------|---------|--------|---------|
| `acter` | [ACTER 1.5](https://github.com/AylaRT/ACTER) | en, fr, nl | corruption, dressage, heart failure, wind energy | CC BY-NC-SA 4.0 | `train`, `test` | `language`, `domains`, `include_named_entities`, `labels` |

## ACTER

ACTER (Annotated Corpora for Term Extraction Research) is the benchmark of the TermEval 2020 shared task. Each annotated text is a document. The `train` split has the corruption, dressage and wind energy domains, and the `test` split heart failure, as in TermEval. By default only English is loaded, and named entities do not count as terms.

ACTER lists the terms of each language and domain, not their occurrences, so the `terms` layer marks every occurrence of a listed term in a document, nested terms included, and `categories` gives its label. ACTER's own sequential IOB labels, for token classification, are in the `iob` layer, alongside `tokens`, `sentences` and `source` (the file name).

::: termxtract.datasets.acter
    options:
      show_root_heading: false
      members:
        - load_acter

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
