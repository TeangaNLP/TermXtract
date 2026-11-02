# Datasets

`termxtract.datasets` loads standard term-extraction datasets as Teanga corpora, with the gold terms in a `terms` span layer. Datasets are not distributed with TermXtract: each one is downloaded from its original source on first use, checked against a SHA-256 checksum, and cached under `~/.cache/termxtract`. Set the `TERMXTRACT_DATA` environment variable, or pass `home=`, to cache them elsewhere.

```python
from termxtract import TermExtractor, datasets, evaluate

corpus = datasets.load(name, split="test")
results = TermExtractor(method="cvalue", words_layer="tokens").extract(corpus)
scores = evaluate(results, corpus)
```

Each dataset keeps its own license, given by `datasets.info(name).license`; please cite the publication in `datasets.info(name).citation` when you use it.

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
