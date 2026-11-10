"""
Standard term-extraction datasets, loaded as Teanga corpora.

```python
from termxtract import datasets

for dataset in datasets.list_datasets():
    print(dataset.name, dataset.languages, dataset.license)

corpus = datasets.load("acter", split="test", language="en")  # downloaded on first use
```

Each corpus has its gold terms in a `terms` span layer, so it can be passed to an
extractor and, as the gold standard, to `termxtract.evaluate`.
"""

# Importing a dataset module registers the dataset
from . import acl_rd_tec, acter, genia, htec
from .base import (
    CATEGORY_LAYER,
    GOLD_LAYER,
    SENTENCES_LAYER,
    SOURCE_LAYER,
    DatasetInfo,
    Resource,
    data_home,
    download,
    gold_terms,
    info,
    list_datasets,
    load,
    register,
)

__all__ = [
    "CATEGORY_LAYER",
    "GOLD_LAYER",
    "SENTENCES_LAYER",
    "SOURCE_LAYER",
    "DatasetInfo",
    "Resource",
    "acl_rd_tec",
    "acter",
    "data_home",
    "download",
    "genia",
    "gold_terms",
    "htec",
    "info",
    "list_datasets",
    "load",
    "register",
]
