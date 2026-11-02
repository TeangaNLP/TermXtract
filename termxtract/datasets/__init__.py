"""
Standard term-extraction datasets, loaded as Teanga corpora.

```python
from termxtract import datasets

for dataset in datasets.list_datasets():
    print(dataset.name, dataset.languages, dataset.license)

corpus = datasets.load(name, split="test")  # downloaded on first use
```

Each corpus has its gold terms in a `terms` span layer, so it can be passed to an
extractor and, as the gold standard, to `termxtract.evaluate`.
"""

from .base import (
    CATEGORY_LAYER,
    GOLD_LAYER,
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
    "DatasetInfo",
    "Resource",
    "data_home",
    "download",
    "gold_terms",
    "info",
    "list_datasets",
    "load",
    "register",
]
