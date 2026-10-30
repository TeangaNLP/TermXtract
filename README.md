# TermXtract

[![CI](https://github.com/TeangaNLP/TermXtract/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/TeangaNLP/TermXtract/actions/workflows/ci-cd.yml)
[![Docs](https://img.shields.io/badge/docs-teanga.io%2FTermXtract-blue)](https://teanga.io/TermXtract/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**TermXtract** is a Python library for **automatic term extraction (ATE)**: finding the domain-specific terms in a collection of documents. It offers a range of classic **unsupervised** scoring methods (TF-IDF, C-Value, Weirdness, RAKE, topic models, …) and **supervised** methods that learn from labelled examples (AdaBoost, Roger, and transformer-based token classification).

Every method takes either a plain list of strings or a [Teanga](https://github.com/TeangaNLP/teanga) `Corpus`, and returns the same `ATEResults` object.

📖 **Documentation:** <https://teanga.io/TermXtract/>

## Installation

TermXtract needs **Python 3.10 or later**. It is not yet on PyPI, so install it from GitHub:

```bash
pip install git+https://github.com/TeangaNLP/TermXtract.git
```

This also installs [Teanga](https://pypi.org/project/teanga/), PyTorch and Hugging Face Transformers (used by the token-classification extractor), so the first install can take a while.

## Quickstart

### Unsupervised extraction

```python
from termxtract import TermExtractor

corpus = [
    "Machine learning is a field of artificial intelligence.",
    "Deep learning is a subfield of machine learning.",
    "Natural language processing applies machine learning to text.",
]

extractor = TermExtractor(method="cvalue", n=2)  # consider unigrams and bigrams
results = extractor.extract(corpus)

for doc in results.terms:
    print(doc["doc_id"], doc["terms"][:3])
# doc_0 [{'term': 'machine', 'score': 0.0}, {'term': 'machine learning', 'score': 1.0}, ...]
```

Pass `threshold=...` to keep only terms scoring at or above a minimum.

### Methods that need a reference corpus

`weirdness`, `domainpertinence` and `relevance` compare the target corpus against a general-language reference corpus:

```python
reference = [
    "The weather was pleasant and people walked in the park.",
    "She bought bread and milk at the shop on the corner.",
]

extractor = TermExtractor(method="weirdness", reference_corpus=reference, n=2)
results = extractor.extract(corpus)
```

### Supervised extraction

Supervised methods are trained from a dictionary of candidate terms labelled `1` (term) or `0` (not a term):

```python
from termxtract import SupervisedTermExtractor

labels = {
    "machine learning": 1,
    "deep learning": 1,
    "natural language processing": 1,
    "field": 0,
    "text": 0,
}

extractor = SupervisedTermExtractor(method="adaboost", n=3)
results = extractor.extract(corpus, labels)
```

The AdaBoost extractor uses [Open English WordNet](https://en-word.net/) through the [`wn`](https://pypi.org/project/wn/) package. The first time it needs WordNet it downloads the `oewn:2025+` lexicon into `~/.wn_data` (set `WN_DATA_DIR` to change this). To install it ahead of time, e.g. for offline use, run `python -m wn download oewn:2025+`. Pass `wordnet_lexicon=...` to use a different lexicon.

### Using a Teanga corpus

Any extractor also accepts a Teanga `Corpus` with a `words` span layer:

```python
from teanga import Corpus

corpus = Corpus()
corpus.add_layer_meta("text")
corpus.add_layer_meta("words", layer_type="span", base="text")
doc = corpus.add_doc("Deep learning is a subfield of machine learning.")
doc.words = [(0, 4), (5, 13), (14, 16), (17, 18), (19, 27), (28, 30), (31, 38), (39, 47)]

results = TermExtractor(method="basic", n=2).extract(corpus)
```

## Available methods

### Unsupervised — `TermExtractor(method=...)`

| `method`           | Description                                         | Extra parameters                      |
|--------------------|-----------------------------------------------------|---------------------------------------|
| `tfidf`            | Term frequency–inverse document frequency           |                                       |
| `ridf`             | Residual IDF                                        |                                       |
| `cvalue`           | C-Value (nested multi-word terms)                   |                                       |
| `basic`            | Basic                                               | `alpha`                               |
| `combobasic`       | ComboBasic                                          | `alpha`, `beta`                       |
| `rake`             | Rapid Automatic Keyword Extraction                  | `stoplist`, `phrase_delimiters` (required) |
| `domaincoherence`  | Domain coherence                                    | `window_size`                         |
| `domainpertinence` | Domain pertinence                                   | `reference_corpus` (required)         |
| `weirdness`        | Weirdness                                           | `reference_corpus` (required)         |
| `relevance`        | Relevance                                           | `reference_corpus` (required)         |
| `topicmodeling`    | LDA topic-model-based scoring                       | `num_topics`                          |
| `nmf`              | NMF topic-model-based scoring                       | `n_topics`, `max_features`            |

All methods accept `n` (maximum n-gram length) and `threshold` (minimum score). `rake` does not use `n`.

### Supervised — `SupervisedTermExtractor(method=...)`

| `method`               | Description                                              | Extra parameters                                         |
|------------------------|----------------------------------------------------------|----------------------------------------------------------|
| `adaboost`             | AdaBoost over linguistic and statistical features        | `threshold`, `max_depth`, `n_estimators`, `estimator`, `wordnet_lexicon` |
| `roger`                | Evolutionary ranking-function search                     | `population_size`, `generations`, `subsample_ratio`      |
| `token-classification` | Fine-tuned Hugging Face token-classification model       | `model_name`, `epochs`, `batch_size`, `learning_rate`, `max_length`, `language` |

## Results

`extract()` returns an `ATEResults` object with:

- `results.terms`: one entry per document, `{"doc_id": ..., "terms": [...]}`. For most methods each term is `{"term": str, "score": float}`.
- `results.offsets`: term character offsets. Currently only populated by `tfidf` on Teanga input; `None` otherwise.
- `results.corpus`: the input corpus.

## Development

```bash
git clone https://github.com/TeangaNLP/TermXtract.git
cd TermXtract
poetry install --with dev

poetry run python -m unittest discover tests   # run the test suite
poetry run mkdocs serve                        # preview the docs locally
```

## License

TermXtract is licensed under the [Apache License 2.0](LICENSE), the same license as [Teanga](https://github.com/TeangaNLP/teanga).
