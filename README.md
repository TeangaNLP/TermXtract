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

The transformer-based `token-classification` extractor needs PyTorch and Hugging Face Transformers, which are large, so they are an optional extra. Install them with:

```bash
pip install "termxtract[transformers] @ git+https://github.com/TeangaNLP/TermXtract.git"
```

All other methods work with the base install.

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
# doc_0 [{'term': 'machine learning', 'score': 1.0}, {'term': 'learning is', 'score': 1.0}, {'term': 'is a', 'score': 1.0}]
# ...
```

Each document's terms are ranked from highest to lowest score.

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
| `token-classification` | Fine-tuned Hugging Face token-classification model (needs the `transformers` extra) | `model_name`, `epochs`, `batch_size`, `learning_rate`, `max_length`, `language` |

## Results

`extract()` returns an `ATEResults` object with:

- `results.terms`: one entry per document, `{"doc_id": ..., "terms": [{"term": str, "score": float}, ...]}`, with terms ranked from highest to lowest score. Every method uses this format.
- `results.offsets`: for Teanga input, one entry per document, `{"doc_id": ..., "offsets": [{"term": str, "offset": (start, end)}, ...]}`, giving the character offsets of every occurrence of each extracted term. It is `None` for plain-text input, and for `nmf` and `domaincoherence`, which score terms over the whole corpus rather than per document.
- `results.corpus`: the input corpus.

## Development

```bash
git clone https://github.com/TeangaNLP/TermXtract.git
cd TermXtract
poetry install --with dev --extras transformers

poetry run pytest                              # run the test suite
poetry run pytest --cov                        # ...with a coverage report
poetry run ruff check termxtract tests         # lint
poetry run ruff format termxtract tests        # format
poetry run mypy termxtract                     # type check
poetry run mkdocs serve                        # preview the docs locally
```

CI runs all of these on every pull request, with the tests on Python 3.10–3.12, on Linux, macOS and Windows, and against the oldest dependency versions `pyproject.toml` allows. The codebase was reformatted in one commit, which `git blame` can skip with `git config blame.ignoreRevsFile .git-blame-ignore-revs`.

## License

TermXtract is licensed under the [Apache License 2.0](LICENSE), the same license as [Teanga](https://github.com/TeangaNLP/teanga).
