# TermXtract

[![CI](https://github.com/TeangaNLP/TermXtract/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/TeangaNLP/TermXtract/actions/workflows/ci-cd.yml)
[![Docs](https://img.shields.io/badge/docs-teanga.io%2FTermXtract-blue)](https://teanga.io/TermXtract/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**TermXtract** is a Python library for **automatic term extraction (ATE)**: finding the domain-specific terms in a collection of documents. It offers a range of classic **unsupervised** scoring methods (TF-IDF, C-Value, Weirdness, RAKE, topic models, …) and **supervised** methods that learn from labelled examples (Roger, and transformer-based token classification).

Every method takes either a plain list of strings or a [Teanga](https://github.com/TeangaNLP/teanga) `Corpus`, and returns the same `ATEResults` object.

📖 **Documentation:** <https://teanga.io/TermXtract/>  
📝 **Changelog:** [CHANGELOG.md](CHANGELOG.md). TermXtract follows [Semantic Versioning](https://semver.org/); before 1.0, minor releases may contain breaking changes.

## Installation

TermXtract needs **Python 3.10 or later**. Install it from PyPI:

```bash
pip install termxtract
```

or install the development version, with the changes since the last release, from GitHub:

```bash
pip install git+https://github.com/TeangaNLP/TermXtract.git
```

The transformer-based `token-classification` extractor needs PyTorch and Hugging Face Transformers, which are large, so they are an optional extra. Install them with:

```bash
pip install "termxtract[transformers]"
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
# doc_0 [{'term': 'machine learning', 'score': 1.0}, {'term': 'artificial intelligence', 'score': 1.0}, {'term': 'machine', 'score': 0.0}]
# ...
```

Each document's terms are ranked from highest to lowest score.

Pass `threshold=...` to keep only terms scoring at or above a minimum.

Candidates that start or end with a stopword, such as `is a` or `of machine`, are dropped by default, using a built-in list of English function words. Stopwords inside a term, as in `degree of freedom`, are kept. Pass `stopwords=None` to keep every candidate, or a list of words to use your own stopwords.

If a Teanga corpus has part-of-speech tags, `pos_pattern="noun_phrase"` keeps only noun phrases, such as `machine learning` or `degree of freedom` (see [Using a Teanga corpus](#using-a-teanga-corpus)).

### Methods that need a reference corpus

`weirdness`, `domainpertinence` and `relevance` compare the target corpus against a general-language reference corpus, or a list of reference corpora:

```python
reference = [
    "The weather was pleasant and people walked in the park.",
    "She bought bread and milk at the shop on the corner.",
]

extractor = TermExtractor(method="weirdness", reference_corpus=reference, n=2)
results = extractor.extract(corpus)
```

### Supervised extraction

Supervised methods are trained from a dictionary of candidate terms labelled `1` (term) or `0` (not a term). Roger ranks multi-word candidates, so it needs multi-word examples of both:

```python
from termxtract import SupervisedTermExtractor

labels = {
    "machine learning": 1,
    "deep learning": 1,
    "natural language processing": 1,
    "is a": 0,
    "applies machine": 0,
}

extractor = SupervisedTermExtractor(method="roger", n=3)
results = extractor.extract(corpus, labels)
```

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

Words are read from the `words` layer or, if there is none, from a `tokens` layer, as written by Teanga's CoNLL-U reader and spaCy service; pass `words_layer=...` to use another layer. If the corpus has part-of-speech tags in a `upos` or `pos` layer based on the words, pass `pos_pattern` to keep only the terms whose tags match a pattern, either `"noun_phrase"` or your own pattern such as `"<ADJ|NOUN>*<NOUN>"`, and if it has lemmas, pass `lemma_layer="lemma"` to count `cell` and `cells` as one term. See the [TF-IDF tutorial](https://teanga.io/TermXtract/tutorials/tfidf_tutorial/#part-of-speech-patterns).

## Available methods

`TermExtractor.METHODS` and `SupervisedTermExtractor.METHODS` map each method name to its implementation, so `list(TermExtractor.METHODS)` lists the available methods. Passing an unknown `method` raises a `ValueError` that lists them too.

### Unsupervised — `TermExtractor(method=...)`

| `method`           | Description                                         | Extra parameters                      |
|--------------------|-----------------------------------------------------|---------------------------------------|
| `tfidf`            | Term frequency–inverse document frequency           |                                       |
| `ridf`             | Residual IDF                                        |                                       |
| `cvalue`           | C-Value (frequent terms not nested in longer ones)  |                                       |
| `basic`            | Basic                                               | `alpha`                               |
| `combobasic`       | ComboBasic                                          | `alpha`, `beta`                       |
| `rake`             | Rapid Automatic Keyword Extraction                  | `stoplist`, `phrase_delimiters` (required) |
| `domaincoherence`  | Domain coherence                                    | `window_size`, `seed_terms`, `context_words` |
| `domainpertinence` | Domain pertinence                                   | `reference_corpus` (required), `tie_break` |
| `weirdness`        | Weirdness                                           | `reference_corpus` (required)         |
| `relevance`        | Relevance                                           | `reference_corpus` (required)         |
| `topicmodeling`    | LDA topic-model-based scoring                       | `num_topics`, `top_words`             |
| `nmf`              | NMF topic-model-based scoring                       | `n_topics`, `max_features`            |

All methods accept `n` (maximum n-gram length), `threshold` (minimum score), `stopwords` (default `"english"`), `min_frequency` (default 1), `word_pattern` (default `"alphanumeric"`, dropping words without a letter or digit), and, for Teanga input, `words_layer`, `lemma_layer`, `pos_pattern` and `pos_layer`. `rake` does not use `n`, `stopwords`, `min_frequency`, `word_pattern`, `words_layer` or `lemma_layer`; it splits candidates at its own `stoplist`.

### Supervised — `SupervisedTermExtractor(method=...)`

| `method`               | Description                                              | Extra parameters                                         |
|------------------------|----------------------------------------------------------|----------------------------------------------------------|
| `roger`                | Ranking function of association measures, learned by evolution (multi-word terms only) | `population_size`, `offspring_size`, `generations`, `runs`, `random_state` |
| `token-classification` | Fine-tuned Hugging Face token-classification model (needs the `transformers` extra) | `model_name`, `epochs`, `batch_size`, `learning_rate`, `max_length`, `language` |

All supervised methods accept `n` (maximum n-gram length, default 6), and `stopwords` and `pos_pattern`, which filter the extracted terms but not the training candidates, and `words_layer` and `pos_layer` for Teanga input.

## Results

`extract()` returns an `ATEResults` object with:

- `results.terms`: one entry per document, `{"doc_id": ..., "terms": [{"term": str, "score": float}, ...]}`, with terms ranked from highest to lowest score. Every method uses this format.
- `results.offsets`: for Teanga input, one entry per document, `{"doc_id": ..., "offsets": [{"term": str, "offset": (start, end)}, ...]}`, giving the character offsets of every occurrence of each extracted term. It is `None` for plain-text input.
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

### Releasing

Version numbers follow [Semantic Versioning](https://semver.org/): bump the patch version for bug fixes, the minor version for new features, and the major version for breaking changes (before 1.0, breaking changes bump the minor version).

1. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a new empty `## [Unreleased]` section above it, and update the comparison links at the bottom of the file.
2. Set the version with `poetry version X.Y.Z` (or `poetry version patch`, `minor` or `major`).
3. Commit both files as `Release X.Y.Z` and get the commit onto `main`.
4. On GitHub, draft a release with a new tag `vX.Y.Z` on that commit, paste the changelog section as the release notes, and publish it. The release workflow checks that the tag matches the version, builds the package and uploads it to PyPI.

## License

TermXtract is licensed under the [Apache License 2.0](LICENSE), the same license as [Teanga](https://github.com/TeangaNLP/teanga).
