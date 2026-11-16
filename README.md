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

The transformer-based `token-classification` and `embedrank` extractors need PyTorch and Hugging Face Transformers, which are large, so they are an optional extra. Install them with:

```bash
pip install "termxtract[transformers]"
```

The `llm` method talks to a large language model through the OpenAI API or a compatible local server, such as Ollama; install its client with `pip install "termxtract[llm]"`. All other methods work with the base install.

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
    print(doc["doc_id"], [(t["term"], round(t["score"], 2)) for t in doc["terms"][:3]])
# doc_0 [('machine learning', 3.21), ('artificial intelligence', 1.07), ('learning', 0.28)]
# ...
```

Each document's terms are ranked from highest to lowest score. Pass `threshold=...` to keep only terms scoring at or above a minimum. Candidates that start or end with a stopword, such as `is a` or `of machine`, are dropped by default; pass `stopwords=None` to keep them, or a list of words to use your own stopwords.

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

`extract` trains on a corpus and extracts terms from the same corpus. To train on one corpus and extract terms from another, use `extractor.fit(train_corpus, labels).predict(test_corpus)`. For a Teanga corpus with gold terms in a `terms` layer, such as those of `termxtract.datasets`, leave out `labels` to train on them.

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

Words are read from the `words` layer or, if there is none, from a `tokens` layer, as written by Teanga's CoNLL-U reader and spaCy service. If the corpus has part-of-speech tags, `pos_pattern="noun_phrase"` keeps only noun phrases, and if it has lemmas, `lemma_layer="lemma"` counts `cell` and `cells` as one term (see the [TF-IDF tutorial](https://teanga.io/TermXtract/tutorials/tfidf_tutorial/#part-of-speech-patterns)).

## Methods

TermXtract has 21 unsupervised methods, from TF-IDF, C-value and weirdness to TextRank, YAKE! and MDERank, and four supervised ones: Roger, a feature-based classifier, token classification with a fine-tuned transformer, and in-context learning with a large language model. Any ranking can be re-ranked by related seed terms, and training data can be extended with synthetic sentences. See [the methods](https://teanga.io/TermXtract/methods/) for the full list and their parameters, and [the tutorials](https://teanga.io/TermXtract/tutorials/getting_started/) for how to use them.

## Benchmarks

[BENCHMARKS.md](BENCHMARKS.md) scores every method on three English datasets and on Hindi, Slovenian and Irish ones, and compares them with published results. In short:

- **With labelled training data**, token classification and in-context learning with a large language model are far ahead of the other methods. Token classification has F1 of 58 to 68 on every dataset with training data, against at most 43 for any unsupervised method, and in-context learning is close behind everywhere but on Hindi. With only a few labelled sentences, in-context learning is best.
- **Without training data**, the methods that compare the corpus with a general-language reference corpus (weirdness, domain pertinence and relevance) rank terms best on five of the nine test sets, followed by the t-score, MDERank and TF-IDF.

See [Which method should I use?](https://teanga.io/TermXtract/benchmarks/) for more.

## Development

See [Development](https://teanga.io/TermXtract/development/) for how to run the tests and checks, and how to make a release.

## License

TermXtract is licensed under the [Apache License 2.0](LICENSE), the same license as [Teanga](https://github.com/TeangaNLP/teanga).
