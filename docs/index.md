# TermXtract

**TermXtract** is a Python library for **automatic term extraction (ATE)**: finding the domain-specific terms in a collection of documents. It has 21 unsupervised methods, from TF-IDF, C-value and weirdness to TextRank, YAKE! and MDERank, and four supervised ones, which learn from labelled examples: Roger, a feature-based classifier, token classification with a fine-tuned transformer, and in-context learning with a large language model.

Every method takes either a plain list of strings or a [Teanga](https://github.com/TeangaNLP/teanga) `Corpus`, and returns the same `ATEResults` object.

## Installation

TermXtract needs Python 3.10 or later:

```bash
pip install termxtract
```

The `token-classification` and `embedrank` methods, and re-ranking, need PyTorch and Hugging Face Transformers, which are large, so they are an optional extra: `pip install "termxtract[transformers]"`. The `llm` method talks to a large language model through the OpenAI API or a compatible local server, such as Ollama; install its client with `pip install "termxtract[llm]"`. All other methods work with the base install.

## A first example

```python
from termxtract import TermExtractor

corpus = [
    "Machine learning is a field of artificial intelligence.",
    "Deep learning is a subfield of machine learning.",
    "Natural language processing applies machine learning to text.",
]

results = TermExtractor(method="cvalue", n=2).extract(corpus)
for doc in results.terms:
    print(doc["doc_id"], [(t["term"], round(t["score"], 2)) for t in doc["terms"][:3]])
```

```text
doc_0 [('machine learning', 3.21), ('artificial intelligence', 1.07), ('learning', 0.28)]
doc_1 [('machine learning', 3.21), ('deep learning', 1.07), ('learning', 0.28)]
doc_2 [('machine learning', 3.21), ('natural language', 1.07), ('language processing', 1.07)]
```

Each document's terms are ranked from highest to lowest score. C-value counts frequencies over the whole corpus, so `machine learning`, which occurs three times, scores the same in every document.

## Where next

- [Tutorials](tutorials/getting_started.md) show how to use every method, starting with [TF-IDF](tutorials/tfidf_tutorial.md), which also covers the options that all methods share.
- [Methods](methods.md) lists the methods and their parameters.
- [Which method should I use?](benchmarks.md) summarises how the methods compare on six benchmark datasets in English, Hindi, Slovenian and Irish.
- The API reference documents every class and function.
- [Development](development.md) covers running the tests and making a release.
