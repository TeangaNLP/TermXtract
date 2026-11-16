# Supervised Methods Tutorial

This tutorial shows how to train a term extractor from examples with `SupervisedTermExtractor`. This page covers what all of them share: the format of the training labels, training on one corpus and extracting from another, and Teanga corpora. The four methods, **Roger**, the **features** classifier, **token classification** and **in-context learning** with a large language model, each have a [page of their own](#the-methods).

## How supervised extraction works

The unsupervised methods score every candidate with a fixed formula. A supervised method instead **learns** what a term looks like from candidates that you have labelled as terms or non-terms, and then scores all the candidates in the corpus.

Labels are a dictionary that maps each candidate to `1` (a term) or `0` (not a term):

```python
labels = {
    "photosynthesis": 1,
    "chloroplast": 1,
    "chlorophyll": 1,
    "light energy": 1,
    "chemical energy": 1,
    "glucose": 1,
    "mitochondria": 1,
    "cellular respiration": 1,
    "plant cell": 1,
    "plant cells": 1,
    "the": 0,
    "in": 0,
    "of": 0,
    "takes place": 0,
    "which": 0,
}
```

A few rules apply to the labels:

- **Write terms in lower case**, with single spaces between words. Plain-text input is lower-cased and split into words before candidates are generated, so `"Photosynthesis"` would never match.
- **Multi-word terms** are only generated as candidates if `n` is at least their number of words. Use `n=2` for the labels above.
- **Unlabelled candidates are only used by Roger and the features method, and only without non-terms.** They learn from the candidates you label, but if you label only terms, as with a list of gold terms, they count every other candidate as a non-term (see [Roger](supervised/roger.md)). Token classification learns only from the terms labelled `1`.
- **Stopwords are filtered from the results, not from training.** As for the unsupervised methods, extracted terms that start or end with a stopword are dropped (see [Stopwords](tfidf_tutorial.md#stopwords)), but candidates such as `the` and `of` are still used as training examples by Roger. The features method takes the candidates of the unsupervised methods, so it neither trains on them nor returns them. Pass `stopwords=None` to keep them.

`extract(corpus, labels)` trains the model on the corpus and then scores the candidates in the **same** corpus. To extract terms from other documents, train with `fit` and then call `predict` (see [Training on one corpus, extracting from another](#training-on-one-corpus-extracting-from-another)). The examples below use this small biology corpus:

```python
from termxtract import SupervisedTermExtractor

corpus = [
    "Photosynthesis takes place in the chloroplast of a plant cell.",
    "The chloroplast contains chlorophyll, which absorbs light energy.",
    "During photosynthesis, light energy is converted into chemical energy.",
    "Plant cells store the chemical energy as glucose.",
    "Mitochondria release the energy stored in glucose by cellular respiration.",
    "Cellular respiration takes place in the mitochondria of the cell.",
]
```

The supervised methods score each candidate once for the whole corpus, and each document's entry in `results.terms` lists the candidates that occur in that document, with their corpus-level scores. To see the ranking of the whole corpus, merge the documents:

```python
def corpus_ranking(results):
    """Merge the terms of all documents into one ranking."""
    scores = {t["term"]: t["score"] for doc in results.terms for t in doc["terms"]}
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
```

## Training on one corpus, extracting from another

`extract()` trains a new model and extracts terms from the corpus it was trained on. To train on one corpus, such as the training split of a dataset or another domain, and extract terms from another, call `fit` and then `predict`:

```python
extractor = SupervisedTermExtractor(method="features", n=2, random_state=0)
extractor.fit(corpus, labels)
results = extractor.predict(["Algae use chlorophyll for photosynthesis in the ocean."])
```

`predict` returns `ATEResults` like `extract`, with the terms of each document and, for a Teanga corpus, their offsets.

## The methods

Each method has a page of its own:

- [Roger](supervised/roger.md) learns a ranking of multi-word candidates from association measures between their words.
- [Features](supervised/features.md) trains a classifier over the scores of the unsupervised methods and other features of the candidates.
- [Token classification](supervised/token_classification.md) fine-tunes a transformer to tag the terms in each sentence.
- [In-context learning](supervised/llm.md) prompts a large language model with similar training sentences and their terms.
- [Synthetic training data](supervised/augmentation.md) adds sentences written by a large language model, or with terms from Wikipedia, to the training data of any of them.

To choose between them, see [Which method should I use?](../benchmarks.md#with-training-data)

## Using a Teanga corpus

All the methods also accept a [Teanga](https://github.com/TeangaNLP/teanga) `Corpus` with a `words` layer in place of the list of strings, with labels in the same format. As for the unsupervised methods, the results then include the offsets of each extracted term.

If the corpus marks its gold terms in a `terms` span layer, as the corpora of `termxtract.datasets` do, `fit` can learn from them without labels. Token classification then tags each annotated occurrence, rather than every occurrence of a labelled term, Roger takes the annotated terms as terms and every other candidate as a non-term, and in-context learning gives each sentence with its annotated terms as a demonstration. Here a model is trained on three domains of ACTER and tags the fourth:

```python
from termxtract import datasets

train = datasets.load("acter", split="train")
test = datasets.load("acter", split="test")
extractor = SupervisedTermExtractor(method="token-classification", random_state=0)
results = extractor.fit(train).predict(test)
```

Token classification and in-context learning split a Teanga document into sentences by its `sentences` layer, if it has one, or else after each `.`, `!` and `?`.

If the corpus has a lemma layer, such as the `lemma` layer of RSDO5 or the ACL RD-TEC, pass `lemma_layer="lemma"` to give each term by its lemmas, so that `cell` and `cells` are one term, as for the unsupervised methods. Roger then computes its measures over the lemmas, while token classification and in-context learning still read the words as written and only name the terms they find by their lemmas. Gold terms of the `terms` layer are read by their lemmas too, but labels are matched to the lemmas as given, so write them as lemmas. To evaluate, read the gold terms in the same way, with `datasets.gold_terms(corpus, lemma_layer="lemma")`.

See the API reference for [`SupervisedTermExtractor`](../api/supervised_term_extractor.md), [Roger](../api/supervised/roger.md), [Features](../api/supervised/features.md), [Token Classification](../api/supervised/token_classification.md) and [In-Context Learning](../api/supervised/llm.md) for all parameters.
