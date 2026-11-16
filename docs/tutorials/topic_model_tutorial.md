# Topic Model Methods

A **topic model** describes a corpus as a mixture of a few topics, each a distribution over words. The words that are most typical of some topic are likely to be terms of the domain. TermXtract has two methods built on topic models: **topic modelling** with LDA, after the Novel Topic Model of Li et al. (2013), and **NMF**, which factorises the TF-IDF matrix of the corpus.

## Setup

We use the corpus of the [C-value tutorial](cvalue_tutorial.md), whose sentences are on two subjects, neural networks and decision trees:

```python
from termxtract import TermExtractor

corpus = [
    "A neural network is trained by gradient descent on the training data.",
    "A recurrent neural network reads a sentence one word at a time.",
    "A convolutional neural network applies filters to the pixels of an image.",
    "Gradient descent updates the weights of the network after each batch of training data.",
    "A decision tree splits the training data on the value of one feature at a time.",
    "A random forest averages the predictions of many decision trees.",
    "Each decision tree of a random forest is trained on a sample of the training data.",
    "The learning rate controls how far gradient descent moves the weights at each step.",
]


def show_terms(results, limit=8):
    scores = {t["term"]: t["score"] for doc in results.terms for t in doc["terms"]}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for term, score in ranked[:limit]:
        print(f"{score:.3f}  {term}")
```

Both methods score each term once for the whole corpus, so `show_terms` merges the documents into one ranking.

## Topic modelling (LDA)

`method="topicmodeling"` trains an LDA topic model with `num_topics` topics (default 20) on the single words of the corpus, leaving out stopwords. The `top_words` (default 200) most probable words of each topic are its typical words. A candidate scores `log f(t) × Σ max_k φ_k(w)`: the sum, over its words, of each word's highest probability in any topic, times the log of the candidate's frequency. Words that are not typical of any topic count 0. Here we ask for two topics, one for each subject:

```python
extractor = TermExtractor(method="topicmodeling", num_topics=2, n=2, min_frequency=2)
results = extractor.extract(corpus)
show_terms(results)
```

```text
0.158  neural network
0.147  gradient descent
0.141  training data
0.114  network
0.084  decision tree
0.077  decision
0.073  gradient
0.073  descent
```

Multi-word candidates add up the probabilities of their words, so they score higher than their words alone. Since the score multiplies by `log f(t)`, a candidate that occurs once scores 0.

Unlike the Novel Topic Model, the model has no background topic for general words, and it is trained with scikit-learn's variational LDA.

## NMF

`method="nmf"` builds the TF-IDF matrix of the documents' candidates, factorises it into `n_topics` topics (default 10) by non-negative matrix factorisation, and scores each candidate by its largest weight in any topic:

```python
extractor = TermExtractor(method="nmf", n_topics=2, n=2, min_frequency=2)
results = extractor.extract(corpus)
show_terms(results)
```

```text
0.641  network
0.555  neural
0.555  neural network
0.535  gradient
0.535  gradient descent
0.535  descent
0.521  decision
0.446  random
```

Each candidate is a column of the matrix, and `neural` only occurs in `neural network`, so the two have the same weights: unlike LDA, NMF does not prefer multi-word candidates to their words. `max_features` keeps only that many candidates, those with the highest frequencies; the others are not scored.

## Choosing a method

On the benchmarks, the two methods are in the middle of the unsupervised methods, and vary from dataset to dataset: NMF is the best method on Slovenian chemistry and second on Hindi, but 14th of 25 on the ACL RD-TEC, where LDA is fifth (see [Which method should I use?](../benchmarks.md#without-training-data)). Both depend on the number of topics, which is worth tuning for your corpus: a few for a corpus on one subject, more for a varied one.

See the API reference for [Topic Modeling (LDA)](../api/unsupervised/topicmodeling.md) and [NMF](../api/unsupervised/nmf.md) for all parameters.
