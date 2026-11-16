# Unithood Methods

The methods of the other tutorials measure **termhood**: how specific a candidate is to the domain. The unithood methods measure **unithood** instead: how strongly the words of a multi-word candidate stick together, compared with how often they occur apart. `machine learning` is a unit because `machine` and `learning` occur together far more often than chance would predict, while `learning is` is not.

TermXtract has five unithood measures, from the statistics of collocations:

| `method` | Measure | Favours |
|---|---|---|
| `pmi` | Pointwise mutual information (Church & Hanks, 1990): `log2(P(x y) / (P(x) P(y)))` | Rare pairs whose words seldom occur apart |
| `llr` | Dunning's (1993) log-likelihood ratio | Frequent pairs |
| `chi2` | Pearson's χ² statistic | Pairs whose words seldom occur apart |
| `dice` | The Dice coefficient (Smadja et al., 1996): `2 f(x y) / (f(x) + f(y))` | Pairs whose words seldom occur apart |
| `tscore` | Student's t-score of the pair's frequency against chance | Frequent pairs |

Each is computed from a contingency table of how often the two parts occur together and apart, over all the sequences of words of the corpus.

## Setup

We use the corpus of the [C-value tutorial](cvalue_tutorial.md), on machine learning:

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
```

## Comparing the measures

The unithood methods only rank multi-word candidates, since a single word has no parts to associate, and by default `n` is 2. Each measure scores a candidate once for the whole corpus. Here are the five best candidates by each:

```python
for method in ["pmi", "llr", "chi2", "dice", "tscore"]:
    results = TermExtractor(method=method).extract(corpus)
    scores = {t["term"]: t["score"] for doc in results.terms for t in doc["terms"]}
    best = sorted(scores, key=scores.get, reverse=True)[:5]
    print(f"{method:7}", ", ".join(best))
```

```text
pmi     applies filters, learning rate, rate controls, sentence one, one word
llr     training data, gradient descent, neural network, random forest, decision tree
chi2    gradient descent, training data, applies filters, random forest, learning rate
dice    gradient descent, training data, applies filters, random forest, learning rate
tscore  training data, gradient descent, neural network, random forest, decision tree
```

- **PMI** puts first the pairs that occur once, with words that occur nowhere else, such as `applies filters` and `rate controls`. Its score depends on how rare the words are, not on how often the pair occurs, so it overrates rare pairs.
- **χ² and Dice** give their highest score to every pair whose words never occur apart, however often it occurs. `gradient descent`, which occurs three times, ties with `applies filters`, which occurs once. `neural network` comes later, because `network` also occurs once without `neural`.
- **The log-likelihood ratio and the t-score** grow with the evidence, so they rank the frequent pairs first: here, the five terms of the corpus.

On the benchmarks, the t-score and the log-likelihood ratio are much the best of the five, and among the best of all the unsupervised methods on Slovenian. PMI, χ² and Dice do worse, except on Hindi, where PMI was the best method (see [Which method should I use?](../benchmarks.md#without-training-data)).

## Longer candidates

With `n=3` or more, a candidate of *k* words can be split into two parts in *k* - 1 places, such as `recurrent | neural network` and `recurrent neural | network`, and each split has its own score. `aggregation` combines them: `"min"` (the default) scores the candidate by its weakest split, so that every part of it must stick together, and `"mean"` by the mean of its splits. For example, `TermExtractor(method="tscore", n=3, aggregation="mean")`.

## Things to know

- **Single-word terms are never found.** If your domain has many, combine a unithood method with a termhood method, or use the [features](supervised/features.md) method, which uses both, with training data.
- **Unithood is not termhood.** A frequent general phrase, such as `long time` or `next year`, sticks together as well as any term. Stopword filtering removes such phrases when they start or end with a stopword, and a [part-of-speech pattern](tfidf_tutorial.md#part-of-speech-patterns) removes those that are not noun phrases.
- **The counts cover every pair of words**, not just the candidates, so filters such as `min_frequency` and `pos_pattern` change which candidates are ranked but not their scores.

See the [Unithood API reference](../api/unsupervised/unithood.md) for all parameters.
