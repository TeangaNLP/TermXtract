# C-Value and Related Methods

This tutorial covers the methods that score a term by its frequency and by how it **nests** in other candidates: **C-value**, **NC-value**, **Basic** and **ComboBasic**, and **Domain coherence**, which builds on Basic. They are among the most widely used term extraction methods, and need nothing but the corpus.

## Setup

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

These methods count frequencies over the whole corpus, so a term has the same score in every document it occurs in. To see the ranking of the whole corpus, we merge the documents with this helper:

```python
def show_terms(results, limit=8):
    scores = {t["term"]: t["score"] for doc in results.terms for t in doc["terms"]}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for term, score in ranked[:limit]:
        print(f"{score:.3f}  {term}")
```

## C-value

**C-value** (Frantzi et al., 2000) favours frequent, long candidates, but discounts a candidate that mostly occurs inside longer ones. A candidate *t* of |*t*| words, with frequency *f*(*t*), scores:

- `log2(|t| + 0.1) × f(t)` if no other candidate contains it;
- `log2(|t| + 0.1) × (f(t) - mean frequency of the candidates that contain it)` otherwise.

The `0.1` lets single words score too, as in ATR4S (Astrakhantsev, 2018). With `n=3`, candidates have up to three words:

```python
extractor = TermExtractor(method="cvalue", n=3)
results = extractor.extract(corpus)
show_terms(results)
```

```text
4.282  training data
2.141  neural network
2.141  gradient descent
1.632  network is trained
1.632  trained by gradient
1.632  recurrent neural network
1.632  neural network reads
1.632  reads a sentence
```

`training data` occurs four times, and never inside a longer candidate. `neural network` occurs three times, but four three-word candidates that occur once contain it, such as `recurrent neural network`, so its score is `log2(2.1) × (3 - 1)`. The three-word candidates that occur once, such as `reads a sentence`, all tie: C-value rewards length, so on a small corpus, fragments score highly.

Most term extractors only rank candidates that occur at least twice. With `min_frequency=2`, the fragments are dropped before scoring, so they no longer discount the terms nested in them:

```python
extractor = TermExtractor(method="cvalue", n=3, min_frequency=2)
results = extractor.extract(corpus)
show_terms(results)
```

```text
4.282  training data
3.211  neural network
3.211  gradient descent
2.141  decision tree
2.141  random forest
0.275  trained
0.275  one
0.275  time
```

Now the five two-word terms come first. The single words score least, since `log2(1.1)` is small, and a word such as `network`, which occurs four times but three times in `neural network`, scores `log2(1.1) × (4 - 3)`, or 0.138.

## NC-value

**NC-value** (Frantzi, Ananiadou & Mima, 2000) re-ranks C-value by the words around each candidate. It takes the `top_terms` (default 200) best candidates by C-value as a sample of terms, and weights each content word near them by the share of the sample it occurs with. Each candidate then scores `cvalue_weight × C-value + (1 - cvalue_weight) × context`, where the context factor adds up the weights of the words within `window_size` (default 5) words of it, and `cvalue_weight` is 0.8 by default:

```python
extractor = TermExtractor(method="ncvalue", n=3, min_frequency=2)
results = extractor.extract(corpus)
show_terms(results)
```

```text
4.120  training data
3.285  gradient descent
3.190  neural network
2.134  random forest
2.049  decision tree
1.110  network
0.968  gradient
0.968  training
```

`gradient descent` now ranks above `neural network`: they have the same C-value, but the words around `gradient descent` occur near more of the sample terms, so its context factor is larger. On the benchmarks, NC-value is a little ahead of C-value on Slovenian and Irish, and behind it on English and Hindi.

## Basic and ComboBasic

**Basic** (Bordea et al., 2013) does the opposite of C-value with nesting: it **rewards** a candidate for occurring inside many other candidates, on the grounds that a term such as `neural network` is the head of many more specific terms. A candidate scores `|t| × log f(t) + alpha × e(t)`, where *e*(*t*) is the number of candidates that contain it, and `alpha` is 0.5 by default:

```python
extractor = TermExtractor(method="basic", n=3)
results = extractor.extract(corpus)
show_terms(results)
```

```text
4.197  neural network
3.697  gradient descent
2.773  training data
1.886  decision tree
1.886  random forest
1.386  network
1.386  training
1.386  data
```

`neural network` is inside four three-word candidates, so it gains 2 points and comes first. Basic only counts the nesting of multi-word candidates, so single words score `log f(t)`.

**ComboBasic** (Astrakhantsev, 2015) also rewards a candidate for **containing** other candidates, with weight `beta` (default 0.5): `|t| × log f(t) + alpha × e(t) + beta × e'(t)`, where *e*'(*t*) is the number of candidates in it. A higher `alpha` favours general terms, such as `neural network`, and a higher `beta` specific ones, such as `recurrent neural network`:

```python
extractor = TermExtractor(method="combobasic", n=3)
results = extractor.extract(corpus)
show_terms(results, limit=10)
```

```text
5.197  neural network
4.697  gradient descent
3.773  training data
2.886  decision tree
2.886  random forest
2.500  recurrent neural network
2.500  neural network reads
2.500  sentence one word
2.500  convolutional neural network
2.500  neural network applies
```

Every three-word candidate contains five candidates, its three words and two pairs, so they all gain 2.5 points, and the terms among them are tied with fragments such as `neural network reads`. As for C-value, `min_frequency=2` or a [part-of-speech pattern](tfidf_tutorial.md#part-of-speech-patterns) keeps such fragments out.

## Domain coherence

**Domain coherence** (Bordea, Buitelaar & Polajnar, 2013) scores candidates by how strongly they are associated with the words that characterise the domain:

1. The `seed_terms` (default 200) best candidates by Basic are a model of the domain.
2. The context words are the content words that occur in at least a quarter of the documents. The `context_words` (default 50) of them that are most associated with the seed terms are kept.
3. Each candidate scores its average association with those context words.

The association is the normalised PMI of co-occurring within `window_size` (default 5) words, which is between -1 and 1:

```python
extractor = TermExtractor(method="domaincoherence", n=2, min_frequency=2)
results = extractor.extract(corpus)
show_terms(results)
```

```text
-0.183  gradient
-0.220  network
-0.292  descent
-0.300  trained
-0.321  data
-0.325  gradient descent
-0.331  training
-0.363  training data
```

On a corpus of eight sentences, few words occur in a quarter of the documents, and most pairs of candidates and context words never co-occur, scoring -1, so all the scores are negative and the single words come first. Domain coherence needs a corpus of hundreds of documents to work, and even then it ranks below C-value on every benchmark dataset.

## Choosing a method

- **C-value** is the best of these on the benchmarks, and among the best methods that need neither a reference corpus nor an encoder. It is a good default for multi-word terms.
- **NC-value** costs a little more and is about as good.
- **Basic** favours general terms that head many others, and **ComboBasic** specific terms, but both do worse than C-value on the benchmarks, as does **domain coherence**.

All of them need enough text for the frequencies to differ: on a small corpus, many candidates tie. Use them with `min_frequency=2` and, for a Teanga corpus with part-of-speech tags, a `pos_pattern` (see the [TF-IDF tutorial](tfidf_tutorial.md#part-of-speech-patterns)).

See the API reference for [C-Value](../api/unsupervised/cvalue.md), [NC-Value](../api/unsupervised/ncvalue.md), [Basic](../api/unsupervised/basic.md), [ComboBasic](../api/unsupervised/combobasic.md) and [Domain Coherence](../api/unsupervised/domaincoherence.md) for all parameters.
