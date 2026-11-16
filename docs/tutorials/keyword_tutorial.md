# Keyphrase Methods

This tutorial covers three methods from **keyphrase extraction**, the task of finding the few phrases that describe a single document: **RAKE**, **YAKE!** and **TextRank**, with its variants SingleRank and PositionRank. None of them needs anything but the text. They were designed for single documents, so they score each document's phrases from that document alone, or from a graph of its words, rather than from counts over a corpus.

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

## RAKE

**RAKE** (Rapid Automatic Keyword Extraction; Rose et al., 2010) splits the text into phrases at stopwords and punctuation, and takes the phrases between them as candidates, however long. It then scores each word by its **degree**, the number of words in the candidates it occurs in, divided by its frequency, and each candidate by the sum of the scores of its words. Long candidates score highest.

RAKE does not use the candidates of the other methods, so it takes none of their parameters, such as `n` and `stopwords`. Instead, it needs a `stoplist` and the `phrase_delimiters`, the punctuation that ends a phrase. Here we use TermXtract's English stopwords:

```python
from termxtract.stopwords import load_stopwords

extractor = TermExtractor(
    method="rake",
    stoplist=sorted(load_stopwords("english")),
    phrase_delimiters=[".", ",", ";", ":", "!", "?"],
)
results = extractor.extract(corpus)

for doc in results.terms[:4]:
    print(doc["doc_id"], [(t["term"], t["score"]) for t in doc["terms"]])
```

```text
doc_0 [('neural network', 4.0), ('gradient descent', 4.0), ('training data', 4.0), ('trained', 1.0)]
doc_1 [('recurrent neural network reads', 16.0), ('sentence one word', 9.0), ('time', 1.0)]
doc_2 [('convolutional neural network applies filters', 25.0), ('pixels', 1.0), ('image', 1.0)]
doc_3 [('gradient descent updates', 9.0), ('training data', 4.0), ('weights', 1.0), ('network', 1.0), ('batch', 1.0)]
```

Each document is scored on its own. The phrases stop only at stopwords and punctuation, so a verb that is not a stopword joins the phrases around it, as in `convolutional neural network applies filters`, which scores 25: each of its five words has degree 5. Two candidates that are separated only by stopwords at least twice in a document, in the same order, are also joined into one, such as `axis of evil`.

RAKE needs no tags, but its candidates are often too long to be terms. On the benchmarks, it ranks in the middle on English and fourth on Irish, but last or nearly last on Hindi and Slovenian, whose terms are short.

## YAKE!

**YAKE!** (Campos et al., 2020) scores each word of a document by five features: how often it is capitalised, how early it occurs, how frequent it is, how many different words occur around it, and how many of the document's sentences it occurs in. A candidate's score combines those of its words and its own frequency. Lower scores are better in YAKE!, so TermXtract reports them negated, so that higher is better, as for every method. Its candidates are those of the other methods, with `n` of up to 3 words by default:

```python
extractor = TermExtractor(method="yake", n=2, corpus_aggregation=None)
results = extractor.extract(corpus)

for doc in results.terms[:3]:
    print(doc["doc_id"], [t["term"] for t in doc["terms"][:4]])
```

```text
doc_0 ['training data', 'gradient descent', 'neural network', 'data']
doc_1 ['network reads', 'neural network', 'one word', 'recurrent neural']
doc_2 ['applies filters', 'convolutional neural', 'network applies', 'neural network']
```

Going down each document's ranking, a candidate is dropped if it is too similar to one already kept, with a similarity above `dedup_threshold` (default 0.9), so that a document's keyphrases are not near-repeats of each other.

`corpus_aggregation=None` keeps each document's own scores. By default (`"max"`), each term gets its best score in any document, in all of them, so that a term scores the same everywhere and the documents' rankings can be merged into one for the corpus. `"mean"` takes the mean instead. With the default, `neural network` ranks first in `doc_2` too, since it scores better in `doc_0` and `doc_1`.

## TextRank

**TextRank** (Mihalcea & Tarau, 2004) builds a graph of the words, linking two words if they occur within `window_size` words of each other, and scores each word by **PageRank** over the graph: a word scores highly if it is linked to many words that score highly. A candidate scores the sum of the scores of its words. `variant` picks one of three versions:

- `"plain"`, TextRank: the links are unweighted, and the window is 1 word by default, so only neighbouring words are linked.
- `"single"`, SingleRank (Wan & Xiao, 2008): each link is weighted by how often its words co-occur, in a window of 9 words by default.
- `"position"`, PositionRank (Florescu & Caragea, 2017): as SingleRank, but words that occur early in their documents score higher.

```python
for variant in ["plain", "single", "position"]:
    extractor = TermExtractor(method="textrank", variant=variant, n=2, min_frequency=2)
    results = extractor.extract(corpus)
    scores = {t["term"]: t["score"] for doc in results.terms for t in doc["terms"]}
    print(f"{variant:8}", ", ".join(sorted(scores, key=scores.get, reverse=True)[:6]))
```

```text
plain    neural network, gradient descent, decision tree, random forest, training data, one
single   neural network, training data, gradient descent, decision tree, random forest, network
position neural network, gradient descent, training data, decision tree, random forest, network
```

Only the words that are not stopwords, and, for a Teanga corpus with part-of-speech tags, the nouns and adjectives, are nodes of the graph. Other words, such as `of` in `degree of freedom`, take up places in the window but score 0.

By default, one graph is built over the whole corpus, so a term has the same score in every document. With `graph="document"`, each document has its own graph, as in the papers, which rank the keyphrases of single documents.

## Choosing a method

On the benchmarks, all three usually rank terms worse than the term extraction methods, such as [C-value](cvalue_tutorial.md) and [weirdness](reference_corpus_tutorial.md): the TextRank variants are among the six worst methods on every dataset but the ACL RD-TEC (see [Which method should I use?](../benchmarks.md#without-training-data)). Use them to find the keyphrases of single documents, or when nothing else is available. RAKE does best of them on English and Irish, and YAKE! on Hindi and Slovenian.

See the API reference for [RAKE](../api/unsupervised/rake.md), [YAKE!](../api/unsupervised/yake.md) and [TextRank](../api/unsupervised/textrank.md) for all parameters.
