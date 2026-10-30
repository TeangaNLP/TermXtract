# TF-IDF Tutorial

This tutorial shows how to extract terms with **TF-IDF** (term frequency–inverse document frequency), the simplest method in TermXtract. You will learn how to run it on plain text and on a Teanga corpus, how to extract multi-word terms, how to choose a threshold, and how to see the scores behind the results.

## How TF-IDF scores a term

TF-IDF rewards n-grams that are **frequent in one document but rare across the corpus**. For an n-gram *t* in document *d*, TermXtract computes:

- **TF**: the number of times *t* occurs in *d*, divided by the total number of n-grams in *d*.
- **IDF**: `log(N / (1 + df))`, where *N* is the number of documents and *df* is the number of documents containing *t*.
- **Score**: `TF × IDF`.

Two consequences are worth knowing before you start:

- **You need several documents.** IDF compares documents with each other. An n-gram that occurs in all, or all but one, of the documents gets an IDF of zero or less, and with a single document *every* score is negative.
- **TF-IDF finds what is distinctive about each document,** not what is typical of a domain. A term that is spread over many documents in your corpus is penalised, even if it is an important term.

## Setup

Install TermXtract (see the [README](https://github.com/TeangaNLP/TermXtract#installation)) and import the `TermExtractor`:

```python
from termxtract import TermExtractor
```

We will use this small corpus of six documents on three topics: photosynthesis, neural networks and gravity.

```python
corpus = [
    "Photosynthesis converts light energy into chemical energy in plant cells.",
    "Chlorophyll in the chloroplast absorbs light for photosynthesis.",
    "A neural network learns weights from training data.",
    "Gradient descent updates the weights of a neural network.",
    "A black hole bends light with its strong gravity.",
    "Gravity holds the planets in orbit around the sun.",
]
```

## Extracting terms from plain text

Create an extractor with `method="tfidf"` and call `extract()` with a list of strings:

```python
extractor = TermExtractor(method="tfidf")
results = extractor.extract(corpus)

print(results.terms[0])
```

```text
{'doc_id': 'doc_0', 'terms': ['photosynthesis', 'converts', 'light', 'energy', 'into', 'chemical', 'in', 'plant', 'cells']}
```

`results.terms` has one entry per document, with an automatically assigned `doc_id` (`doc_0`, `doc_1`, …). Notice that:

- Text is lower-cased and split into words, and punctuation is dropped.
- Without a threshold, **every** word in the document is returned.
- Terms are listed in the order they first appear, **not ranked by score**.
- Unlike the other methods, TF-IDF returns terms as **plain strings, without scores**. See [Seeing the scores](#seeing-the-scores) below for how to get them.

## Filtering with a threshold

Use `threshold` to keep only n-grams scoring at or above a minimum:

```python
extractor = TermExtractor(method="tfidf", threshold=0.1)
results = extractor.extract(corpus)

for doc in results.terms:
    print(doc["doc_id"], doc["terms"])
```

```text
doc_0 ['converts', 'energy', 'into', 'chemical', 'plant', 'cells']
doc_1 ['chlorophyll', 'chloroplast', 'absorbs', 'for']
doc_2 ['learns', 'from', 'training', 'data']
doc_3 ['gradient', 'descent', 'updates', 'of']
doc_4 ['black', 'hole', 'bends', 'with', 'its', 'strong']
doc_5 ['holds', 'planets', 'orbit', 'around', 'sun']
```

Words that occur in more than one document are gone, including function words like `the` but also topical words like `photosynthesis`, `network` and `gravity`. What remains are the words specific to each document.

**Choosing a threshold.** TF-IDF scores are small: TF is a fraction of the document's length, so scores shrink as documents get longer or as `n` grows. A threshold that works for one corpus can remove everything in another. Here, `threshold=0.15` keeps just one term in the whole corpus: `energy`, whose score is doubled because it occurs twice in `doc_0`. Look at the actual scores first (see below) and pick a value from them.

## Multi-word terms

Set `n` to the maximum n-gram length to consider. With `n=2`, both single words and two-word phrases are scored:

```python
extractor = TermExtractor(method="tfidf", n=2, threshold=0.05)
results = extractor.extract(corpus)

print(results.terms[3])
```

```text
{'doc_id': 'doc_3', 'terms': ['gradient', 'gradient descent', 'descent', 'descent updates', 'updates', 'updates the', 'the weights', 'weights of', 'of', 'of a']}
```

`gradient descent` is found, but so are fragments like `updates the` and `weights of`, and `neural network` is missing because it appears in two documents (`doc_2` and `doc_3`). TF-IDF has no notion of what makes a good phrase. If multi-word terms matter to you, also try [C-Value](../api/unsupervised/cvalue.md), which is designed for them.

## Seeing the scores

Because TF-IDF returns only the term strings, you can use the extractor's own `compute_tf` and `compute_idf` methods to see, and rank by, the scores. The tokenization below matches the one TermXtract uses internally:

```python
import re

def tfidf_scores(corpus, n=1):
    """Return a {term: score} dict for each document, ranked from highest to lowest."""
    extractor = TermExtractor(method="tfidf", n=n).extractor
    tokenized = [re.findall(r"\b\w+\b", doc.lower()) for doc in corpus]
    ngrams = [extractor.generate_ngrams_strings(words) for words in tokenized]
    idf = extractor.compute_idf(ngrams)

    ranked = []
    for doc_ngrams in ngrams:
        tf = extractor.compute_tf(doc_ngrams)
        scores = {term: tf[term] * idf[term] for term in tf}
        ranked.append(dict(sorted(scores.items(), key=lambda item: -item[1])))
    return ranked

scores = tfidf_scores(corpus)
for term, score in scores[1].items():
    print(f"{score:.3f}  {term}")
```

```text
0.137  chlorophyll
0.137  chloroplast
0.137  absorbs
0.137  for
0.087  photosynthesis
0.051  in
0.051  the
0.051  light
```

This makes the threshold behaviour from earlier easy to see: everything at or above 0.1 is kept, and no term in this document reaches 0.15.

## Using a Teanga corpus

The extractor also accepts a [Teanga](https://github.com/TeangaNLP/teanga) `Corpus` that has a `words` span layer. This lets you use your own tokenization, and the results then include the **character offsets** of every occurrence of each term.

Here we build the `words` layer with a simple regular expression:

```python
import re
from teanga import Corpus

teanga_corpus = Corpus()
teanga_corpus.add_layer_meta("text")
teanga_corpus.add_layer_meta("words", layer_type="span", base="text")
for text in corpus:
    doc = teanga_corpus.add_doc(text)
    doc.words = [match.span() for match in re.finditer(r"\w+", text)]

extractor = TermExtractor(method="tfidf", n=2, threshold=0.05)
results = extractor.extract(teanga_corpus)

doc_id = list(teanga_corpus.doc_ids)[3]
print(results.terms[3]["terms"][:3])
print(results.offsets[3]["offsets"][:3])
```

```text
['Gradient', 'Gradient descent', 'descent']
[{'term': 'Gradient', 'offset': (0, 8)}, {'term': 'Gradient descent', 'offset': (0, 16)}, {'term': 'descent', 'offset': (9, 16)}]
```

Each offset is a `(start, end)` pair, so `teanga_corpus.doc_by_id(doc_id).text[0:16]` is `"Gradient descent"`. Note the differences from plain-text input:

- `doc_id`s are Teanga's document IDs rather than `doc_0`, `doc_1`, ….
- Words are taken exactly as they appear in your `words` layer, so **case is preserved** (`Gradient`, not `gradient`).
- The extractor adds a `terms` layer definition to your corpus, so running a second extractor on the same `Corpus` object fails with *"Layer with name terms already exists"*. Build a fresh corpus for each run.

## Limitations and tips

- **No stopword filtering.** Function words such as `for` and `of` score just as highly as content words when they are rare in your corpus. Use a larger corpus, or post-filter the results with a stopword list.
- **Use enough documents.** With very few documents, IDF can only take a handful of values and many terms tie.
- **Compare against a reference corpus.** To find terms typical of a *domain* rather than of individual documents, try [Weirdness](../api/unsupervised/weirdness.md) or [Domain Pertinence](../api/unsupervised/domainpertinence.md), which compare your corpus with general-language text.

See the [TF-IDF API reference](../api/unsupervised/tfidf.md) for the full list of parameters and methods.
