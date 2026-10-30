# TF-IDF Tutorial

This tutorial shows how to extract terms with **TF-IDF** (term frequency–inverse document frequency), the simplest method in TermXtract. You will learn how to run it on plain text and on a Teanga corpus, how to read the scores, how to choose a threshold, and how to extract multi-word terms.

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

print(results.terms[1]["doc_id"])
for term in results.terms[1]["terms"]:
    print(f"{term['score']:.3f}  {term['term']}")
```

```text
doc_1
0.137  chlorophyll
0.137  chloroplast
0.137  absorbs
0.137  for
0.087  photosynthesis
0.051  in
0.051  the
0.051  light
```

`results.terms` has one entry per document, with an automatically assigned `doc_id` (`doc_0`, `doc_1`, …). Each term is a `{"term": ..., "score": ...}` dictionary, the same format as every other method. Notice that:

- Text is lower-cased and split into words, and punctuation is dropped.
- Terms are **ranked by score**, highest first. Terms with equal scores keep the order in which they first appear.
- Without a threshold, **every** word in the document is returned.

The words found only in this document (`chlorophyll`, `chloroplast`, `absorbs`, `for`) score highest. `photosynthesis` also occurs in `doc_0`, so it scores lower, and `in`, `the` and `light` occur in three documents each.

## Filtering with a threshold

Use `threshold` to keep only n-grams scoring at or above a minimum:

```python
extractor = TermExtractor(method="tfidf", threshold=0.1)
results = extractor.extract(corpus)

for doc in results.terms:
    print(doc["doc_id"], [term["term"] for term in doc["terms"]])
```

```text
doc_0 ['energy', 'converts', 'into', 'chemical', 'plant', 'cells']
doc_1 ['chlorophyll', 'chloroplast', 'absorbs', 'for']
doc_2 ['learns', 'from', 'training', 'data']
doc_3 ['gradient', 'descent', 'updates', 'of']
doc_4 ['black', 'hole', 'bends', 'with', 'its', 'strong']
doc_5 ['holds', 'planets', 'orbit', 'around', 'sun']
```

Words that occur in more than one document are gone, including function words like `the` but also topical words like `photosynthesis`, `network` and `gravity`. What remains are the words specific to each document.

**Choosing a threshold.** TF-IDF scores are small: TF is a fraction of the document's length, so scores shrink as documents get longer or as `n` grows. A threshold that works for one corpus can remove everything in another. Here, `threshold=0.15` keeps just one term in the whole corpus: `energy`, whose score is doubled because it occurs twice in `doc_0` (which is also why it is ranked first above). Look at the scores without a threshold first and pick a value from them. In the scores for `doc_1` shown earlier, everything at or above 0.1 is kept, and nothing reaches 0.15.

## Multi-word terms

Set `n` to the maximum n-gram length to consider. With `n=2`, both single words and two-word phrases are scored:

```python
extractor = TermExtractor(method="tfidf", n=2, threshold=0.05)
results = extractor.extract(corpus)

print([term["term"] for term in results.terms[3]["terms"]])
```

```text
['gradient', 'gradient descent', 'descent', 'descent updates', 'updates', 'updates the', 'the weights', 'weights of', 'of', 'of a']
```

`gradient descent` is found, but so are fragments like `updates the` and `weights of`, and `neural network` is missing because it appears in two documents (`doc_2` and `doc_3`). TF-IDF has no notion of what makes a good phrase. If multi-word terms matter to you, also try [C-Value](../api/unsupervised/cvalue.md), which is designed for them.

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

doc_id = results.offsets[3]["doc_id"]
print([term["term"] for term in results.terms[3]["terms"]][:3])
print(results.offsets[3]["offsets"][:3])
```

```text
['Gradient', 'Gradient descent', 'descent']
[{'term': 'Gradient', 'offset': (0, 8)}, {'term': 'Gradient descent', 'offset': (0, 16)}, {'term': 'descent', 'offset': (9, 16)}]
```

`results.offsets` lists every occurrence of each extracted term, in document order. Each offset is a `(start, end)` pair, so `teanga_corpus.doc_by_id(doc_id).text[0:16]` is `"Gradient descent"`. Note the differences from plain-text input:

- `doc_id`s are Teanga's document IDs rather than `doc_0`, `doc_1`, ….
- Words are taken exactly as they appear in your `words` layer, so **case is preserved** (`Gradient`, not `gradient`).

## Limitations and tips

- **No stopword filtering.** Function words such as `for` and `of` score just as highly as content words when they are rare in your corpus. Use a larger corpus, or post-filter the results with a stopword list.
- **Use enough documents.** With very few documents, IDF can only take a handful of values and many terms tie.
- **Compare against a reference corpus.** To find terms typical of a *domain* rather than of individual documents, try the [reference corpus methods](reference_corpus_tutorial.md), which compare your corpus with general-language text.

See the [TF-IDF API reference](../api/unsupervised/tfidf.md) for the full list of parameters and methods.
