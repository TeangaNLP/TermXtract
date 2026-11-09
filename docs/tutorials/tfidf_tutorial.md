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
0.087  photosynthesis
0.051  light
```

`results.terms` has one entry per document, with an automatically assigned `doc_id` (`doc_0`, `doc_1`, …). Each term is a `{"term": ..., "score": ...}` dictionary, the same format as every other method. Notice that:

- Text is lower-cased and split into words, and punctuation is dropped.
- Terms are **ranked by score**, highest first. Terms with equal scores keep the order in which they first appear.
- Without a threshold, every word in the document is returned, except **stopwords** such as `in`, `the` and `for` (see [Stopwords](#stopwords) below).

The words found only in this document (`chlorophyll`, `chloroplast`, `absorbs`) score highest. `photosynthesis` also occurs in `doc_0`, so it scores lower, and `light` occurs in three documents.

## Filtering with a threshold

Use `threshold` to keep only n-grams scoring at or above a minimum:

```python
extractor = TermExtractor(method="tfidf", threshold=0.1)
results = extractor.extract(corpus)

for doc in results.terms:
    print(doc["doc_id"], [term["term"] for term in doc["terms"]])
```

```text
doc_0 ['energy', 'converts', 'chemical', 'plant', 'cells']
doc_1 ['chlorophyll', 'chloroplast', 'absorbs']
doc_2 ['learns', 'training', 'data']
doc_3 ['gradient', 'descent', 'updates']
doc_4 ['black', 'hole', 'bends', 'strong']
doc_5 ['holds', 'planets', 'orbit', 'around', 'sun']
```

Words that occur in more than one document are gone, including topical words like `photosynthesis`, `network` and `gravity`. What remains are the words specific to each document.

**Choosing a threshold.** TF-IDF scores are small: TF is a fraction of the document's length, so scores shrink as documents get longer or as `n` grows. A threshold that works for one corpus can remove everything in another. Here, `threshold=0.15` keeps just one term in the whole corpus: `energy`, whose score is doubled because it occurs twice in `doc_0` (which is also why it is ranked first above). Look at the scores without a threshold first and pick a value from them. In the scores for `doc_1` shown earlier, everything at or above 0.1 is kept, and nothing reaches 0.15.

## Multi-word terms

Set `n` to the maximum n-gram length to consider. With `n=2`, both single words and two-word phrases are scored:

```python
extractor = TermExtractor(method="tfidf", n=2, threshold=0.05)
results = extractor.extract(corpus)

print([term["term"] for term in results.terms[3]["terms"]])
```

```text
['gradient', 'gradient descent', 'descent', 'descent updates', 'updates']
```

`gradient descent` is found, but so is the fragment `descent updates`, and `neural network` is missing because it appears in two documents (`doc_2` and `doc_3`). TF-IDF has no notion of what makes a good phrase. If multi-word terms matter to you, also try [C-Value](../api/unsupervised/cvalue.md), which is designed for them.

## Stopwords

By default, every method drops candidates that **start or end with a stopword**, using a built-in list of 198 English function words (NLTK's English stopword list). Stopwords *inside* a multi-word candidate are kept, so a term like `degree of freedom` can still be found. Filtering only removes candidates from the results: the scores of the remaining terms are the same as without it.

Pass `stopwords=None` to see every candidate:

```python
extractor = TermExtractor(method="tfidf", n=2, threshold=0.05, stopwords=None)
results = extractor.extract(corpus)

print([term["term"] for term in results.terms[3]["terms"]])
```

```text
['gradient', 'gradient descent', 'descent', 'descent updates', 'updates', 'updates the', 'the weights', 'weights of', 'of', 'of a']
```

These are fragments such as `updates the`, `weights of` and `of a`, and the function word `of`, which scores highly only because no other document contains it.

You can also pass your own list, for example to add words that are common in your domain but are not terms, or to use another language:

```python
extractor = TermExtractor(
    method="tfidf", threshold=0.1, stopwords=["energy", "converts", "into"]
)
results = extractor.extract(corpus)

print([term["term"] for term in results.terms[0]["terms"]])
```

```text
['chemical', 'plant', 'cells']
```

A custom list **replaces** the built-in one, which is why `into` has to be included. To extend the built-in list, use `termxtract.stopwords.ENGLISH_STOPWORDS | {"energy", "converts"}`.

## Using a Teanga corpus

The extractor also accepts a [Teanga](https://github.com/TeangaNLP/teanga) `Corpus` that has a `words` span layer (or a `tokens` layer, or another span layer named with `words_layer`). This lets you use your own tokenization, and the results then include the **character offsets** of every occurrence of each term.

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

## Part-of-speech patterns

Stopwords only remove function words from the edges of candidates, so fragments such as `descent updates` still get through. Most terms are **noun phrases**, so if your Teanga corpus has part-of-speech tags, you can keep only the candidates whose tags match a pattern.

TermXtract reads the tags from a layer named `upos` or `pos`, which must be based on the layer holding the words. This is how Teanga's [CoNLL-U reader](https://github.com/TeangaNLP/teanga) (`upos` on a `tokens` layer) and its spaCy service (`pos` on a `tokens` layer) store them, and TermXtract reads words from a `tokens` layer when there is no `words` layer, so their output works as it is. Here we add the same layers by hand, with [Universal Dependencies](https://universaldependencies.org/u/pos/) tags:

```python
tags = [
    "NOUN VERB NOUN NOUN ADP ADJ NOUN ADP NOUN NOUN",
    "NOUN ADP DET NOUN VERB NOUN ADP NOUN",
    "DET ADJ NOUN VERB NOUN ADP NOUN NOUN",
    "NOUN NOUN VERB DET NOUN ADP DET ADJ NOUN",
    "DET ADJ NOUN VERB NOUN ADP PRON ADJ NOUN",
    "NOUN VERB DET NOUN ADP NOUN ADP DET NOUN",
]

tagged_corpus = Corpus()
tagged_corpus.add_layer_meta("text")
tagged_corpus.add_layer_meta("tokens", layer_type="span", base="text")
tagged_corpus.add_layer_meta("upos", layer_type="seq", base="tokens", data="string")
for text, doc_tags in zip(corpus, tags):
    doc = tagged_corpus.add_doc(text)
    doc.tokens = [match.span() for match in re.finditer(r"\w+", text)]
    doc.upos = doc_tags.split()

extractor = TermExtractor(
    method="tfidf", n=2, threshold=0.05, pos_pattern="noun_phrase"
)
results = extractor.extract(tagged_corpus)

for doc in results.terms[2:4]:
    print([term["term"] for term in doc["terms"]])
```

```text
['training', 'training data', 'data']
['Gradient', 'Gradient descent', 'descent']
```

Without the pattern, these documents also list the verbs `learns` and `updates` and the fragments `network learns`, `learns weights` and `descent updates`. With it, only nouns and noun phrases such as `training data` and `Gradient descent` remain. A candidate is kept if **any** of its occurrences matches the pattern, and only the occurrences that match are counted, so the scores of terms whose occurrences all match do not change. A word that is sometimes a noun and sometimes a verb is scored by its uses as a noun.

`pos_pattern="noun_phrase"` allows nouns and proper nouns, optionally preceded by adjectives or nouns, and optionally followed by a preposition and a second such phrase, as in `degree of freedom`. You can also write your own pattern, in the syntax of NLTK's `RegexpParser`: each tag is written in angle brackets and combined with regular-expression operators, so `"<ADJ>*<NOUN>+"` allows adjectives followed by nouns. Inside the angle brackets, `|` separates alternative tags and `.` matches any character of a tag, as in `"<ADJ|NOUN>*<N.*>"`.

A few more things to know:

- Pass `pos_layer=...` or `words_layer=...` if your layers have other names.
- `pos_pattern` needs tags, so it only works with a Teanga corpus, and raises a `ValueError` for plain text.
- The supervised methods accept `pos_pattern` too. They are still trained on every candidate, and only their results are filtered. So is RAKE, whose scores do not change.

### Rare and noisy candidates

Two more filters decide which n-grams are candidates, for every method except RAKE:

- `min_frequency` drops candidates that occur fewer times in the corpus. Many methods rank candidates that occur only once highly, and most term extractors, such as ATR4S, use `min_frequency=2`. The default of 1 keeps every candidate.
- `word_pattern` is a regular expression that every word of a candidate must match. The default, `"alphanumeric"`, drops candidates with a word that has no letter or digit, such as `95 % CI`, and `"atr4s"` is ATR4S's stricter noise filter, which only keeps words of three or more letters, digits and hyphens. Pass `None` to keep every word.

Like `pos_pattern`, these filters are applied before scoring: only the occurrences of candidates are counted, and a candidate that is too rare cannot make a longer one look nested in it for methods such as C-value.

### Lemmas

By default, `cell` and `cells` are different candidates, each with its own frequency. If your corpus has a lemma layer, such as the `lemma` layer of Teanga's spaCy service or of the ACL RD-TEC dataset, pass `lemma_layer="lemma"` to read every word as its lemma. Inflected forms are then counted together, and each term is returned as its lemmas, such as `cell` or `degree of freedom`, while `results.offsets` still gives the position of every form in the text. The lemma layer must be a sequence layer based on the words layer, and `pos_pattern` matches the lemmas' tags as before. To compare with gold terms, read those by their lemmas too, with `datasets.gold_terms(corpus, lemma_layer="lemma")`.

## Limitations and tips

- **Stopwords are only removed from the edges of candidates.** Other general words, such as `converts` or `updates`, still score highly when they are rare in your corpus. Use a larger corpus, add them to `stopwords`, use a [part-of-speech pattern](#part-of-speech-patterns), or try a method that compares with a reference corpus.
- **Use enough documents.** With very few documents, IDF can only take a handful of values and many terms tie.
- **Compare against a reference corpus.** To find terms typical of a *domain* rather than of individual documents, try the [reference corpus methods](reference_corpus_tutorial.md), which compare your corpus with general-language text.

See the [TF-IDF API reference](../api/unsupervised/tfidf.md) for the full list of parameters and methods.
