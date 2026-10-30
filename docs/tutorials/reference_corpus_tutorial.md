# Reference Corpus Tutorial

This tutorial covers the three methods that find terms by comparing your documents with a **reference corpus** of general-language text: **Weirdness**, **Domain Pertinence** and **Relevance**. You will learn what each one measures, how to pass the reference corpus, and how to read and filter the scores.

## Why a reference corpus?

Methods such as [TF-IDF](tfidf_tutorial.md) only look at your own documents, so they find what is distinctive about *each document*. A term that runs through your whole collection, such as `photosynthesis` in a set of biology texts, is penalised for being everywhere.

The reference-corpus methods ask a different question: **which words are much more common in my domain than in ordinary language?** To answer it, they need a sample of ordinary language: news, fiction, web text or anything else that is not about your domain. The words that are frequent in your documents but rare in the reference corpus are the domain terms.

All three methods count each n-gram in both corpora and turn the counts into **relative frequencies**, `p(t) = count(t) / number of n-grams in the corpus`, so that corpora of different sizes can be compared.

## Setup

```python
from termxtract import TermExtractor
```

Our domain corpus is five sentences about photosynthesis:

```python
corpus = [
    "Photosynthesis takes place in the chloroplast of a plant cell.",
    "The chloroplast contains chlorophyll, which absorbs light energy.",
    "During photosynthesis, light energy is converted into chemical energy.",
    "Plant cells store the chemical energy as glucose.",
    "Chlorophyll gives the leaf of a plant its green colour.",
]
```

The reference corpus is everyday text. It is a **list of strings** like the domain corpus; the reference corpus must always have the same type as the corpus you extract terms from.

```python
reference = [
    "The train was late, so we waited on the platform for an hour.",
    "She turned on the light and read a book in the kitchen.",
    "The shop on the corner sells bread, milk and green apples.",
    "We planted a tree in the garden at the start of spring.",
    "He needs a lot of energy in the morning, so he drinks coffee.",
    "The children played in the park until it was dark.",
    "Most of the town takes its water from a lake, which is in the hills.",
    "The film was converted into a play during the summer.",
]
```

By default, every method drops candidates that start or end with a stopword, such as `the` or `of` (see [Stopwords](tfidf_tutorial.md#stopwords) in the TF-IDF tutorial). The examples below turn this off with `stopwords=None`, so that you can see how each method scores function words on its own. In practice, keep the default.

These methods score each n-gram over the **whole corpus**, so a term has the same score in every document it occurs in. `results.terms` still lists terms per document, as for every method. To see the ranking for the whole corpus, we collect the scores from all documents with this helper:

```python
def show_terms(results, limit=None):
    scores = {}
    for doc in results.terms:
        for term in doc["terms"]:
            scores[term["term"]] = term["score"]
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for term, score in ranked[:limit]:
        print(f"{score:.3f}  {term}")
```

## Weirdness

**Weirdness** (Ahmad et al., 1999) is the ratio of an n-gram's relative frequency in your corpus to its relative frequency in the reference corpus:

`weirdness(t) = p_target(t) / p_ref(t)`

A score of 4 means the n-gram is four times more common in your documents than in general language. Many terms never appear in the reference corpus at all, so TermXtract adds one to every reference count, `p_ref(t) = (count_ref(t) + 1) / N_ref`, to avoid dividing by zero.

Pass the reference corpus with `reference_corpus`:

```python
extractor = TermExtractor(method="weirdness", reference_corpus=reference, stopwords=None)
results = extractor.extract(corpus)
show_terms(results, limit=8)
```

```text
6.400  plant
4.267  photosynthesis
4.267  chloroplast
4.267  chlorophyll
4.267  energy
4.267  chemical
2.133  place
2.133  cell
```

The domain terms come first. `plant` scores highest because it occurs three times in our corpus (as `Plant` and `plant`; text is lower-cased) and never in the reference corpus. `energy` also occurs in the reference corpus, but its four occurrences in our corpus put it level with `photosynthesis`, `chloroplast`, `chlorophyll` and `chemical`, which occur twice each and never in the reference corpus.

At the bottom of the ranking are the words that are *more* common in the reference corpus, with a score below 1:

```python
show_terms(results)
```

```text
6.400  plant
4.267  photosynthesis
4.267  chloroplast
4.267  chlorophyll
4.267  energy
4.267  chemical
2.133  place
2.133  cell
2.133  contains
2.133  absorbs
2.133  light
2.133  cells
2.133  store
2.133  as
2.133  glucose
2.133  gives
2.133  leaf
2.133  colour
1.067  takes
1.067  of
1.067  which
1.067  during
1.067  is
1.067  converted
1.067  into
1.067  its
1.067  green
0.711  a
0.533  the
0.356  in
```

The words in the middle show the main limitation of a small reference corpus. Function words such as `of`, `which` and `is` score just above 1, as if they were slightly domain-specific, because eight reference sentences are too few to show how common they are. With a reference corpus of thousands of documents, function words are frequent in both corpora and score close to or below 1.

## Domain Pertinence

**Domain Pertinence** (Navigli & Velardi, 2004) compares each n-gram's relative frequency in your corpus with the higher of its relative frequencies in the two corpora:

`domain_pertinence(t) = p_target(t) / max(p_target(t), p_ref(t))`

The score is between 0 and 1. It is **1** for every n-gram that is at least as common in your corpus as in the reference corpus, and falls towards 0 for n-grams that are more typical of the reference corpus. It does not rank domain terms against each other; it is a **filter** that removes words belonging to general language. Use it with `threshold=1`:

```python
extractor = TermExtractor(
    method="domainpertinence", reference_corpus=reference, threshold=1, stopwords=None
)
results = extractor.extract(corpus)
print(results.terms[0])
```

```text
{'doc_id': 'doc_0', 'terms': [{'term': 'photosynthesis', 'score': 1.0}, {'term': 'takes', 'score': 1.0}, {'term': 'place', 'score': 1.0}, {'term': 'chloroplast', 'score': 1.0}, {'term': 'of', 'score': 1.0}, {'term': 'plant', 'score': 1.0}, {'term': 'cell', 'score': 1.0}]}
```

The general-language words `in`, `the` and `a` are gone from the first document, and everything else is kept with a score of 1. Without a threshold, you can see how strongly the filtered words belong to the reference corpus:

```python
extractor = TermExtractor(
    method="domainpertinence", reference_corpus=reference, stopwords=None
)
results = extractor.extract(corpus)
show_terms(results, limit=None)
```

```text
1.000  photosynthesis
1.000  takes
1.000  place
1.000  chloroplast
1.000  of
1.000  plant
1.000  cell
1.000  contains
1.000  chlorophyll
1.000  which
1.000  absorbs
1.000  light
1.000  energy
1.000  during
1.000  is
1.000  converted
1.000  into
1.000  chemical
1.000  cells
1.000  store
1.000  as
1.000  glucose
1.000  gives
1.000  leaf
1.000  its
1.000  green
1.000  colour
0.853  a
0.569  the
0.427  in
```

Like Weirdness, Domain Pertinence needs a large reference corpus to filter well. Here, `of`, `which` and `is` pass the filter for the same reason as before.

## Relevance

**Relevance** (Peñas et al., 2001) also rewards terms that are **spread across your documents**, not just frequent:

`relevance(t) = 1 - 1 / log2(2 + p_target(t) * df_target(t) / p_ref(t))`

where `df_target(t)` is the fraction of your documents that contain the n-gram. The reference count is smoothed as for Weirdness. Scores are between 0 and 1.

```python
extractor = TermExtractor(method="relevance", reference_corpus=reference, stopwords=None)
results = extractor.extract(corpus)
show_terms(results, limit=10)
```

```text
0.607  plant
0.543  energy
0.471  photosynthesis
0.471  chloroplast
0.471  chlorophyll
0.471  chemical
0.339  light
0.218  place
0.218  the
0.218  of
```

`light` now ranks above other words that occur twice, such as `cells` and `leaf`, because it occurs in two documents rather than one. The flip side is that a word found in almost every document gets a boost too: `the` occurs in four of our five documents, which is enough to lift it to the level of `cell` and `glucose` despite being common in the reference corpus. As before, a larger reference corpus fixes this, because the reference frequency of `the` would then be much more reliable.

## Choosing a method

| Method | Score | Good for |
|---|---|---|
| Weirdness | Ratio of frequencies, from 0 upwards | Ranking candidates by how domain-specific they are |
| Domain Pertinence | 0 to 1, and 1 for most domain terms | Filtering out general-language words with `threshold=1` |
| Relevance | 0 to 1 | Ranking terms that are both domain-specific and widespread in your corpus |

## Multi-word terms

Set `n` to score n-grams up to that length. The reference corpus is split into n-grams in the same way:

```python
extractor = TermExtractor(
    method="weirdness", reference_corpus=reference, n=2, stopwords=None
)
results = extractor.extract(corpus)
print([term["term"] for term in results.terms[2]["terms"]][:6])
```

```text
['photosynthesis', 'light energy', 'energy', 'chemical', 'chemical energy', 'during photosynthesis']
```

Most two-word phrases never occur in the reference corpus, so phrases such as `light energy` and `chemical energy` score highly as long as they occur more than once. As with TF-IDF, fragments such as `during photosynthesis` score highly too; the reference-corpus methods know nothing about phrase structure.

## Using a Teanga corpus

With a [Teanga](https://github.com/TeangaNLP/teanga) `Corpus`, the reference corpus must also be a Teanga `Corpus` with a `words` layer, and the results include the character offsets of each term:

```python
import re
from teanga import Corpus


def to_teanga(texts):
    teanga_corpus = Corpus()
    teanga_corpus.add_layer_meta("text")
    teanga_corpus.add_layer_meta("words", layer_type="span", base="text")
    for text in texts:
        doc = teanga_corpus.add_doc(text)
        doc.words = [match.span() for match in re.finditer(r"\w+", text)]
    return teanga_corpus


extractor = TermExtractor(
    method="weirdness",
    reference_corpus=to_teanga(reference),
    threshold=2,
    stopwords=None,
)
results = extractor.extract(to_teanga(corpus))
print(results.terms[1]["terms"])
print(results.offsets[1]["offsets"])
```

```text
[{'term': 'chloroplast', 'score': 4.2666666666666675}, {'term': 'energy', 'score': 4.2666666666666675}, {'term': 'contains', 'score': 2.1333333333333337}, {'term': 'chlorophyll', 'score': 2.1333333333333337}, {'term': 'absorbs', 'score': 2.1333333333333337}, {'term': 'light', 'score': 2.1333333333333337}]
[{'term': 'chloroplast', 'offset': (4, 15)}, {'term': 'contains', 'offset': (16, 24)}, {'term': 'chlorophyll', 'offset': (25, 36)}, {'term': 'absorbs', 'offset': (44, 51)}, {'term': 'light', 'offset': (52, 57)}, {'term': 'energy', 'offset': (58, 64)}]
```

Teanga words are used exactly as they appear in the `words` layer, so **case is preserved**. Here `chlorophyll` scores 2.133 rather than 4.267, because the `Chlorophyll` at the start of the last document counts as a different word. Lower-case both corpora's text first, or build the `words` layer from a normalised layer, if that matters for your data.

Mixing the two types raises a `TypeError`:

```python
try:
    TermExtractor(method="weirdness", reference_corpus=reference).extract(
        to_teanga(corpus)
    )
except TypeError as error:
    print(error)
```

```text
The reference corpus is a list of strings, so the corpus to extract terms from must also be a list of strings, not a Teanga Corpus.
```

## Tips

- **Use a large, general reference corpus.** Most problems in this tutorial come from a reference corpus of eight sentences. A few thousand documents of news or web text, in the same language as your corpus, gives far more reliable frequencies. The [Brown corpus](https://www.nltk.org/nltk_data/) or a sample of Wikipedia are common choices.
- **Keep the tokenization the same.** Plain-text input is lower-cased and split on word characters for both corpora. For Teanga input, build both `words` layers the same way.
- **Keep the default stopword filtering.** It removes the function words that a small reference corpus fails to penalise, such as `of`, `which` and `is` in this tutorial, before the scores are even looked at.

See the API reference for [Weirdness](../api/unsupervised/weirdness.md), [Domain Pertinence](../api/unsupervised/domainpertinence.md) and [Relevance](../api/unsupervised/relevance.md) for all parameters.
