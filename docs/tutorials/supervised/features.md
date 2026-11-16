# Features

This page uses the `corpus`, `labels` and `corpus_ranking` of the [overview](../supervised_tutorial.md#how-supervised-extraction-works).

The features method (`method="features"`) follows HAMLET (Rigouts Terryn et al., 2021): it describes each candidate by many **features** and trains a **classifier**, a random forest by default, to tell the terms from the other candidates. The features are the scores of the unsupervised methods (TF-IDF, C-value, Basic and so on), Roger's association measures, the frequency of the candidate, its shape (length, digits, hyphens, case) and, for a Teanga corpus with part-of-speech tags, its tags. Its candidates are those of the unsupervised methods, so it takes their `min_frequency` and `word_pattern`, and it ranks single words too:

```python
terms = {term: 1 for term, label in labels.items() if label == 1}
extractor = SupervisedTermExtractor(method="features", n=2, random_state=0)
results = extractor.extract(corpus, terms)

for term, score in corpus_ranking(results):
    print(f"{score:.3f}  {term}")
```

```text
0.759  light energy
0.748  chemical energy
0.668  photosynthesis
0.661  mitochondria
0.652  chloroplast
0.620  takes place
0.620  cellular respiration
0.544  respiration
0.525  glucose
```

The scores are the probability the classifier gives each candidate of being a term, and by default only the candidates it classifies as terms, with a probability of at least 0.5, are returned. Pass `threshold=None` to rank every candidate, or another threshold to trade recall for precision.

`feature_importances()` shows which features the classifier relies on, here the length of the candidate and its TF-IDF and C-value scores:

```python
for name, importance in list(extractor.feature_importances().items())[:3]:
    print(f"{importance:.3f}  {name}")
```

```text
0.216  shape:characters
0.124  termhood:tfidf
0.098  termhood:cvalue
```

Some options:

- `reference_corpus` adds the scores of relevance, domain pertinence and weirdness, which compare the corpus with general-language text. On ACTER, it improves both the ranking and the classification.
- `pos_pattern="learned"`, for a Teanga corpus with part-of-speech tags, takes as candidates only the sequences of words whose tags are those of a term of the training corpus, as HAMLET does.
- `classifier` takes any scikit-learn classifier with `predict_proba`, such as `LogisticRegression()`, in place of the random forest.


See the [Features API reference](../../api/supervised/features.md) for all parameters.
