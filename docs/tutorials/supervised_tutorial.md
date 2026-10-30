# Supervised Methods Tutorial

This tutorial shows how to train a term extractor from examples with `SupervisedTermExtractor`. It covers the format of the training labels, and the three supervised methods: **AdaBoost**, **Roger** and **token classification**.

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
- **Unlabelled candidates count as non-terms** when training AdaBoost and Roger. Label as many real terms as you can, or the model learns that your unlabelled terms are not terms.

`extract(corpus, labels)` trains the model on the corpus and then scores the candidates in the **same** corpus. The examples below use this small biology corpus:

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

## AdaBoost

The AdaBoost method trains an [AdaBoost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost) classifier of decision trees on three features of each candidate:

- **Semantic content:** the number of [WordNet](https://wordnet.princeton.edu/) senses of the candidate, divided by its number of words plus one. Ordinary words have several senses; most technical terms have one or none.
- **Greek and Latin morphemes:** whether the candidate contains `bio`, `neuro`, `cardio`, `psycho`, `pharm`, `ology`, `itis`, `oma` or `osis`.
- **Collocation strength:** an MI3 score based on how frequent the candidate is in the corpus.

The WordNet feature uses the [`wn`](https://github.com/goodmami/wn) package and Open English WordNet. The lexicon is **downloaded automatically on first use** (about 13 MB). To download it in advance, for example on a machine that will be offline, run `python -m wn download oewn:2025+`.

```python
extractor = SupervisedTermExtractor(method="adaboost", n=2)
results = extractor.extract(corpus, labels)

for term in results.terms[0]["terms"][:10]:
    print(f"{term['score']:.3f}  {term['term']}")
```

```text
0.681  photosynthesis
0.681  chloroplast
0.681  chemical energy
0.681  glucose
0.681  mitochondria
0.681  cellular respiration
0.580  plant cell
0.580  chlorophyll
0.580  plant cells
0.465  takes place
```

The score is the classifier's **probability that the candidate is a term**, between 0 and 1. Most of the labelled terms come first, and the phrases labelled `0`, such as `takes place`, rank lower. But the labelled term `light energy` also scores 0.465, the same as `takes place`: neither is in WordNet, neither has a Greek or Latin morpheme, and both occur twice, so the features cannot tell them apart. Each feature takes only a few values on a corpus this small, so many candidates share a score.

The supervised methods score each candidate once for the whole corpus. At present, every document's entry in `results.terms` holds the **same ranking of all the candidates in the corpus**, not just the candidates in that document, so it is enough to read `results.terms[0]`.

Use `threshold` to keep only candidates the classifier considers more likely than not to be terms:

```python
extractor = SupervisedTermExtractor(method="adaboost", n=2, threshold=0.5)
results = extractor.extract(corpus, labels)
print([term["term"] for term in results.terms[0]["terms"]])
```

```text
['photosynthesis', 'chloroplast', 'chemical energy', 'glucose', 'mitochondria', 'cellular respiration', 'plant cell', 'chlorophyll', 'plant cells']
```

### Scoring new documents

`extract()` always trains a new model. To apply a model you have already trained to other documents, call `predict_terms_strings` on the underlying extractor. It returns a dictionary of scores, which you can rank yourself:

```python
new_documents = [
    "Algae use chlorophyll for photosynthesis in the ocean.",
    "Glucose is broken down during cellular respiration.",
]
scores = extractor.extractor.predict_terms_strings(new_documents)
ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
for term, score in ranked[:6]:
    print(f"{score:.3f}  {term}")
```

```text
0.661  algae
0.661  chlorophyll
0.661  photosynthesis
0.661  glucose
0.661  cellular respiration
0.443  algae use
```

The model recognises the terms it was trained on, and also scores the new word `algae` highly: it has a single WordNet sense, like the technical terms in the training labels.

## Roger

The Roger method (Azé et al., 2005) learns a **ranking function** instead of a classifier. It computes statistical measures of each candidate, such as its frequency, mutual information and *t*-test score, and uses an evolutionary algorithm to find the weighted sum of these measures that best ranks the labelled terms above the labelled non-terms. It repeats this on random subsamples of the candidates (bagging) and scores each candidate by the median of the learned functions.

The evolutionary search is random. Set NumPy's random seed if you need the same scores on every run:

```python
import numpy as np

np.random.seed(0)
extractor = SupervisedTermExtractor(method="roger", n=2, generations=30)
results = extractor.extract(corpus, labels)

for term in results.terms[0]["terms"][:8]:
    print(f"{term['score']:.3f}  {term['term']}")
```

```text
-0.592  photosynthesis
-0.592  takes
-0.592  takes place
-0.592  place
-0.592  place in
-0.592  in the
-0.592  the chloroplast
-0.592  chloroplast
```

`generations` (default 100) is the number of rounds of the evolutionary search, and `population_size` (default 50) the number of candidate ranking functions in each round. Fewer generations train faster.

Roger's scores are only meaningful for **ranking**: their scale and sign depend on the learned weights.

**Limitation.** In the current implementation, every measure Roger uses is computed from the candidate's frequency alone, so Roger can only learn whether frequent candidates tend to be terms. All the candidates that occur equally often get the same score, which is why `takes place` and `in the` tie with `photosynthesis` above. On real data, prefer AdaBoost or token classification.

## Token classification

The token-classification method fine-tunes a [Hugging Face](https://huggingface.co/) transformer model to tag each token as the beginning of a term, the continuation of a term or not part of a term. Unlike AdaBoost and Roger, it learns from the **context** in which terms appear, and can find terms that are not in your labels.

It needs the optional `transformers` dependencies (PyTorch, transformers, accelerate and sacremoses):

```bash
pip install 'termxtract[transformers]'
```

The example below is not run when this documentation is built: the first run downloads the model (about 1 GB for `xlm-roberta-base`), and fine-tuning is slow without a GPU.

```python
from termxtract import SupervisedTermExtractor

extractor = SupervisedTermExtractor(
    method="token-classification",
    model_name="xlm-roberta-base",
    epochs=3,
    batch_size=8,
)
results = extractor.extract(corpus, labels)

for doc in results.terms:
    print(doc["doc_id"], [term["term"] for term in doc["terms"]])
```

Things to know about this method:

- **Only the terms labelled `1` are used.** Every occurrence of a labelled term in the corpus is tagged as a term, and all other tokens as non-terms, so the `0` labels make no difference. Labels are matched case-insensitively.
- **Every extracted term has a score of 1.0.** The model tags terms rather than scoring them, so `threshold` is not used.
- **Each document lists the extracted terms that occur in it.**
- `n` (default 6) is the longest term, in words, that is tagged from the labels. `max_length` (default 512) is the maximum number of model tokens per sentence; longer sentences are truncated.
- `language` (default `"en"`) selects the [Moses](https://github.com/hplt-project/sacremoses) tokenizer used to split sentences into words.
- Training writes checkpoints to `./results` and logs to `./logs` in the current directory.

Any Hugging Face model that supports token classification can be used as `model_name`. A model pre-trained on text from your domain usually works best.

## Using a Teanga corpus

All three methods also accept a [Teanga](https://github.com/TeangaNLP/teanga) `Corpus` with a `words` layer in place of the list of strings, with labels in the same format. Unlike the unsupervised methods, the supervised methods do not return term offsets.

See the API reference for [`SupervisedTermExtractor`](../api/supervised_term_extractor.md), [AdaBoost](../api/supervised/adaboost.md), [Roger](../api/supervised/roger.md) and [Token Classification](../api/supervised/token_classification.md) for all parameters.
