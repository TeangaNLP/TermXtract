# Supervised Methods Tutorial

This tutorial shows how to train a term extractor from examples with `SupervisedTermExtractor`. It covers the format of the training labels, and the two supervised methods: **Roger** and **token classification**.

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
- **Unlabelled candidates are not used for training.** Roger learns only from the candidates you label (see [Roger](#roger)), and token classification only from the terms labelled `1`.
- **Stopwords are filtered from the results, not from training.** As for the unsupervised methods, extracted terms that start or end with a stopword are dropped (see [Stopwords](tfidf_tutorial.md#stopwords)), but candidates such as `the` and `of` are still used as training examples. Pass `stopwords=None` to keep them in the results.

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

The supervised methods score each candidate once for the whole corpus, and each document's entry in `results.terms` lists the candidates that occur in that document, with their corpus-level scores. To see the ranking of the whole corpus, merge the documents:

```python
def corpus_ranking(results):
    """Merge the terms of all documents into one ranking."""
    scores = {t["term"]: t["score"] for doc in results.terms for t in doc["terms"]}
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
```

## Roger

The Roger method (Azé et al., 2005) learns a **ranking function** instead of a classifier. It describes each multi-word candidate by twelve **association measures** between its first words and its last word, such as mutual information, the Dice coefficient, log-likelihood and the *t*-test. These measures compare how often the words occur together with how often they occur apart, so candidates that occur equally often, such as `light energy` and `takes place`, can still be told apart.

An evolutionary algorithm then searches for the function of these measures that ranks the labelled terms above the labelled non-terms, by maximising the area under the ROC curve. It is run several times (`runs`, default 21), and each candidate's score is the median of its scores under the learned functions.

Two things to know about Roger:

- It **ranks only multi-word candidates**, from 2 to `n` words, because the association measures need two parts. Single words, and labels for them, are ignored, and `n` must be at least 2.
- It **trains only on the labelled candidates**, so you need at least one multi-word candidate labelled `1` and one labelled `0`. Unlabelled candidates are scored, but not used for training.

The labels above have only one multi-word non-term, `takes place`, which is too few to learn from, so the example below adds three more. The evolutionary search is random, so it also passes `random_state` to get the same scores on every run:

```python
roger_labels = {
    **labels,
    "photosynthesis takes": 0,
    "absorbs light": 0,
    "energy stored": 0,
}
extractor = SupervisedTermExtractor(method="roger", n=2, random_state=0)
results = extractor.extract(corpus, roger_labels)

for term, score in corpus_ranking(results)[:8]:
    print(f"{score:.3f}  {term}")
```

```text
0.067  light energy
0.067  chemical energy
0.011  chloroplast contains
0.011  plant cells
0.011  mitochondria release
0.009  takes place
0.009  cellular respiration
-0.010  photosynthesis takes
```

The labelled terms `light energy` and `chemical energy` come first, and the labelled non-terms `photosynthesis takes`, `absorbs light` and `energy stored` rank lower. But `takes place` ties with the term `cellular respiration`: each occurs twice, and its words never occur apart, so every association measure has the same value for both. On a corpus this small many candidates have the same counts, so Roger needs a larger corpus to rank well.

`generations` (default 100) is the number of rounds of the evolutionary search in each run, `population_size` (default 20) the number of ranking functions kept from one round to the next, and `offspring_size` (default 100) the number of new functions tried in each round. Fewer generations or runs train faster, and so do these options:

- `n_jobs` trains the runs in parallel: `n_jobs=-1` uses every CPU.
- `max_negatives` (default 10,000) trains each run on at most this many candidates labelled `0`, sampled at random, and on every candidate labelled `1`. It only matters for large sets of labels, such as when every candidate of a corpus is labelled. The area under the ROC curve does not depend on the share of positives, so the sample changes the learned functions little.
- `patience` stops a run early when its best function has not improved for that many generations.

Roger's scores are only meaningful for **ranking**: their scale depends on the learned functions.

### Scoring new documents

`extract()` always trains a new model. To apply a model you have already trained to other documents, call `predict_terms_strings` on the underlying extractor. It returns a dictionary of scores, which you can rank yourself:

```python
new_documents = [
    "Algae use chlorophyll for photosynthesis in the ocean.",
    "Glucose is broken down during cellular respiration.",
]
scores = extractor.extractor.predict_terms_strings(new_documents)
ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
for term, score in ranked[:3]:
    print(f"{score:.3f}  {term}")
```

```text
-0.034  algae use
-0.034  use chlorophyll
-0.034  chlorophyll for
```

Here every candidate ties, `cellular respiration` included. In two short sentences each pair of words occurs once, and so does each of its words, so all the association measures are the same. Like training, scoring needs documents large enough for the counts to differ.

## Token classification

The token-classification method fine-tunes a [Hugging Face](https://huggingface.co/) transformer model to tag each token as the beginning of a term, the continuation of a term or not part of a term. Unlike Roger, it learns from the **context** in which terms appear, and can find terms that are not in your labels.

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
- **Each document lists the terms the model tagged in it.**
- `n` (default 6) is the longest term, in words, that is tagged from the labels. `max_length` (default 512) is the maximum number of model tokens per sentence; longer sentences are truncated.
- `language` (default `"en"`) selects the [Moses](https://github.com/hplt-project/sacremoses) tokenizer used to split sentences into words.
- Training saves no checkpoints: the trained model is kept in memory.

Any Hugging Face model that supports token classification can be used as `model_name`. A model pre-trained on text from your domain usually works best.

## Using a Teanga corpus

Both methods also accept a [Teanga](https://github.com/TeangaNLP/teanga) `Corpus` with a `words` layer in place of the list of strings, with labels in the same format. Unlike the unsupervised methods, the supervised methods do not return term offsets.

See the API reference for [`SupervisedTermExtractor`](../api/supervised_term_extractor.md), [Roger](../api/supervised/roger.md) and [Token Classification](../api/supervised/token_classification.md) for all parameters.
