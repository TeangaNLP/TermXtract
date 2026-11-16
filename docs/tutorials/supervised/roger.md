# Roger

This page uses the `corpus`, `labels` and `corpus_ranking` of the [overview](../supervised_tutorial.md#how-supervised-extraction-works).

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


## Scoring new documents

As for every supervised method, `fit` trains a model and `predict` scores the candidates of other documents (see [Training on one corpus, extracting from another](../supervised_tutorial.md#training-on-one-corpus-extracting-from-another)). Here a Roger model trained on the biology corpus scores two new documents:

```python
extractor = SupervisedTermExtractor(method="roger", n=2, random_state=0)
extractor.fit(corpus, roger_labels)
new_documents = [
    "Algae use chlorophyll for photosynthesis in the ocean.",
    "Glucose is broken down during cellular respiration.",
]
results = extractor.predict(new_documents)
for term, score in corpus_ranking(results)[:3]:
    print(f"{score:.3f}  {term}")
```

```text
-0.034  algae use
-0.034  use chlorophyll
-0.034  cellular respiration
```

Here every candidate ties, the term `cellular respiration` included. In two short sentences each pair of words occurs once, and so does each of its words, so all the association measures are the same. Like training, scoring needs documents large enough for the counts to differ.


See the [Roger API reference](../../api/supervised/roger.md) for all parameters.
