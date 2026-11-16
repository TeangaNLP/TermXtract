# Embedding Methods

This tutorial covers the methods that compare the **meanings** of words and phrases, by their embeddings from a sentence encoder: **EmbedRank** and **MDERank**, which score candidates by how close they are to the meaning of their documents, and the **re-rankers** AdaText and SemRe-Rank, which revise any method's ranking by how related its terms are to some seed terms.

They need the optional `transformers` dependencies (PyTorch and Hugging Face Transformers):

```bash
pip install 'termxtract[transformers]'
```

The first run downloads the encoder, by default [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (about 90 MB). It runs on a CPU, but a GPU is much faster on a large corpus.

## Setup

We use the corpus of the [C-value tutorial](cvalue_tutorial.md), on machine learning:

```python
from termxtract import TermExtractor
from termxtract.embeddings import Encoder

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


def show_terms(results, limit=6):
    """Rank each term by its highest score in any document."""
    scores = {}
    for doc in results.terms:
        for t in doc["terms"]:
            scores[t["term"]] = max(t["score"], scores.get(t["term"], t["score"]))
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for term, score in ranked[:limit]:
        print(f"{score:.3f}  {term}")
```

Each extractor loads its own encoder unless it is given one, so we load one `Encoder` and pass it to all of them. `Encoder` also sets the model (`model_name`), the batch size, the maximum length of a text in tokens, and the device:

```python
encoder = Encoder()
```

## EmbedRank

**EmbedRank** (Bennani-Smires et al., 2018) embeds each document and each candidate, and scores a candidate in a document by the cosine similarity of their embeddings. A candidate that sums up what the document is about scores highly. Its candidates are those of the other methods, with up to `n` (default 3) words:

```python
extractor = TermExtractor(method="embedrank", n=2, min_frequency=2, encoder=encoder)
results = extractor.extract(corpus)
show_terms(results)
```

```text
0.703  random forest
0.693  gradient descent
0.678  decision tree
0.636  neural network
0.547  training data
0.547  gradient
```

The scores of a term in the documents it occurs in are combined by `corpus_aggregation`: by default (`"max"`), each term gets its best score in any document, in all of them. `"mean"` takes the mean, `"centroid"` scores each candidate by its similarity to the mean of the document embeddings, and `None` keeps each document's own scores.

### Diversity

The best candidates of a document are often similar to each other. With `diversity`, EmbedRank re-ranks them by **Maximal Marginal Relevance**, as EmbedRank++ does: the candidates are taken one at a time, each time the one with the highest `(1 - diversity) × similarity to the document - diversity × highest similarity to a candidate already taken`. With `diversity=0.5`, `decision tree` falls below `trained`, since it is similar to `random forest`:

```python
extractor = TermExtractor(
    method="embedrank", n=2, min_frequency=2, encoder=encoder, diversity=0.5
)
results = extractor.extract(corpus)
show_terms(results)
```

```text
0.352  random forest
0.182  gradient descent
0.066  trained
0.043  neural network
0.006  decision tree
-0.003  weights
```

For term extraction, a small weight is best: on the benchmarks, `diversity=0.2` raised EmbedRank's average precision on all three English datasets, and made it the best method on the ACL RD-TEC.

## MDERank

**MDERank** (Zhang et al., 2022) turns EmbedRank around. It masks every occurrence of a candidate in the document, re-embeds the document, and scores the candidate by how far the embedding moved: `1 - cos(d, d_masked)`. A candidate whose removal changes the meaning of the document most scores best. Pass `variant="mderank"`:

```python
extractor = TermExtractor(
    method="embedrank", variant="mderank", n=2, min_frequency=2, encoder=encoder
)
results = extractor.extract(corpus)
show_terms(results)
```

```text
0.710  decision tree
0.655  random forest
0.638  training data
0.608  gradient descent
0.590  neural network
0.453  tree
```

MDERank embeds each document once for every candidate in it, so it is much slower than EmbedRank: on the benchmarks, 73 seconds for the 2,000 abstracts of GENIA on a GPU, against 6 for EmbedRank. It is also much better: among the best unsupervised methods on English and Irish, and ahead of EmbedRank on every dataset but the ACL RD-TEC.

Long documents are split into chunks of at most the encoder's maximum length, and a document's embedding is the mean of its chunks'. For languages other than English, use a multilingual encoder, such as `Encoder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")`.

## Re-ranking with related words

Any method's ranking can be revised with a re-ranker from `termxtract.rerank`, passed to `TermExtractor` as `rerank`. A re-ranker raises the terms whose words are related to some **seed terms** that you know belong to the domain, by a score from a graph of the corpus's words. Here TF-IDF ranks the words of `random forest` first, along with words such as `trained` and `one`:

```python
extractor = TermExtractor(method="tfidf", n=2, min_frequency=2)
show_terms(extractor.extract(corpus))
```

```text
0.052  random
0.052  random forest
0.052  forest
0.043  trained
0.043  one
0.043  time
```

With `decision tree` as the seed, SemRe-Rank raises the words related to it, such as `forest`, above the others:

```python
from termxtract.rerank import SemReRank

reranker = SemReRank(seeds=["decision tree"], encoder=encoder)
extractor = TermExtractor(method="tfidf", n=2, min_frequency=2, rerank=reranker)
show_terms(extractor.extract(corpus))
```

```text
1.413  decision
1.353  forest
1.182  decision tree
1.177  random forest
1.139  tree
1.000  random
```

There are two re-rankers:

- `SemReRank` (Zhang, Gao & Ciravegna, 2018) links each document's words to the words most similar to them, and scores them by PageRank, jumping to the words of the seeds. It divides the method's scores by the highest, so the new scores are at most 2.
- `AdaText` (Zhang, Petrak & Maynard, 2018) runs TextRank over the words of the corpus that are more similar than `threshold` (default 0.7) to a seed. Its seeds can be any phrases of the domain, such as the entries of a glossary, even if they do not occur in the corpus.

With `seeds=200`, the seeds are the 200 most frequent candidates, so no seeds are needed, but then the most frequent non-terms are seeds too. On the ACL RD-TEC and GENIA, with the terms among the 200 most frequent candidates as seeds, re-ranking raised the precision of most methods by up to 4 points, and lowered a few by up to 1.5. With the 200 most frequent candidates, it helped less (see the [benchmarks](https://github.com/TeangaNLP/TermXtract/blob/main/BENCHMARKS.md#re-ranking)).

See the API reference for [EmbedRank](../api/unsupervised/embedrank.md), [Embeddings](../api/embeddings.md) and [Re-ranking](../api/rerank.md) for all parameters.
