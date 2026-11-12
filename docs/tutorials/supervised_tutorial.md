# Supervised Methods Tutorial

This tutorial shows how to train a term extractor from examples with `SupervisedTermExtractor`. It covers the format of the training labels, and the four supervised methods: **Roger**, the **features** classifier, **token classification** and **in-context learning** with a large language model.

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
- **Unlabelled candidates are only used by Roger and the features method, and only without non-terms.** They learn from the candidates you label, but if you label only terms, as with a list of gold terms, they count every other candidate as a non-term (see [Roger](#roger)). Token classification learns only from the terms labelled `1`.
- **Stopwords are filtered from the results, not from training.** As for the unsupervised methods, extracted terms that start or end with a stopword are dropped (see [Stopwords](tfidf_tutorial.md#stopwords)), but candidates such as `the` and `of` are still used as training examples by Roger. The features method takes the candidates of the unsupervised methods, so it neither trains on them nor returns them. Pass `stopwords=None` to keep them.

`extract(corpus, labels)` trains the model on the corpus and then scores the candidates in the **same** corpus. To extract terms from other documents, train with `fit` and then call `predict` (see [Training on one corpus, extracting from another](#training-on-one-corpus-extracting-from-another)). The examples below use this small biology corpus:

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

## Features

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

## Training on one corpus, extracting from another

`extract()` trains a new model and extracts terms from the corpus it was trained on. To train on one corpus, such as the training split of a dataset or another domain, and extract terms from another, call `fit` and then `predict`. `predict` returns `ATEResults` like `extract`, with the terms of each document and, for a Teanga corpus, their offsets. Here a Roger model trained on the biology corpus scores two new documents:

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

- **Only the terms labelled `1` are used.** Every occurrence of a labelled term in the corpus is tagged as a term, and all other tokens as non-terms, so the `0` labels make no difference. Labels are matched case-insensitively. Where labelled terms overlap, as `heart failure` and `acute heart failure` do, the longest is tagged.
- **Every extracted term has a score of 1.0.** The model tags terms rather than scoring them, so `threshold` is not used.
- **Each document lists the terms the model tagged in it.**
- `n` (default 6) is the longest term, in words, that is tagged from the labels. `max_length` (default 512) is the maximum number of model tokens per sentence; longer sentences are truncated.
- `language` (default `"en"`) selects the [Moses](https://github.com/hplt-project/sacremoses) tokenizer used to split sentences into words.
- Training saves no checkpoints: the trained model is kept in memory. To keep it, call `extractor.save(directory)`, and later pass the directory as `model_name` to extract terms without training again.
- `random_state` seeds training, so that the same data gives the same model.

To choose the number of epochs, pass a validation corpus to `fit`, labelled in the same way. The model is scored after each epoch by the F1 of the terms it tags in the validation corpus, and the best epoch is kept. With `patience`, training stops when the F1 has not improved for that many epochs:

```python
extractor = SupervisedTermExtractor(
    method="token-classification", epochs=10, patience=2, random_state=0
)
extractor.fit(train_corpus, labels, validation=validation_corpus)
```

Any Hugging Face model that supports token classification can be used as `model_name`. A model pre-trained on text from your domain usually works best.

## In-context learning with a large language model

The `llm` method trains nothing. It sends each sentence to a large language model in a prompt with `k` training sentences and their terms, the *demonstrations*, and reads the terms from the model's answer, as in Banerjee, Chakravarthi and McCrae (2024). With the default `selection="knn"`, the demonstrations are the training sentences most similar to the sentence, by the cosine of their [embeddings](../api/embeddings.md) (Liu et al., 2022); `selection="random"` samples them instead.

The model is reached through a client. `OpenAIClient` speaks the OpenAI API, which local servers such as [Ollama](https://ollama.com/), vLLM and llama.cpp also offer. It needs the `llm` extra, and the "knn" selection needs the `transformers` extra:

```bash
pip install 'termxtract[llm,transformers]'
```

The example below is not run when this documentation is built, as it needs a model server. Here it uses a model served by Ollama on this machine:

```python
from termxtract import SupervisedTermExtractor
from termxtract.llm import OpenAIClient

client = OpenAIClient(
    "qwen2.5:7b", base_url="http://localhost:11434/v1", cache_dir="llm-cache"
)
extractor = SupervisedTermExtractor(method="llm", client=client, k=5)
results = extractor.fit(corpus, labels).predict(["Acute heart failure is common."])
```

Each prompt looks like this, with the most similar demonstration last:

```text
Find the domain-specific terms in the following text.

Input: Patients with chronic heart failure often receive beta-blockers.
Output: chronic heart failure, heart failure, beta-blockers

Input: Acute heart failure is common.
Output:
```

Things to know about this method:

- **The model is asked at temperature 0**, so that the same prompt gets the same answer. With `cache_dir`, every answer is kept on disk, and a prompt that was answered before is not sent again.
- **Only the terms the model finds in the sentence are kept.** The answer is split at commas (or else at newlines and semicolons), and each term is looked for in the sentence, ignoring case and spacing. Terms that are not there, such as a term the model made up, are dropped, as are numbers, terms longer than `n` words (default 10), and terms that start or end with a stopword.
- **Every extracted term has a score of 1.0**, as for token classification.
- `template` picks one of the eight prompt templates of `termxtract.supervised.llm.TEMPLATES`, which pair four instructions with two kinds of markers, `Input:`/`Output:` and `Text:`/`Terminology:`. Template 5, the default, did best in the paper, by a point or two of F1. You can also pass your own `(instruction, input marker, output marker)` tuple.
- **A sentence is never its own demonstration**, so `extract` can find the terms of the training corpus without being shown them.
- `k=0` asks the model with no demonstrations at all.
- `n_jobs` sends several prompts at once. A local server only answers them in parallel if it is set up to, for example with `OLLAMA_NUM_PARALLEL` for Ollama.

In the paper, with GPT-3.5-Turbo on ACTER, in-context learning beat fine-tuning XLM-RoBERTa by far with 5 training sentences, and by less as the training set grew: by 30 sentences the two were close. More demonstrations helped in-context learning only a little, about 2 points of F1 from 5 to 30 sentences, and kNN selection beat random selection by 1 to 2 points. With TermXtract and the open model `gemma4:26b`, in-context learning beat fine-tuning at every size up to 30 sentences on all four domains of ACTER, and with all of ACTER's training data it came within 4 points of F1 of token classification (see the [benchmarks](https://github.com/TeangaNLP/TermXtract/blob/main/BENCHMARKS.md)). So in-context learning is the method to try when only a handful of sentences are labelled, and fine-tuning when there are many.

## Using a Teanga corpus

All the methods also accept a [Teanga](https://github.com/TeangaNLP/teanga) `Corpus` with a `words` layer in place of the list of strings, with labels in the same format. As for the unsupervised methods, the results then include the offsets of each extracted term.

If the corpus marks its gold terms in a `terms` span layer, as the corpora of `termxtract.datasets` do, `fit` can learn from them without labels. Token classification then tags each annotated occurrence, rather than every occurrence of a labelled term, Roger takes the annotated terms as terms and every other candidate as a non-term, and in-context learning gives each sentence with its annotated terms as a demonstration. Here a model is trained on three domains of ACTER and tags the fourth:

```python
from termxtract import datasets

train = datasets.load("acter", split="train")
test = datasets.load("acter", split="test")
extractor = SupervisedTermExtractor(method="token-classification", random_state=0)
results = extractor.fit(train).predict(test)
```

Token classification and in-context learning split a Teanga document into sentences by its `sentences` layer, if it has one, or else after each `.`, `!` and `?`.

If the corpus has a lemma layer, such as the `lemma` layer of RSDO5 or the ACL RD-TEC, pass `lemma_layer="lemma"` to give each term by its lemmas, so that `cell` and `cells` are one term, as for the unsupervised methods. Roger then computes its measures over the lemmas, while token classification and in-context learning still read the words as written and only name the terms they find by their lemmas. Gold terms of the `terms` layer are read by their lemmas too, but labels are matched to the lemmas as given, so write them as lemmas. To evaluate, read the gold terms in the same way, with `datasets.gold_terms(corpus, lemma_layer="lemma")`.

See the API reference for [`SupervisedTermExtractor`](../api/supervised_term_extractor.md), [Roger](../api/supervised/roger.md), [Token Classification](../api/supervised/token_classification.md) and [In-Context Learning](../api/supervised/llm.md) for all parameters.
