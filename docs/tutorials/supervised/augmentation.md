# Synthetic Training Data

This page uses the `corpus` and `labels` of the [overview](../supervised_tutorial.md#how-supervised-extraction-works).

When few sentences are labelled, `termxtract.augmentation.augment` can add synthetic ones, as in Banerjee (2025, ch. 5). It chooses up to `seeds` (default 40) of the labelled sentences, by default those that together have the most distinct terms, and writes `n` (default 50) new sentences from them by one of four methods:

- `"term"`: the model rewrites the seeds with other terms of the domain.
- `"context"`: the model writes new sentences that use the terms of the seeds.
- `"combined"`: the model writes new sentences with new terms.
- `"wiki"`: each term of a seed is replaced by a similar term from a lexicon, with no model.

It returns the training data with the new sentences added: a list of strings with the labels of their terms, or, for a Teanga corpus, a new corpus with one document per sentence. Either can be passed to `fit` for any supervised method. The example below, not run when this documentation is built, writes 50 sentences with a model served by Ollama and fine-tunes token classification on them and the gold sentences:

```python
from termxtract import SupervisedTermExtractor
from termxtract.augmentation import augment
from termxtract.llm import OpenAIClient

client = OpenAIClient(
    "qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    cache_dir="llm-cache",
    max_tokens=1000,
)
texts, augmented_labels = augment(
    corpus, labels, method="term", client=client, domain="heart failure"
)
extractor = SupervisedTermExtractor(method="token-classification")
extractor.fit(texts, augmented_labels)
```

The prompts are those of the thesis. They show the seeds with their terms in `<tag>` tags, and the model tags the terms of the sentences it writes in the same way, except for `"context"`, whose sentences are annotated with the terms of the seeds they contain. Each prompt asks for `per_prompt` sentences (default 10), so the client's `max_tokens` must leave room for them: about 100 tokens a sentence. The seeds are shuffled for each prompt, so that each prompt is different and its answer can be cached. Sentences with unbalanced tags, without terms, or that repeat a sentence are dropped.

For `"wiki"`, `termxtract.wikipedia.lexicon` builds the lexicon as in the thesis's Algorithm 1. It reads the links of the articles in the categories of the domain's article, up to 2,000, and maps each gold term to the links whose embeddings are similar to its own:

```python
from termxtract.wikipedia import WikipediaAPI, WikipediaDump, lexicon

terms = [term for term, label in labels.items() if label == 1]
replacements = lexicon(WikipediaAPI(cache_dir="wiki-cache"), "Heart failure", terms)
texts, augmented_labels = augment(corpus, labels, method="wiki", lexicon=replacements)
```

`WikipediaAPI` sends 10 to 20 requests for a domain, and the result changes as Wikipedia does. `WikipediaDump` reads a fixed snapshot from the SQL dumps of a Wikipedia, about 14 GB for English, which `download()` fetches. It reads each dump several times for each call to `candidates`, so pass every domain's article to one call; the results are cached.

In the thesis, with GPT-4 on ACTER, every method improved on fine-tuning XLM-RoBERTa alone, by 18 to 32 points of F1 with 5 training sentences. With TermXtract, `gemma4:26b` and token classification trained for 50 epochs, which alone does 30 to 36 points better than the thesis's baseline, augmentation did not reliably help: each method was between 1.5 points worse and 0.7 better on average over 5 to 30 training sentences, though on wind energy, where recall is low, it added 5 to 7 points of F1 with 5 and 15 sentences. Asking for 10 or for 50 sentences in each prompt made little difference, and in-context learning stayed ahead of every method (see the [benchmarks](https://github.com/TeangaNLP/TermXtract/blob/main/BENCHMARKS.md)). So try augmentation when a fine-tuned model misses many terms, and check it on held-out sentences.


See the API reference for [Data Augmentation](../../api/augmentation.md) and [Wikipedia](../../api/wikipedia.md) for all parameters.
